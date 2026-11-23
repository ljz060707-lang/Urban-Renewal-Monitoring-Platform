from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from shapely.geometry import box
from shapely.geometry import mapping
import fiona
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from xian_r5.alignment.field import DisplacementCell, LocalDisplacementField, group_fields
from xian_r5.alignment.field_io import (read_composed_alignment_fields,
                                        read_lightglue_local_field_candidates)
from xian_r5.alignment.field_io import read_direct_base_tie_fields
from xian_r5.alignment.field_io import (_lightglue_cell_tile_provenance,
                                        _base_tie_settings_signature,
                                        _lightglue_source_component_scope,
                                        _prepare_source_component_index,
                                        _source_component_scope_for_support,
                                        paired_source_component_scope_for_support)
from xian_r5.alignment.local_field import LOCAL_FIELD_METHOD, LOCAL_TRANSFORM_MODEL
from xian_r5.alignment.comparison import compare_alignment_fields
from xian_r5.alignment.mosaic_grid import load_mosaic_grids_by_source_epoch


def write_lightglue_fixture(
    root: Path, *, transform_model: str | None = None,
) -> tuple[Path, Path, str]:
    """Create a hash-consistent local-field receipt and candidate GeoPackage."""
    import hashlib

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    dates = ["2018-04-29", "2019-01-01", "2020-12-31"]
    base = root / "base.tif"
    mosaic = root / "mosaic.json"
    model = root / "model.json"
    weights = [root / "aliked.pth", root / "lightglue.pth"]
    for path, payload in ((base, b"synthetic base grid"),
                          (model, b"model receipt"), (weights[0], b"aliked weights"),
                          (weights[1], b"lightglue weights")):
        path.write_bytes(payload)

    grid = {
        "crs": "EPSG:32649", "tile_pixel_size": [256, 256],
        "tile_window": {"x0": 420796, "y0": 312172, "nx": 2, "ny": 2},
        "pixel_transform": [10 / 256, 0, 0, 0, -10 / 256, 10, 0, 0, 1],
        "bounds": [0, -10, 20, 10],
    }
    submeter_tiles = {}
    for actual_date in dates:
        epoch = actual_date[:4]
        submeter_tiles[actual_date] = [
            {"tile_id": f"{420796 + col}:{312172 + row}",
             "path": str(root / f"{actual_date}_{col}_{row}.jpg"),
             "selected_source_epoch": epoch,
             "sha256": hashlib.sha256(f"{actual_date}:{col}:{row}".encode()).hexdigest(),
             "duplicate_source_status": "SINGLE_EXACT_DATE_SOURCE"}
            for row in range(2) for col in range(2)]
    mosaic.write_text(json.dumps({
        "schema_version": "xian_r5.mosaic_registration_diagnostic.v1",
        "grid": grid, "submeter_tiles": submeter_tiles,
    }), encoding="utf-8")

    vector = root / "LOCAL_TILE_FIELD_CANDIDATES.gpkg"
    vector_properties = {
        "from_date": "str:10", "to_date": "str:10", "cell_id": "str:16",
        "status": "str:40", "raw_n": "int", "eligible_n": "int", "inlier_n": "int",
        "ratio": "float", "coverage": "float", "p90_px": "float", "p90_m": "float",
        "corr_dx_m": "float", "corr_dy_m": "float", "reviewed": "int", "applied": "int",
    }
    if transform_model is not None:
        vector_properties["transform_model"] = "str:64"
    schema = {"geometry": "Polygon", "properties": vector_properties}
    pair_values = [(dates[0], dates[1], 1.25, -0.5),
                   (dates[0], dates[2], 2.0, -1.0),
                   (dates[1], dates[2], 0.75, -0.25)]
    with fiona.open(vector, "w", driver="GPKG", layer="local_tile_field_candidates",
                    schema=schema, crs="EPSG:32649") as sink:
        for from_date, to_date, dx, dy in pair_values:
            properties = {
                "from_date": from_date, "to_date": to_date, "cell_id": "r00c00",
                "status": "LOCAL_CANDIDATE_NOT_REVIEWED", "raw_n": 20,
                "eligible_n": 12, "inlier_n": 7, "ratio": 0.58, "coverage": 0.2,
                "p90_px": 1.5, "p90_m": 0.05, "corr_dx_m": dx, "corr_dy_m": dy,
                "reviewed": 0, "applied": 0,
            }
            if transform_model is not None:
                properties["transform_model"] = transform_model
            sink.write({"geometry": mapping(box(0, 0, 10, 10)), "properties": properties})

    cells_by_pair = []
    for from_date, to_date, dx, dy in pair_values:
        cell = {
            "from_date": from_date, "to_date": to_date, "cell_id": "r00c00",
            "tile_col0": 0, "tile_row0_north_to_south": 0, "patch_tiles": 1,
            "common_tile_coverage_fraction": 1.0,
            "status": "LOCAL_CANDIDATE_NOT_REVIEWED", "matched_features_raw": 20,
            "eligible_matches": 12, "inliers": 7, "inlier_ratio": 0.58,
            "coverage_4x4": 0.2, "p90_residual_px": 1.5, "p90_residual_m": 0.05,
            "correction_dx_m_candidate": dx, "correction_dy_m_candidate": dy,
            "ground_mask_reviewed": False, "field_interpolated": False,
            "applied_to_image_or_geometry": False,
            "polygon_epsg32649": mapping(box(0, 0, 10, 10)),
        }
        if transform_model is not None:
            cell["transform_model"] = transform_model
        cells_by_pair.append(cell)
    pairs = []
    for from_date, to_date, _, _ in pair_values:
        cell = next(row for row in cells_by_pair
                    if row["from_date"] == from_date and row["to_date"] == to_date)
        pairs.append({"from_date": from_date, "to_date": to_date, "cell_count": 1,
                      "field_interpolated": False, "field_applied": False,
                      "cells": [cell]})

    diagnostic = root / "LOCAL_TILE_FIELD_DIAGNOSTIC.json"
    report = {
        "schema_version": "xian_r5.local_tile_lightglue_field_diagnostic.v1",
        "case_id": "SYNTHETIC", "scope": "FIXED_CASE_LOCAL_ONLY",
        "registration_promoted": False, "registration_applied": False,
        "field_interpolated": False,
        "mask": {"ground_mask_reviewed": False},
        "grid": grid,
        "base_raster": {"path": str(base), "crs": "EPSG:32649",
                         "resolution": [10.0, 10.0], "sha256": digest(base)},
        "input_mosaic_diagnostic": str(mosaic),
        "input_mosaic_diagnostic_sha256": digest(mosaic),
        "model": {"receipt_path": str(model), "receipt_sha256": digest(model),
                  "device": "cpu", "weights_sha256": {str(path): digest(path) for path in weights}},
        "outputs": {"candidate_vector": str(vector)},
        "actual_dates_in_order": dates,
        "pairwise_fields": pairs,
    }
    if transform_model is not None:
        report["transform_model"] = transform_model
    diagnostic.write_text(json.dumps(report), encoding="utf-8")
    return diagnostic, base, digest(base)


def attach_synthetic_stitched_mosaics(root: Path, diagnostic: Path) -> dict:
    """Replace placeholder tiles with real JPEGs and tiny georeferenced mosaics."""
    report = json.loads(diagnostic.read_text(encoding="utf-8"))
    mosaic_path = Path(report["input_mosaic_diagnostic"])
    mosaic = json.loads(mosaic_path.read_text(encoding="utf-8"))
    scenes = {"2018": "xa180429", "2019": "xa190101", "2020": "xa201231"}
    candidate_epochs = {}
    for actual_date, entries in mosaic["submeter_tiles"].items():
        source_epochs = {str(entry["selected_source_epoch"]) for entry in entries}
        if len(source_epochs) != 1:
            raise AssertionError("Fixture expects one selected source epoch per date")
        epoch = next(iter(source_epochs))
        scene = scenes[epoch]
        candidate_epochs[actual_date] = [epoch]
        task_root = root / scene
        tile_root = task_root / "20"
        tile_root.mkdir(parents=True, exist_ok=True)
        (root / f"{scene}.geid").write_text(
            f"[HistoryDate]\nHistoryDate={epoch}-01-01\n\n"
            "[Zoom]\nZoom1=20\nZoom2=20\n\n"
            "[XY]\nMinX_20=420796\nMinY_20=312172\n"
            "MaxX_20=420797\nMaxY_20=312173\n\n"
            "[AREA]\nLeft_Longitude_download_20=0\n"
            "Right_Longitude_download_20=2\nTop_Latitude_download_20=2\n"
            "Bottom_Latitude_download_20=0\n",
            encoding="utf-8",
        )
        stitched = root / f"{scene}_combined" / f"{scene}_zoom_20.tif"
        stitched.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            stitched,
            "w",
            driver="GTiff",
            width=512,
            height=512,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(0, 2, 2 / 512, 2 / 512),
        ) as sink:
            sink.write(np.zeros((3, 512, 512), dtype=np.uint8))

        for entry in entries:
            tile_x, tile_y = (int(value) for value in entry["tile_id"].split(":"))
            tile_dir = tile_root / str(tile_x)
            tile_dir.mkdir(parents=True, exist_ok=True)
            tile_path = tile_dir / f"gesh_{tile_x}_{tile_y}_20.jpg"
            Image.new("RGB", (256, 256), (tile_x % 255, tile_y % 255, 80)).save(
                tile_path, format="JPEG")
            tile_hash = hashlib.sha256(tile_path.read_bytes()).hexdigest()
            entry["path"] = str(tile_path)
            entry["sha256"] = tile_hash
            entry["sha256_after_read"] = tile_hash

    mosaic["scope"] = "FIXED_CASE_LOCAL_ONLY"
    mosaic["actual_date_association"] = {
        "candidate_source_epochs_by_actual_date": candidate_epochs,
    }
    mosaic_path.write_text(json.dumps(mosaic), encoding="utf-8")
    report["input_mosaic_diagnostic_sha256"] = hashlib.sha256(
        mosaic_path.read_bytes()).hexdigest()
    diagnostic.write_text(json.dumps(report), encoding="utf-8")
    return load_mosaic_grids_by_source_epoch(
        mosaic_path, source_root=root, scenes_by_epoch=scenes)


def downgrade_mosaic_receipt_to_legacy(diagnostic: Path) -> Path:
    """Emulate the older fixed-case receipt schema without tile-level epochs."""
    report = json.loads(diagnostic.read_text(encoding="utf-8"))
    mosaic_path = Path(report["input_mosaic_diagnostic"])
    mosaic = json.loads(mosaic_path.read_text(encoding="utf-8"))
    mosaic["epochs"] = {
        actual_date: {"selected_source_epoch": actual_date[:4]}
        for actual_date in mosaic["submeter_tiles"]
    }
    mosaic.pop("actual_date_association", None)
    for entries in mosaic["submeter_tiles"].values():
        for entry in entries:
            entry.pop("selected_source_epoch", None)
    mosaic_path.write_text(json.dumps(mosaic), encoding="utf-8")
    report["input_mosaic_diagnostic_sha256"] = hashlib.sha256(
        mosaic_path.read_bytes()).hexdigest()
    diagnostic.write_text(json.dumps(report), encoding="utf-8")
    return mosaic_path


def cell(cell_id: str = "c0", *, bounds=(0, 0, 10, 10),
         status="LOCAL_CANDIDATE_NOT_REVIEWED", dx=1.5, dy=-2.0,
         reviewed=False, independent=False, position_uncertainty=None,
         source_component_scope="SINGLE_SOURCE_COMPONENT_PER_CELL") -> DisplacementCell:
    return DisplacementCell(
        cell_id=cell_id,
        support_geometry=box(*bounds),
        correction_dx_m_candidate=dx,
        correction_dy_m_candidate=dy,
        fit_dispersion_m=0.4,
        status=status,
        method="SYNTHETIC_TEST_TIE",
        source_ref=f"test:{cell_id}",
        reviewed=reviewed,
        independent_ground_evidence=independent,
        positional_uncertainty_m=position_uncertainty,
        metadata={
            "source_component_scope": source_component_scope,
            "source_component_ids": ([{
                "actual_date": "2018-11-01", "tile_id": "420796:312172",
                "path": "source/420796_312172.jpg", "sha256": "a" * 64,
            }] if source_component_scope == "SINGLE_SOURCE_COMPONENT_PER_CELL" else []),
            "source_component_coverage_fraction": (
                1.0 if source_component_scope == "SINGLE_SOURCE_COMPONENT_PER_CELL" else 0.5),
        },
    )


class DisplacementFieldTests(unittest.TestCase):
    def test_multi_anchor_base_ties_require_identical_local_support_settings(self):
        baseline = {
            "radius_tiles": 10, "roof_exclusion_buffer_m": 5.0,
            "local_grid_size": 0,
            "local_support_mode": "EXACT_SOURCE_TILE_COMPONENTS",
            "local_search_radius_px": 1, "local_minimum_samples": 20,
        }

        signature = _base_tie_settings_signature(baseline)

        different_threshold = {**baseline, "local_minimum_samples": 30}
        different_support = {**baseline, "local_support_mode": "UNIFORM_GRID"}
        self.assertEqual(signature, _base_tie_settings_signature(dict(baseline)))
        self.assertNotEqual(signature, _base_tie_settings_signature(different_threshold))
        self.assertNotEqual(signature, _base_tie_settings_signature(different_support))

    def test_candidate_can_be_sampled_but_never_applied(self):
        field = LocalDisplacementField("f0", "2018-04-29", "2018-11-01", (cell(),))

        sample = field.sample(5, 5)

        self.assertEqual(sample.status, "LOCAL_CANDIDATE_NOT_REVIEWED")
        self.assertEqual(sample.correction_dx_m_candidate, 1.5)
        self.assertEqual(sample.correction_dy_m_candidate, -2.0)
        self.assertIsNone(sample.accepted_correction_m)
        self.assertFalse(sample.interpolated)

    def test_accepted_correction_requires_independent_review_and_accuracy(self):
        cases = (
            cell(status="ACCEPTED_REVIEWED_GROUND", reviewed=True,
                 independent=False, position_uncertainty=0.5),
            cell(status="ACCEPTED_REVIEWED_GROUND", reviewed=True,
                 independent=True, position_uncertainty=2.01),
            cell(status="LOCAL_CANDIDATE_NOT_REVIEWED", reviewed=True,
                 independent=True, position_uncertainty=0.5),
        )
        for candidate in cases:
            with self.subTest(candidate=candidate):
                field = LocalDisplacementField("f0", "2018-04-29", "2018-11-01", (candidate,))
                self.assertIsNone(field.sample(5, 5).accepted_correction_m)

        accepted = cell(status="ACCEPTED_REVIEWED_GROUND", reviewed=True,
                        independent=True, position_uncertainty=1.25)
        field = LocalDisplacementField("f0", "2018-04-29", "2018-11-01", (accepted,))
        self.assertEqual(field.sample(5, 5).accepted_correction_m, (1.5, -2.0))

    def test_reviewed_correction_requires_single_hash_identified_source_component(self):
        candidate = cell(status="ACCEPTED_REVIEWED_GROUND", reviewed=True,
                         independent=True, position_uncertainty=0.5,
                         source_component_scope="MULTIPLE_SOURCE_COMPONENTS_PER_CELL")
        field = LocalDisplacementField("f0", "2018-04-29", "2018-11-01", (candidate,))

        self.assertFalse(candidate.accepted)
        self.assertEqual(candidate.source_component_issue,
                         "MOSAIC_COMPONENT_BOUNDARY_UNRESOLVED")
        self.assertIsNone(field.sample(5, 5).accepted_correction_m)

    def test_source_component_scope_requires_complete_coverage(self):
        candidate = cell(status="ACCEPTED_REVIEWED_GROUND", reviewed=True,
                         independent=True, position_uncertainty=0.5)
        candidate.metadata["source_component_coverage_fraction"] = 0.99

        self.assertFalse(candidate.source_component_scoped)
        self.assertFalse(candidate.accepted)

    def test_gap_and_overlapping_cells_are_unresolved_without_interpolation(self):
        field = LocalDisplacementField("f0", "2018-04-29", "2018-11-01",
                                       (cell(bounds=(0, 0, 4, 10)),
                                        cell("c1", bounds=(6, 0, 10, 10))))

        self.assertEqual(field.sample(5, 5).status, "OUTSIDE_FIELD")
        self.assertIsNone(field.sample(5, 5).accepted_correction_m)

        overlap = LocalDisplacementField("f1", "2018-04-29", "2018-11-01",
                                          (cell(bounds=(0, 0, 7, 10)),
                                           cell("c1", bounds=(5, 0, 10, 10))))
        self.assertEqual(overlap.sample(6, 5).status, "AMBIGUOUS_OVERLAPPING_CELLS")

    def test_actual_dates_and_image_grid_scope_are_mandatory(self):
        with self.assertRaises(ValueError):
            LocalDisplacementField("f0", "2018", "2019-01-01", (cell(),))
        with self.assertRaises(ValueError):
            LocalDisplacementField("f0", "2019-01-01", "2018-01-01", (cell(),))
        unknown_reference = LocalDisplacementField(
            "base2019", None, "2019-12-02", (cell(),),
            reference_label="Xian_2019_TrueColor",
            reference_source="base:Xian_2019_TrueColor.tif",
            reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN",
        )
        self.assertEqual(unknown_reference.reference_date, None)
        self.assertEqual(unknown_reference.reference_date_status,
                         "REFERENCE_ACQUISITION_DATE_UNKNOWN")
        with self.assertRaises(ValueError):
            LocalDisplacementField("bad", None, "2019-12-02", (cell(),))
        with self.assertRaises(ValueError):
            LocalDisplacementField("f0", "2018-01-01", "2018-02-01", (cell(),),
                                   correction_scope="ROOF_TO_GROUND_PARALLAX")
        with self.assertRaises(ValueError):
            DisplacementCell("bad", box(0, 0, 1, 1), 1.0, 2.0, 0.5,
                             "LOCAL_CANDIDATE_NOT_REVIEWED", "test", "test", applied=True)

    def test_grouping_preserves_actual_date_pairs(self):
        fields = group_fields([
            ("2018-04-29", "2018-11-01", cell("a")),
            ("2018-04-29", "2018-11-01", cell("b", bounds=(10, 0, 20, 10))),
            ("2018-11-01", "2019-12-02", cell("c")),
        ])

        self.assertEqual([field.field_id for field in fields], [
            "2018-04-29_to_2018-11-01", "2018-11-01_to_2019-12-02"])
        self.assertEqual([len(field.cells) for field in fields], [2, 1])

    def test_composed_field_reader_checks_paired_manifest_and_keeps_candidate_unapplied(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gpkg = root / "composed_alignment_candidates.gpkg"
            manifest_path = root / "COMPOSED_ALIGNMENT_DIAGNOSTIC.json"
            schema = {"geometry": "Polygon", "properties": {
                "field_id": "str:64", "cell_id": "str:24", "moving_date": "str:16",
                "reference_label": "str:80", "reference_date": "str:16",
                "reference_date_status": "str:48", "status": "str:64",
                "candidate_dx_m": "float", "candidate_dy_m": "float",
                "base_dx_m": "float", "base_dy_m": "float",
                "highres_dx_m": "float", "highres_dy_m": "float",
                "cycle_rmse_m": "float", "cycle_max_m": "float", "cycle_dof": "int",
                "peak_corr": "float", "peak_margin": "float", "eligible_px": "int",
                "reviewed": "int", "ground_evidence": "int", "applied": "int"}}
            with fiona.open(gpkg, "w", driver="GPKG", layer="composed_alignment_candidates",
                            schema=schema, crs="EPSG:32649") as sink:
                sink.write({"geometry": mapping(box(0, 0, 10, 10)), "properties": {
                    "field_id": "2018-04-29_to_Xian_2019_TrueColor", "cell_id": "r0c0",
                    "moving_date": "2018-04-29", "reference_label": "Xian_2019_TrueColor",
                    "reference_date": None,
                    "reference_date_status": "REFERENCE_ACQUISITION_DATE_UNKNOWN",
                    "status": "COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY",
                    "candidate_dx_m": 1.25, "candidate_dy_m": -0.75,
                    "base_dx_m": 0.5, "base_dy_m": -0.25,
                    "highres_dx_m": 0.75, "highres_dy_m": -0.5,
                    "cycle_rmse_m": 0.8, "cycle_max_m": 1.6, "cycle_dof": 10,
                    "peak_corr": 0.9, "peak_margin": 0.2, "eligible_px": 100,
                    "reviewed": 0, "ground_evidence": 0, "applied": 0}})
            base_pair_source = root / "BASE_MOSAIC_PAIR_DIAGNOSTIC.json"
            base_tie_source = root / "BASE_TIE_DIAGNOSTIC.json"
            base_pair_source.write_text(json.dumps({"source": "pair"}), encoding="utf-8")
            base_tie_source.write_text(json.dumps({"source": "tie"}), encoding="utf-8")

            def sha256(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            manifest = {
                "schema_version": "xian_r5.composed_alignment_candidates.v1",
                "anchor_year_label": 2019,
                "anchor_path": "reference:Xian_2019_TrueColor.tif",
                "anchor_acquisition_date_status": "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN",
                "crs": "EPSG:32649", "field_applied": False, "field_interpolated": False,
                "date_fields": [{"actual_date": "2018-04-29", "cell_id": "r0c0",
                                 "status": "COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY",
                                 "composed_correction_m_candidate": [1.25, -0.75],
                                 "support_bounds_epsg32649": [0, 0, 10, 10]}],
                "source_diagnostics": {
                    "base_pair_path": str(base_pair_source),
                    "base_pair_sha256_before": sha256(base_pair_source),
                    "base_pair_sha256_after": sha256(base_pair_source),
                    "base_tie_path": str(base_tie_source),
                    "base_tie_sha256_before": sha256(base_tie_source),
                    "base_tie_sha256_after": sha256(base_tie_source),
                },
                "products": {"geopackage": gpkg.name,
                             "position_accuracy_claimed": False, "accepted_cells": 0}}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            fields = read_composed_alignment_fields(gpkg, manifest_path)

            self.assertEqual(len(fields), 1)
            self.assertEqual(fields[0].reference_date, None)
            self.assertEqual(fields[0].reference_date_status,
                             "REFERENCE_ACQUISITION_DATE_UNKNOWN")
            sample = fields[0].sample(5, 5)
            self.assertEqual((sample.correction_dx_m_candidate,
                              sample.correction_dy_m_candidate), (1.25, -0.75))
            self.assertIsNone(sample.accepted_correction_m)
            self.assertFalse(fields[0].cells[0].applied)

            base_pair_source.write_text(json.dumps({"source": "changed"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source diagnostic hash mismatch"):
                read_composed_alignment_fields(gpkg, manifest_path)

    def test_composed_field_reader_rejects_vector_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gpkg = root / "composed_alignment_candidates.gpkg"
            manifest_path = root / "COMPOSED_ALIGNMENT_DIAGNOSTIC.json"
            schema = {"geometry": "Polygon", "properties": {
                "field_id": "str:64", "cell_id": "str:24", "moving_date": "str:16",
                "reference_label": "str:80", "reference_date": "str:16",
                "reference_date_status": "str:48", "status": "str:64",
                "candidate_dx_m": "float", "candidate_dy_m": "float",
                "base_dx_m": "float", "base_dy_m": "float",
                "highres_dx_m": "float", "highres_dy_m": "float",
                "cycle_rmse_m": "float", "cycle_max_m": "float", "cycle_dof": "int",
                "peak_corr": "float", "peak_margin": "float", "eligible_px": "int",
                "reviewed": "int", "ground_evidence": "int", "applied": "int"}}
            with fiona.open(gpkg, "w", driver="GPKG", layer="composed_alignment_candidates",
                            schema=schema, crs="EPSG:32649") as sink:
                sink.write({"geometry": mapping(box(0, 0, 10, 10)), "properties": {
                    "field_id": "2018-04-29_to_Xian_2019_TrueColor", "cell_id": "r0c0",
                    "moving_date": "2018-04-29", "reference_label": "Xian_2019_TrueColor",
                    "reference_date": None,
                    "reference_date_status": "REFERENCE_ACQUISITION_DATE_UNKNOWN",
                    "status": "COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY",
                    "candidate_dx_m": 1.3, "candidate_dy_m": -0.75,
                    "base_dx_m": 0.5, "base_dy_m": -0.25,
                    "highres_dx_m": 0.75, "highres_dy_m": -0.5,
                    "cycle_rmse_m": 0.8, "cycle_max_m": 1.6, "cycle_dof": 10,
                    "peak_corr": 0.9, "peak_margin": 0.2, "eligible_px": 100,
                    "reviewed": 0, "ground_evidence": 0, "applied": 0}})
            manifest_path.write_text(json.dumps({
                "schema_version": "xian_r5.composed_alignment_candidates.v1",
                "anchor_year_label": 2019, "anchor_path": "reference.tif",
                "anchor_acquisition_date_status": "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN",
                "crs": "EPSG:32649", "field_applied": False, "field_interpolated": False,
                "date_fields": [{"actual_date": "2018-04-29", "cell_id": "r0c0",
                                 "status": "COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY",
                                 "composed_correction_m_candidate": [1.25, -0.75],
                                 "support_bounds_epsg32649": [0, 0, 10, 10]}],
                "products": {"geopackage": gpkg.name,
                             "position_accuracy_claimed": False, "accepted_cells": 0}}),
                encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "vector mismatch"):
                read_composed_alignment_fields(gpkg, manifest_path)

    def test_direct_base_tie_adapter_keeps_only_subpixel_nonboundary_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "BASE_TIE_DIAGNOSTIC.json"
            windows = ([0, 0, 10, 10], [10, 0, 20, 10],
                       [0, 10, 10, 20], [10, 10, 20, 20])
            cells = []
            for index, window in enumerate(windows):
                best = {"corr": 0.8, "dx_px": 1, "dy_px": 0, "n": 120,
                        "subpixel_peak": {"dx_px": 1.25, "dy_px": -0.5,
                                          "status": "SUBPIXEL_CORRELATION_PEAK_CANDIDATE_NOT_APPLIED"}}
                if index == 1:
                    best["subpixel_peak"] = None
                elif index == 2:
                    best = None
                elif index == 3:
                    best["dx_px"] = 3
                cells.append({"row": index // 2, "col": index % 2,
                              "base_pixel_window": window,
                              "valid_pixel_count": 200,
                              "maximum_masked_samples_available": 80 if index == 2 else 140,
                              "minimum_samples_required": 100,
                              "sample_support_status": (
                                  "INSUFFICIENT_MASKED_SAMPLES" if index == 2
                                  else "NUMERIC_PEAK_AVAILABLE"),
                              "best": best,
                              "peak_margin": 0.05 if best else None,
                              "status": "COARSE_TIE_CANDIDATE_NOT_APPLIED" if best else "NO_VALID_TIE"})
            base_path = Path(temp) / "Xian_2019_TrueColor.tif"
            report = {
                "schema_version": "xian_r5.base_tie_diagnostic.v1",
                "automatic_registration_promoted": False,
                "parameters": {"base_reference_year": 2019},
                "epochs": {"2018": {
                    "actual_date": "2018-04-29", "base_path": str(base_path),
                    "base_sha256": "synthetic-hash",
                    "submeter_tiles": {"tile_ids": [{
                        "tile_id": "420796:312172",
                        "actual_date": "2018-04-29",
                        "path": "synthetic/source_tile.jpg",
                        "sha256": "a" * 64,
                        "support_polygon_epsg32649": {
                            "type": "Polygon",
                            "coordinates": [[[100, 200], [200, 200], [200, 100],
                                             [100, 100], [100, 200]]],
                        },
                    }]},
                    "base_grid": {"crs": "EPSG:32649", "resolution": [10.0, 10.0],
                                  "window": [50, 60, 20, 20],
                                  "transform": [10, 0, 100, 0, -10, 200]},
                    "local_spatial_tie_diagnostic": {
                        "grid_rows": 2, "grid_cols": 2, "search_radius_10m_pixels": 3,
                        "field_interpolated": False, "field_applied": False,
                        "cells": cells}}}}
            report_path.write_text(json.dumps(report), encoding="utf-8")

            fields = read_direct_base_tie_fields(report_path)

            self.assertEqual(len(fields), 1)
            field = fields[0]
            self.assertEqual(field.reference_date, None)
            self.assertEqual(field.reference_label, "Xian_2019_TrueColor")
            self.assertEqual(field.cells[0].status, "DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED")
            self.assertAlmostEqual(field.cells[0].correction_dx_m_candidate, -12.5)
            self.assertAlmostEqual(field.cells[0].correction_dy_m_candidate, -5.0)
            self.assertTrue(field.cells[0].source_component_scoped)
            self.assertEqual(field.cells[0].metadata["source_component_coverage_fraction"], 1.0)
            self.assertEqual(field.cells[0].metadata["source_component_ids"][0]["tile_id"],
                             "420796:312172")
            self.assertEqual(field.cells[1].status, "DIRECT_BASE_TIE_COARSE_ONLY_UNRESOLVED")
            self.assertIsNone(field.cells[1].correction_dx_m_candidate)
            self.assertEqual(field.cells[2].status, "DIRECT_BASE_TIE_NO_VALID_TIE")
            self.assertEqual(field.cells[2].metadata["sample_support_status"],
                             "INSUFFICIENT_MASKED_SAMPLES")
            self.assertEqual(field.cells[2].metadata["maximum_masked_samples_available"], 80)
            self.assertEqual(field.cells[3].status, "DIRECT_BASE_TIE_SEARCH_BOUNDARY_UNRESOLVED")

    def test_direct_base_tie_adapter_preserves_exact_source_tile_component_support(self):
        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "BASE_TIE_DIAGNOSTIC.json"
            base_path = Path(temp) / "Xian_2019_TrueColor.tif"
            support = mapping(box(100, 100, 200, 200))
            tile_entry = {
                "tile_id": "420796:312172", "actual_date": "2018-04-29",
                "path": "source/420796_312172.jpg", "sha256": "a" * 64,
                "support_polygon_epsg32649": support,
            }
            cell = {
                "cell_id": "t420796_312172", "tile_id": tile_entry["tile_id"],
                "base_pixel_window": [0, 0, 12, 12],
                "support_polygon_epsg32649": support,
                "best": {
                    "corr": 0.81, "dx_px": 0, "dy_px": 0, "n": 24,
                    "subpixel_peak": {
                        "dx_px": 0.25, "dy_px": -0.5,
                        "status": "SUBPIXEL_CORRELATION_PEAK_CANDIDATE_NOT_APPLIED",
                    },
                },
                "runner_up": {"corr": 0.72, "dx_px": -1, "dy_px": 0, "n": 23},
                "peak_margin": 0.09,
                "maximum_masked_samples_available": 27,
                "minimum_samples_required": 20,
                "sample_support_status": "NUMERIC_PEAK_AVAILABLE",
                "status": "COMPONENT_TILE_TIE_CANDIDATE_NOT_APPLIED",
            }
            report_path.write_text(json.dumps({
                "schema_version": "xian_r5.base_tie_diagnostic.v1",
                "automatic_registration_promoted": False,
                "parameters": {"base_reference_year": 2019,
                               "local_search_radius_px": 1},
                "epochs": {"2018": {
                    "actual_date": "2018-04-29", "base_reference_year": 2019,
                    "base_path": str(base_path), "base_sha256": "c" * 64,
                    "base_grid": {"crs": "EPSG:32649", "resolution": [10, 10],
                                  "window": [50, 60, 20, 20],
                                  "transform": [10, 0, 100, 0, -10, 200]},
                    "submeter_tiles": {"tile_ids": [tile_entry]},
                    "local_spatial_tie_diagnostic": {
                        "support_mode": "EXACT_SOURCE_TILE_COMPONENTS",
                        "search_radius_px": 1,
                        "search_radius_10m_pixels": 1,
                        "field_interpolated": False, "field_applied": False,
                        "cells": [cell],
                    },
                }},
            }), encoding="utf-8")

            fields = read_direct_base_tie_fields(report_path)

        self.assertEqual(len(fields), 1)
        adapted = fields[0].cells[0]
        self.assertEqual(adapted.cell_id, "t420796_312172")
        self.assertEqual(adapted.metadata["cell_support_mode"],
                         "EXACT_SOURCE_TILE_COMPONENTS")
        self.assertEqual(adapted.metadata["source_component_tile_id"],
                         "420796:312172")
        self.assertTrue(adapted.source_component_scoped)
        self.assertEqual(adapted.correction_dx_m_candidate, -2.5)
        self.assertEqual(adapted.correction_dy_m_candidate, -5.0)
        self.assertFalse(adapted.applied)

    def test_direct_base_tie_accepts_date_specific_annual_rasters_on_verified_shared_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_path = root / "BASE_TIE_DIAGNOSTIC.json"
            references = {
                2018: root / "Xian_2018_TrueColor.tif",
                2019: root / "Xian_2019_TrueColor.tif",
            }
            digests = {2018: "a" * 64, 2019: "b" * 64}

            def make_epoch(actual_date, year):
                return {
                    "actual_date": actual_date,
                    "base_reference_year": year,
                    "base_path": str(references[year]),
                    "base_sha256": digests[year],
                    "base_grid": {
                        "crs": "EPSG:32649", "resolution": [10.0, 10.0],
                        "window": [50, 60, 20, 20],
                        "transform": [10, 0, 100, 0, -10, 200],
                    },
                    "local_spatial_tie_diagnostic": {
                        "grid_rows": 1, "grid_cols": 1,
                        "search_radius_10m_pixels": 3,
                        "field_interpolated": False, "field_applied": False,
                        "cells": [{
                            "row": 0, "col": 0,
                            "base_pixel_window": [0, 0, 20, 20],
                            "best": {
                                "corr": 0.75, "dx_px": 0, "dy_px": 0, "n": 120,
                                "subpixel_peak": {
                                    "dx_px": 0.25, "dy_px": -0.5,
                                    "status": "SUBPIXEL_CORRELATION_PEAK_CANDIDATE_NOT_APPLIED",
                                },
                            },
                            "peak_margin": 0.08,
                            "maximum_masked_samples_available": 120,
                            "minimum_samples_required": 100,
                            "sample_support_status": "NUMERIC_PEAK_AVAILABLE",
                        }],
                    },
                }

            report_path.write_text(json.dumps({
                "schema_version": "xian_r5.base_tie_diagnostic.v1",
                "automatic_registration_promoted": False,
                "parameters": {"base_reference_year": None},
                "epochs": {
                    "2018": make_epoch("2018-04-29", 2018),
                    "2020": make_epoch("2019-12-02", 2019),
                },
            }), encoding="utf-8")
            verified = {str(path.resolve()): digests[year]
                        for year, path in references.items()}

            with self.assertRaisesRegex(ValueError, "verified shared-grid raster stack"):
                read_direct_base_tie_fields(report_path)

            fields = read_direct_base_tie_fields(
                report_path, verified_reference_raster_hashes=verified)

            self.assertEqual([field.reference_label for field in fields], [
                "Xian_2018_TrueColor", "Xian_2019_TrueColor",
            ])
            self.assertEqual([field.moving_date for field in fields], [
                "2018-04-29", "2019-12-02",
            ])
            self.assertTrue(all(field.cells[0].correction_dx_m_candidate is not None
                                for field in fields))
            mismatched = dict(verified)
            mismatched[str(references[2018].resolve())] = "c" * 64
            with self.assertRaisesRegex(ValueError, "differs from the verified 10 m stack"):
                read_direct_base_tie_fields(
                    report_path, verified_reference_raster_hashes=mismatched)

    def test_alignment_path_comparison_reports_difference_not_accuracy(self):
        direct = LocalDisplacementField(
            "direct", None, "2018-04-29", (cell(dx=1.0, dy=2.0),),
            reference_label="Xian_2019_TrueColor", reference_source="base:2019",
            reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")
        alternate = LocalDisplacementField(
            "alternate", None, "2018-04-29", (cell(dx=4.0, dy=6.0),),
            reference_label="Xian_2019_TrueColor", reference_source="base:2019",
            reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")

        result = compare_alignment_fields(direct, alternate)

        self.assertEqual(result["both_numeric_candidate_count"], 1)
        self.assertEqual(result["max_pathway_difference_m"], 5.0)
        self.assertFalse(result["correction_applied"])
        self.assertEqual(result["interpretation"],
                         "PATHWAY_AGREEMENT_SCREEN_ONLY_NOT_POSITIONAL_ACCURACY")

    def test_lightglue_local_field_reader_preserves_unreviewed_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            diagnostic, base, base_hash = write_lightglue_fixture(Path(temp))

            fields = read_lightglue_local_field_candidates(
                diagnostic, expected_base_raster=base, expected_base_sha256=base_hash)

            self.assertEqual(len(fields), 3)
            self.assertEqual(fields[0].field_id, "ALIKED_2018-04-29_to_2019-01-01")
            sample = fields[0].sample(5, 5)
            self.assertEqual(sample.status, "LOCAL_CANDIDATE_NOT_REVIEWED")
            self.assertEqual((sample.correction_dx_m_candidate,
                              sample.correction_dy_m_candidate), (1.25, -0.5))
            self.assertIsNone(sample.accepted_correction_m)
            self.assertFalse(fields[0].cells[0].reviewed)
            self.assertFalse(fields[0].cells[0].applied)
            self.assertFalse(fields[0].cells[0].independent_ground_evidence)
            self.assertIsNone(fields[0].cells[0].positional_uncertainty_m)
            self.assertEqual(fields[0].cells[0].method, LOCAL_FIELD_METHOD)
            self.assertEqual(fields[0].cells[0].metadata["transform_model"],
                             LOCAL_TRANSFORM_MODEL)
            self.assertEqual(fields[0].cells[0].metadata["model_device"], "cpu")
            self.assertEqual(fields[0].cells[0].metadata["p90_residual_m"], 0.05)
            tile_grid = fields[0].cells[0].metadata["source_tile_grid"]
            self.assertEqual(tile_grid["tile_window"],
                             {"x0": 420796, "y0": 312172, "nx": 2, "ny": 2})
            self.assertEqual(tile_grid["pixel_window"], [0, 0, 256, 256])
            self.assertEqual(
                tile_grid["tiles_by_actual_date"]["2018-04-29"]["present_tiles"][0]["tile_id"],
                "420796:312173")
            self.assertEqual(
                tile_grid["tiles_by_actual_date"]["2019-01-01"]["present_tiles"][0]["path"],
                str(Path(temp) / "2019-01-01_0_1.jpg"))
            self.assertEqual(fields[0].cells[0].metadata["source_component_scope"],
                             "SINGLE_SOURCE_COMPONENT_PER_CELL")
            self.assertTrue(fields[0].cells[0].source_component_scoped)

    def test_lightglue_local_field_reader_rejects_unknown_transform_model(self):
        with tempfile.TemporaryDirectory() as temp:
            diagnostic, base, base_hash = write_lightglue_fixture(Path(temp))
            report = json.loads(diagnostic.read_text(encoding="utf-8"))
            report["transform_model"] = "HOMOGRAPHY"
            diagnostic.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "(?i)unsupported.*transform model"):
                read_lightglue_local_field_candidates(
                    diagnostic, expected_base_raster=base,
                    expected_base_sha256=base_hash)

    def test_lightglue_local_field_reader_validates_explicit_transform_model(self):
        with tempfile.TemporaryDirectory() as temp:
            diagnostic, base, base_hash = write_lightglue_fixture(
                Path(temp), transform_model=LOCAL_TRANSFORM_MODEL)

            fields = read_lightglue_local_field_candidates(
                diagnostic, expected_base_raster=base,
                expected_base_sha256=base_hash)

            self.assertEqual(fields[0].cells[0].method, LOCAL_FIELD_METHOD)
            self.assertEqual(fields[0].cells[0].metadata["transform_model"],
                             LOCAL_TRANSFORM_MODEL)

    def test_source_component_support_is_single_only_when_one_tile_covers_cell(self):
        from shapely.geometry import mapping

        entries = [
            {"tile_id": "420796:312172", "actual_date": "2018-04-29",
             "path": "tile-a.jpg", "sha256": "a" * 64,
             "support_polygon_epsg32649": mapping(box(0, 0, 50, 100))},
            {"tile_id": "420797:312172", "actual_date": "2018-04-29",
             "path": "tile-b.jpg", "sha256": "b" * 64,
             "support_polygon_epsg32649": mapping(box(50, 0, 100, 100))},
        ]

        single = _source_component_scope_for_support(
            box(5, 5, 45, 95), entries, "2018-04-29")
        prepared = _prepare_source_component_index(entries, "2018-04-29")
        indexed_single = _source_component_scope_for_support(
            box(5, 5, 45, 95), entries, "2018-04-29",
            prepared_index=prepared)
        crossing = _source_component_scope_for_support(
            box(45, 5, 55, 95), entries, "2018-04-29")
        indexed_crossing = _source_component_scope_for_support(
            box(45, 5, 55, 95), entries, "2018-04-29",
            prepared_index=prepared)
        incomplete = _source_component_scope_for_support(
            box(45, 5, 55, 95), entries[:1], "2018-04-29")

        self.assertEqual(single["source_component_scope"],
                         "SINGLE_SOURCE_COMPONENT_PER_CELL")
        self.assertEqual(single["source_component_coverage_fraction"], 1.0)
        self.assertEqual(indexed_single, single)
        self.assertEqual(crossing["source_component_scope"],
                         "MULTIPLE_SOURCE_COMPONENTS_PER_CELL")
        self.assertEqual(indexed_crossing, crossing)
        self.assertEqual(incomplete["source_component_scope"],
                         "INCOMPLETE_SOURCE_COMPONENT_COVERAGE")

    def test_paired_source_component_scope_tracks_both_actual_date_mosaics(self):
        from shapely.geometry import mapping

        def entry(tile_id, actual_date, geometry, suffix):
            return {
                "tile_id": tile_id, "actual_date": actual_date,
                "path": f"{suffix}.jpg", "sha256": suffix[0] * 64,
                "support_polygon_epsg32649": mapping(geometry),
            }

        support = box(0, 0, 10, 10)
        left_date, right_date = "2018-04-29", "2018-11-01"
        matching = {
            left_date: [entry("420796:312172", left_date, support, "a")],
            right_date: [entry("420796:312172", right_date, support, "b")],
        }
        single = paired_source_component_scope_for_support(
            support, matching, (left_date, right_date))
        multiple = paired_source_component_scope_for_support(
            support,
            {**matching, left_date: [
                entry("420796:312172", left_date, box(0, 0, 5, 10), "a"),
                entry("420797:312172", left_date, box(5, 0, 10, 10), "c"),
            ]},
            (left_date, right_date),
        )
        incomplete = paired_source_component_scope_for_support(
            support,
            {**matching, left_date: [
                entry("420796:312172", left_date, box(0, 0, 5, 10), "a")
            ]},
            (left_date, right_date),
        )
        mismatch = paired_source_component_scope_for_support(
            support,
            {**matching, right_date: [
                entry("420797:312172", right_date, support, "c")
            ]},
            (left_date, right_date),
        )
        unknown = paired_source_component_scope_for_support(
            support, {left_date: matching[left_date]}, (left_date, right_date))

        self.assertEqual(single["source_component_scope"],
                         "SINGLE_SOURCE_COMPONENT_PER_CELL")
        self.assertEqual(single["source_component_coverage_fraction"], 1.0)
        self.assertEqual(len(single["source_component_ids"]), 2)
        self.assertEqual(multiple["source_component_scope"],
                         "MULTIPLE_SOURCE_COMPONENTS_PER_CELL")
        self.assertEqual(incomplete["source_component_scope"],
                         "INCOMPLETE_SOURCE_COMPONENT_COVERAGE")
        self.assertEqual(mismatch["source_component_scope"],
                         "SOURCE_COMPONENT_ID_MISMATCH")
        self.assertEqual(unknown["source_component_scope"],
                         "SOURCE_COMPONENT_SCOPE_UNKNOWN")

    def test_lightglue_scope_requires_one_same_tile_for_every_date(self):
        record = {"tile_id": "420796:312172", "path": "tile.jpg",
                  "sha256": "c" * 64}
        provenance = {"tiles_by_actual_date": {
            "2018-04-29": {"expected_tile_ids": [record["tile_id"]],
                            "present_tiles": [record], "missing_tile_ids": []},
            "2018-11-01": {"expected_tile_ids": [record["tile_id"]],
                            "present_tiles": [record], "missing_tile_ids": []},
        }}

        single = _lightglue_source_component_scope(
            provenance, "2018-04-29", "2018-11-01")
        provenance["tiles_by_actual_date"]["2018-11-01"]["expected_tile_ids"].append(
            "420797:312172")
        provenance["tiles_by_actual_date"]["2018-11-01"]["present_tiles"].append(
            {**record, "tile_id": "420797:312172"})
        multiple = _lightglue_source_component_scope(
            provenance, "2018-04-29", "2018-11-01")

        self.assertEqual(single["source_component_scope"],
                         "SINGLE_SOURCE_COMPONENT_PER_CELL")
        self.assertEqual(single["source_component_coverage_fraction"], 1.0)
        self.assertEqual(multiple["source_component_scope"],
                         "MULTIPLE_SOURCE_COMPONENTS_PER_CELL")

    def test_lightglue_reader_attaches_verified_stitched_mosaic_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            diagnostic, base, base_hash = write_lightglue_fixture(root)
            source_grids = attach_synthetic_stitched_mosaics(root, diagnostic)

            fields = read_lightglue_local_field_candidates(
                diagnostic,
                expected_base_raster=base,
                expected_base_sha256=base_hash,
                mosaic_grids_by_source_epoch=source_grids,
            )

            self.assertEqual(len(fields), 3)
            tile_grid = fields[0].cells[0].metadata["source_tile_grid"]
            tile = tile_grid["tiles_by_actual_date"]["2018-04-29"]["present_tiles"][0]
            stitched = tile["stitched_mosaic_grid"]
            self.assertTrue(stitched["source_component_window_verified"])
            self.assertEqual(stitched["nominal_task_date"], "2018-01-01")
            self.assertEqual(stitched["pixel_window_col_row_width_height"], [0, 0, 256, 256])
            self.assertEqual(stitched["window_bounds_epsg4326"], [0.0, 1.0, 1.0, 2.0])
            self.assertEqual(tile["selected_source_epoch"], "2018")

    def test_legacy_mosaic_receipt_resolves_epoch_from_exact_source_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            diagnostic, base, base_hash = write_lightglue_fixture(root)
            attach_synthetic_stitched_mosaics(root, diagnostic)
            mosaic_path = downgrade_mosaic_receipt_to_legacy(diagnostic)

            source_grids = load_mosaic_grids_by_source_epoch(
                mosaic_path,
                source_root=root,
                scenes_by_epoch={
                    "2018": "xa180429", "2019": "xa190101", "2020": "xa201231",
                },
            )
            fields = read_lightglue_local_field_candidates(
                diagnostic,
                expected_base_raster=base,
                expected_base_sha256=base_hash,
                mosaic_grids_by_source_epoch=source_grids,
            )

            self.assertEqual(set(source_grids), {"2018", "2019", "2020"})
            self.assertEqual(source_grids["2018"].summary()["actual_acquisition_date_status"],
                             "MUST_BE_RESOLVED_PER_TILE")
            tile = fields[0].cells[0].metadata["source_tile_grid"][
                "tiles_by_actual_date"]["2018-04-29"]["present_tiles"][0]
            self.assertEqual(tile["selected_source_epoch"], "2018")
            self.assertTrue(tile["stitched_mosaic_grid"]["source_component_window_verified"])

    def test_legacy_mosaic_receipt_rejects_tile_path_from_another_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            diagnostic, _, _ = write_lightglue_fixture(root)
            attach_synthetic_stitched_mosaics(root, diagnostic)
            mosaic_path = downgrade_mosaic_receipt_to_legacy(diagnostic)
            mosaic = json.loads(mosaic_path.read_text(encoding="utf-8"))
            mosaic["submeter_tiles"]["2018-04-29"][0]["path"] = str(
                root / "xa190101" / "20" / "420796" / "gesh_420796_312172_20.jpg")
            mosaic_path.write_text(json.dumps(mosaic), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source epoch/path is not in the actual-date receipt"):
                load_mosaic_grids_by_source_epoch(
                    mosaic_path,
                    source_root=root,
                    scenes_by_epoch={
                        "2018": "xa180429", "2019": "xa190101", "2020": "xa201231",
                    },
                )

    def test_lightglue_edge_clipped_cell_uses_recorded_tile_dimensions(self):
        grid = {
            "tile_window": {"x0": 10, "y0": 20, "nx": 4, "ny": 5},
            "tile_pixel_size": [256, 256],
            "pixel_transform": [10 / 256, 0, 100, 0, -10 / 256, 200],
        }
        cell = {
            "cell_id": "r01c01",
            "tile_col0": 3,
            "tile_row0_north_to_south": 3,
            "tile_rows_in_cell": 2,
            "tile_cols_in_cell": 1,
            "patch_tiles": 3,
        }
        records = {
            "2018-04-29": {
                (13, 21): {"tile_id": "13:21"},
                (13, 20): {"tile_id": "13:20"},
            }
        }

        provenance, geometry = _lightglue_cell_tile_provenance(
            cell, cell_id="r01c01", grid=grid, records_by_date=records)

        self.assertEqual(provenance["pixel_window"], [768, 768, 1024, 1280])
        self.assertEqual(provenance["tile_rows_in_cell"], 2)
        self.assertEqual(provenance["tile_cols_in_cell"], 1)
        self.assertEqual(geometry.bounds, (130.0, 150.0, 140.0, 170.0))
        self.assertEqual(
            [tile["tile_id"] for tile in provenance["tiles_by_actual_date"][
                "2018-04-29"]["present_tiles"]],
            ["13:21", "13:20"],
        )

    def test_lightglue_local_field_reader_rejects_geometry_outside_affine_tile_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            diagnostic, base, base_hash = write_lightglue_fixture(Path(temp))
            report = json.loads(diagnostic.read_text(encoding="utf-8"))
            report["pairwise_fields"][0]["cells"][0]["polygon_epsg32649"] = mapping(
                box(1, 0, 11, 10))
            diagnostic.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError,
                                        "support geometry differs from affine tile grid"):
                read_lightglue_local_field_candidates(
                    diagnostic, expected_base_raster=base, expected_base_sha256=base_hash)

    def test_lightglue_local_field_reader_rejects_report_vector_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            diagnostic, base, base_hash = write_lightglue_fixture(Path(temp))
            report = json.loads(diagnostic.read_text(encoding="utf-8"))
            report["pairwise_fields"][0]["cells"][0]["correction_dx_m_candidate"] = 99.0
            diagnostic.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "GeoPackage corr_dx_m differs from JSON"):
                read_lightglue_local_field_candidates(
                    diagnostic, expected_base_raster=base, expected_base_sha256=base_hash)

    def test_lightglue_local_field_reader_rejects_metric_residual_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            diagnostic, base, base_hash = write_lightglue_fixture(Path(temp))
            report = json.loads(diagnostic.read_text(encoding="utf-8"))
            report["pairwise_fields"][0]["cells"][0]["p90_residual_m"] = 0.25
            diagnostic.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "GeoPackage p90_m differs from JSON"):
                read_lightglue_local_field_candidates(
                    diagnostic, expected_base_raster=base,
                    expected_base_sha256=base_hash)


if __name__ == "__main__":
    unittest.main()
