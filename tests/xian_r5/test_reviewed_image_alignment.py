from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.affinity import translate
from shapely.geometry import Point, box, mapping, shape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xian_r5.alignment.field import DisplacementCell, LocalDisplacementField
from xian_r5.alignment.reviewed_field import attach_reviewed_alignment_cells
from xian_r5.alignment.roof_candidate import apply_accepted_roof_grid_alignment
from xian_r5.reconstruction.core import Observation, _tbia_observation, reconstruct
from xian_r5.reconstruction.products import write_products


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_relative_tie_sidecar(
    sidecar_path: Path,
    alignment_field,
    evidence_path: Path,
    *,
    tie_overrides: dict[int, dict] | None = None,
    sidecar_overrides: dict | None = None,
) -> dict:
    tie_overrides = tie_overrides or {}
    sidecar_overrides = sidecar_overrides or {}
    cell = alignment_field.cells[0]
    evidence_path = evidence_path.resolve()
    evidence_digest = sha256(evidence_path)
    ties = []
    for index in range(6):
        role = "FIT" if index < 3 else "HOLDOUT"
        ge_x = 101.0 + index
        ge_y = 105.0
        residual_x = 0.6 if role == "FIT" else 0.9
        candidate_dx = float(cell.correction_dx_m_candidate)
        candidate_dy = float(cell.correction_dy_m_candidate)
        reference_x = ge_x + candidate_dx - residual_x
        reference_y = ge_y + candidate_dy
        tie = {
            "tie_id": f"synthetic-tie-{index + 1}",
            "field_id": alignment_field.field_id,
            "cell_id": cell.cell_id,
            "moving_date": alignment_field.moving_date,
            "FIT_or_HOLDOUT": role,
            "feature_type": "PAVED_CURB_JUNCTION",
            "ge_x": ge_x,
            "ge_y": ge_y,
            "reference_x": reference_x,
            "reference_y": reference_y,
            "candidate_dx_m": candidate_dx,
            "candidate_dy_m": candidate_dy,
            "residual_dx_m": residual_x,
            "residual_dy_m": 0.0,
            "residual_m": residual_x,
            "ge_localization_bound_m": 0.1,
            "reference_localization_bound_m": 0.1,
            "adjusted_bound_m": residual_x + 0.2,
            "localization_basis": (
                "Distinct curb junction is bracketed by visible edges in both images; "
                "each point is localized to within 0.1 m by the visible edge geometry."),
            "evidence_path": str(evidence_path),
            "evidence_sha256": evidence_digest,
            "within_exact_support": True,
            "independence_group": f"synthetic-ground-feature-{index + 1}",
            "precision_eligible": True,
            "precision_ineligibility_reason": None,
            "feature_class": "STABLE_NONBUILDING_GROUND_FEATURE",
            "stable_nonbuilding_ground_evidence": True,
            "stability_basis": (
                "Non-building paved curb junction is unchanged across the compared "
                "observations and supported by the attached evidence artifact."),
        }
        tie.update(tie_overrides.get(index, {}))
        tie["within_exact_support"] = all(cell.support_geometry.covers(point) for point in (
            Point(tie["ge_x"], tie["ge_y"]),
            Point(tie["reference_x"], tie["reference_y"]),
            Point(tie["ge_x"] + candidate_dx, tie["ge_y"] + candidate_dy),
        )) if "within_exact_support" not in tie_overrides.get(index, {}) else tie["within_exact_support"]
        ties.append(tie)

    eligible = [tie for tie in ties if tie["precision_eligible"]]
    adjusted = {"FIT": [], "HOLDOUT": []}
    for tie in eligible:
        actual_dx = (tie["ge_x"] + tie["candidate_dx_m"]) - tie["reference_x"]
        actual_dy = (tie["ge_y"] + tie["candidate_dy_m"]) - tie["reference_y"]
        adjusted[tie["FIT_or_HOLDOUT"]].append(
            (actual_dx ** 2 + actual_dy ** 2) ** 0.5
            + tie["ge_localization_bound_m"]
            + tie["reference_localization_bound_m"])
    fit_max = max(adjusted["FIT"]) if adjusted["FIT"] else 0.0
    holdout_max = max(adjusted["HOLDOUT"]) if adjusted["HOLDOUT"] else 0.0
    sidecar = {
        "schema_version": "xian_r5.relative_alignment_tie_sidecar.v2",
        "review_contract_amendment": "R5_RELATIVE_ALIGNMENT_REVIEW_CONTRACT_AMENDMENT_V2",
        "uncertainty_rule": "R5_RELATIVE_ALIGNMENT_UNCERTAINTY_RULE_V1",
        "coordinates_crs": "EPSG:32649",
        "field_id": alignment_field.field_id,
        "cell_id": cell.cell_id,
        "moving_date": alignment_field.moving_date,
        "candidate_source_ref": cell.source_ref,
        "candidate_dx_m": float(cell.correction_dx_m_candidate),
        "candidate_dy_m": float(cell.correction_dy_m_candidate),
        "reference_source": str(Path(alignment_field.reference_source).resolve()),
        "reference_source_sha256": sha256(Path(alignment_field.reference_source)),
        "support_geometry_sha256": hashlib.sha256(cell.support_geometry.wkb).hexdigest(),
        "fit_max_adjusted_bound_m": fit_max,
        "holdout_max_adjusted_bound_m": holdout_max,
        "relative_registration_uncertainty_m": holdout_max,
        "positional_uncertainty_m": holdout_max,
        "uncertainty_semantics": "RELATIVE_IMAGE_GRID_ONLY",
        "absolute_position_accuracy": "UNKNOWN",
        "ties": ties,
    }
    sidecar.update(sidecar_overrides)
    sidecar_path.write_text(json.dumps(sidecar, indent=2, allow_nan=True) + "\n",
                            encoding="utf-8")
    return sidecar


def candidate_field(field_id: str, actual_date: str, roof, reference: Path,
                    dx: float, dy: float) -> LocalDisplacementField:
    cell = DisplacementCell(
        "cell-01", roof.buffer(10), dx, dy, 0.25,
        "LOCAL_CANDIDATE_NOT_REVIEWED", "SYNTHETIC_IMAGE_TIE",
        f"candidate:{field_id}:cell-01",
        metadata={
            "source_component_scope": "SINGLE_SOURCE_COMPONENT_PER_CELL",
            "source_component_ids": [{
                "actual_date": actual_date, "tile_id": "420796:312172",
                "path": f"source/{actual_date}/420796_312172.jpg",
                "sha256": "a" * 64,
            }],
            "source_component_coverage_fraction": 1.0,
        })
    return LocalDisplacementField(
        field_id, None, actual_date, (cell,),
        reference_label="fixed-grid", reference_source=str(reference.resolve()),
        reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")


def spatial_composition_candidate_field(actual_date: str, roof,
                                        reference: Path) -> LocalDisplacementField:
    cell = DisplacementCell(
        "composed-cell", roof.buffer(10), 2.0, -3.0, None,
        "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY",
        "SPATIAL_OVERLAP_TRANSLATION_POSE_GRAPH_CANDIDATE",
        "spatial-composition:test/composed-cell",
        metadata={
            "candidate_status": "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY",
            "composition_method": "SPATIAL_OVERLAP_TRANSLATION_POSE_GRAPH",
            "source_component_scope": "SOURCE_COMPONENT_SCOPE_UNKNOWN",
            "source_candidate_cell_refs": ["edge:one", "edge:two"],
            "source_edge_count": 2,
            "anchor_acquisition_date_status": (
                "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN"),
            "reference_component_id_status": (
                "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"),
            "reference_component_id_statuses": [
                "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"],
            "cycle_limit_semantics": "INTERNAL_CLOSURE_SCREEN_ONLY_NOT_POSITIONAL_ACCURACY",
            "closure_max_m": 0.8,
            "roof_to_ground_parallax_resolved": False,
        })
    return LocalDisplacementField(
        f"SPATIAL_{actual_date}_TO_fixed-grid", None, actual_date, (cell,),
        reference_label="fixed-grid", reference_source=str(reference.resolve()),
        reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")


class ReviewedImageAlignmentTests(unittest.TestCase):
    def test_zero_accepted_roofs_propagate_unresolved_not_background(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "products"
            observations = [
                Observation("a", "2018", "2018-04-29", None, None, box(10, 10, 20, 20), "raw:a"),
                Observation("b", "2019", "2018-11-01", None, None, box(12, 10, 22, 20), "raw:b"),
            ]
            manifest = write_products(output, reconstruct(observations))
            self.assertEqual(manifest["roof_registered_count"], 0)
            self.assertEqual(manifest["presence_state_counts"]["U"], 2)
            for prefix in ("registered", "canonical"):
                with rasterio.open(output / f"2018-04-29_to_2018-11-01_{prefix}_change_NON_GROUND.tif") as src:
                    values = src.read(1)
                    self.assertTrue((values == 4).any())
                    self.assertFalse(np.isin(values, [1, 2, 3]).any())
            with rasterio.open(output / "2018-04-29_registration_status.tif") as src:
                self.assertTrue((src.read(1) == 3).any())

    def test_pending_review_and_evidence_hash_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.tif"
            reference.write_bytes(b"reference")
            evidence = root / "evidence.json"
            evidence.write_bytes(b"review")
            field = candidate_field("field", "2018-04-29", box(0, 0, 5, 5), reference, 2, 3)
            for name, override, pattern in [
                ("pending", {"reviewed": 0}, "completed human review"),
                ("hash", {"evidence_sha256": "0" * 64}, "SHA-256 mismatch"),
                ("date", {"moving_date": "2020-01-01"}, "date mismatch"),
            ]:
                path = root / f"{name}.gpkg"
                self._write_review_layer(path, [field], evidence, reference, overrides={"field": override})
                with self.assertRaisesRegex(ValueError, pattern):
                    attach_reviewed_alignment_cells([field], path)

    def test_empty_accepted_layer_preserves_candidates_without_accuracy_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.tif"
            field = candidate_field("test", "2018-04-29", box(0, 0, 5, 5), reference, 2, 3)
            path = root / "empty.gpkg"
            self._write_review_layer(path, [], root / "unused.json", reference)
            fields, receipt = attach_reviewed_alignment_cells([field], path)
            self.assertEqual(fields, [field])
            self.assertEqual(receipt["accepted_cell_count"], 0)
            self.assertEqual(receipt["review_status"], "NO_REVIEWABLE_ALIGNMENT_EVIDENCE")
            self.assertFalse(receipt["absolute_position_accuracy_claimed"])
            self.assertFalse(fields[0].cells[0].accepted)

    def _write_review_layer(self, path, fields, evidence_path, reference, *, overrides=None):
        overrides = overrides or {}
        schema = {"geometry": "Polygon", "properties": {
            "field_id": "str:120", "cell_id": "str:120", "moving_date": "str:16",
            "reference_source": "str:512", "reference_source_sha256": "str:64",
            "candidate_source_ref": "str:254", "correction_dx_m": "float",
            "correction_dy_m": "float", "independent_ground_evidence": "int",
            "ground_evidence_type": "str:48", "review_basis": "str:48",
            "alignment_evidence_type": "str:64", "reviewed": "int",
            "positional_uncertainty_m": "float", "reviewer_id": "str:96",
            "positional_uncertainty_semantics": "str:40",
            "uncertainty_semantics": "str:40",
            "relative_registration_uncertainty_m": "float",
            "fit_max_adjusted_bound_m": "float", "holdout_max_adjusted_bound_m": "float",
            "nonbuilding_tie_pairs": "int", "holdout_tie_pairs": "int",
            "relative_ties_within_support_cell": "int",
            "reviewed_at": "str:40", "evidence_path": "str:512",
            "evidence_sha256": "str:64"}}
        with fiona.open(path, "w", driver="GPKG", layer="reviewed_image_alignment_cells",
                        schema=schema, crs="EPSG:32649") as sink:
            for alignment_field in fields:
                cell = alignment_field.cells[0]
                record = {
                    "field_id": alignment_field.field_id, "cell_id": cell.cell_id,
                    "moving_date": alignment_field.moving_date,
                    "reference_source": str(reference.resolve()),
                    "reference_source_sha256": sha256(reference),
                    "candidate_source_ref": cell.source_ref,
                    "correction_dx_m": float(cell.correction_dx_m_candidate),
                    "correction_dy_m": float(cell.correction_dy_m_candidate),
                    "independent_ground_evidence": 1,
                    "ground_evidence_type": "SURVEY_CONTROL", "reviewed": 1,
                    "review_basis": "INDEPENDENT_GROUND_CONTROL",
                    "alignment_evidence_type": "INDEPENDENT_GROUND_CONTROL",
                    "positional_uncertainty_m": 0.4,
                    "positional_uncertainty_semantics": "ABSOLUTE_GROUND_CONTROL",
                    "uncertainty_semantics": "",
                    "relative_registration_uncertainty_m": None,
                    "fit_max_adjusted_bound_m": None,
                    "holdout_max_adjusted_bound_m": None,
                    "nonbuilding_tie_pairs": 0, "holdout_tie_pairs": 0,
                    "relative_ties_within_support_cell": 1,
                    "reviewer_id": "reviewer-01",
                    "reviewed_at": "2026-09-25T10:00:00+08:00",
                    "evidence_path": str(evidence_path.resolve()),
                    "evidence_sha256": sha256(evidence_path),
                }
                row_override = dict(overrides.get(alignment_field.field_id, {}))
                tie_overrides = row_override.pop("_tie_overrides", None)
                sidecar_overrides = row_override.pop("_sidecar_overrides", None)
                if row_override.get("review_basis") == "RELATIVE_REFERENCE_IMAGE_FEATURES":
                    sidecar_path = path.with_name(f"{path.stem}_{cell.cell_id}_ties.json")
                    tie_sidecar = write_relative_tie_sidecar(
                        sidecar_path, alignment_field, Path(evidence_path),
                        tie_overrides=tie_overrides,
                        sidecar_overrides=sidecar_overrides)
                    eligible_ties = [
                        tie for tie in tie_sidecar["ties"]
                        if tie["precision_eligible"]
                    ]
                    record["uncertainty_semantics"] = "RELATIVE_IMAGE_GRID_ONLY"
                    record.update({
                        "positional_uncertainty_m": tie_sidecar["positional_uncertainty_m"],
                        "positional_uncertainty_semantics": "RELATIVE_IMAGE_GRID_ONLY",
                        "relative_registration_uncertainty_m": (
                            tie_sidecar["relative_registration_uncertainty_m"]),
                        "fit_max_adjusted_bound_m": tie_sidecar["fit_max_adjusted_bound_m"],
                        "holdout_max_adjusted_bound_m": tie_sidecar[
                            "holdout_max_adjusted_bound_m"],
                        "nonbuilding_tie_pairs": sum(
                            tie["FIT_or_HOLDOUT"] == "FIT" for tie in eligible_ties),
                        "holdout_tie_pairs": sum(
                            tie["FIT_or_HOLDOUT"] == "HOLDOUT" for tie in eligible_ties),
                        "relative_ties_within_support_cell": int(all(
                            tie["within_exact_support"] for tie in tie_sidecar["ties"])),
                        "evidence_path": str(sidecar_path.resolve()),
                        "evidence_sha256": sha256(sidecar_path),
                    })
                record.update(row_override)
                sink.write({"geometry": mapping(cell.support_geometry), "properties": record})

    def test_reviewed_cells_apply_to_tracking_geometry_but_not_ground_products(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "fixed_grid.tif"
            with rasterio.open(reference, "w", driver="GTiff", width=40, height=40,
                               count=1, dtype="uint8", crs="EPSG:32649",
                               transform=from_origin(0, 400, 10, 10)) as sink:
                sink.write(np.zeros((1, 40, 40), dtype="uint8"))
            evidence = root / "survey_control.txt"
            evidence.write_text("independent control record\n", encoding="utf-8")
            first_roof = box(100, 100, 110, 110)
            second_roof = box(98, 103, 108, 113)
            fields = [
                candidate_field("first", "2018-04-29", first_roof, reference, 2, -3),
                candidate_field("second", "2018-11-01", second_roof, reference, 4, -6),
            ]
            review_layer = root / "review.gpkg"
            self._write_review_layer(review_layer, fields, evidence, reference)

            reviewed_fields, report = attach_reviewed_alignment_cells(fields, review_layer)
            self.assertEqual(report["accepted_cell_count"], 2)
            self.assertFalse(report["field_application_performed"])
            self.assertTrue(all(field.cells[0].accepted for field in reviewed_fields))
            observations = [
                Observation("first", "2018", "2018-04-29", None, None,
                            first_roof, "roof:first", candidate_identity="track-a"),
                Observation("second", "2019", "2018-11-01", None, None,
                            second_roof, "roof:second", candidate_identity="track-a"),
            ]
            for observation in observations:
                application = apply_accepted_roof_grid_alignment(
                    observation.roof, observation.actual_date, reviewed_fields,
                    reference_sources=(str(reference.resolve()),))
                self.assertTrue(application.applied)
                observation.registered_roof = application.geometry
                observation.image_alignment_status = application.status
                observation.image_alignment_field_id = application.field_id
                observation.image_alignment_cell_id = application.cell_id
                observation.image_alignment_dx_m = application.correction_dx_m
                observation.image_alignment_dy_m = application.correction_dy_m
                observation.image_alignment_positional_uncertainty_m = application.positional_uncertainty_m
                observation.image_alignment_review_metadata = application.review_metadata
            self.assertEqual(observations[0].registered_roof.bounds,
                             observations[1].registered_roof.bounds)
            self.assertEqual(observations[0].roof.bounds, first_roof.bounds)
            self.assertEqual(observations[1].roof.bounds, second_roof.bounds)
            tracking_feature = _tbia_observation(observations[0])
            self.assertAlmostEqual(tracking_feature.x, 107.0)
            self.assertAlmostEqual(tracking_feature.y, 102.0)
            self.assertEqual(tracking_feature.alignment_dx, 0.0)
            self.assertEqual(tracking_feature.alignment_dy, 0.0)

            result = reconstruct(observations)
            output = root / "products"
            manifest = write_products(output, result, anchor_raster=reference,
                                      alignment_fields=reviewed_fields)
            self.assertTrue(manifest["image_alignment_field_applied"])
            self.assertEqual(manifest["registered_roof_candidate_count"], 2)
            self.assertEqual(manifest["formal_ground_observations"], 0)
            with fiona.open(output / "building_lineage.gpkg", layer="roof_candidates") as layer:
                raw = {row["properties"]["uid"]: shape(row["geometry"]) for row in layer}
            with fiona.open(output / "building_lineage.gpkg", layer="registered_roof_candidates") as layer:
                registered = {row["properties"]["uid"]: shape(row["geometry"]) for row in layer}
            with fiona.open(output / "building_lineage.gpkg", layer="roof_raw") as layer:
                explicit_raw = {row["properties"]["uid"]: shape(row["geometry"]) for row in layer}
            with fiona.open(output / "building_lineage.gpkg", layer="roof_registered") as layer:
                explicit_registered = {
                    row["properties"]["uid"]: shape(row["geometry"]) for row in layer}
            self.assertEqual(raw["first"].bounds, first_roof.bounds)
            self.assertEqual(explicit_raw["first"].bounds, first_roof.bounds)
            self.assertEqual(registered["first"].bounds,
                             translate(first_roof, xoff=2, yoff=-3).bounds)
            self.assertEqual(explicit_registered["first"].bounds, registered["first"].bounds)
            self.assertTrue((output / "2018-04-29_roof_raw.tif").is_file())
            self.assertTrue((output / "2018-04-29_roof_registered.tif").is_file())
            self.assertTrue((output / "2018-04-29_presence.tif").is_file())
            with rasterio.open(output / "2018-04-29_ground_footprint.tif") as raster:
                values = set(np.unique(raster.read(1)).tolist())
                self.assertNotIn(1, values)
                self.assertIn(2, values)
            with fiona.open(output / "building_lineage.gpkg", layer="identity_graph_nodes") as layer:
                node_geometries = {row["properties"]["uid"]: shape(row["geometry"]) for row in layer}
            self.assertEqual(node_geometries["first"].bounds, registered["first"].bounds)

    def test_canonical_medoid_uses_reviewed_identity_and_registered_roofs_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "fixed_grid.tif"
            with rasterio.open(reference, "w", driver="GTiff", width=40, height=40,
                               count=1, dtype="uint8", crs="EPSG:32649",
                               transform=from_origin(0, 400, 10, 10)) as sink:
                sink.write(np.zeros((1, 40, 40), dtype="uint8"))
            alignment_evidence = root / "alignment_review.txt"
            alignment_evidence.write_text("reviewed image-grid tie\n", encoding="utf-8")
            identity_evidence = root / "identity_review.txt"
            identity_evidence.write_text("same physical building across dates\n", encoding="utf-8")
            roofs = [box(100, 100, 110, 110), box(98, 103, 108, 113)]
            fields = [
                candidate_field("first", "2018-04-29", roofs[0], reference, 2, -3),
                candidate_field("second", "2018-11-01", roofs[1], reference, 4, -6),
            ]
            review_layer = root / "review.gpkg"
            self._write_review_layer(review_layer, fields, alignment_evidence, reference)
            reviewed_fields, _ = attach_reviewed_alignment_cells(fields, review_layer)
            observations = [
                Observation("first", "2018", "2018-04-29", None, None,
                            roofs[0], "roof:first"),
                Observation("second", "2019", "2018-11-01", None, None,
                            roofs[1], "roof:second"),
            ]
            for observation in observations:
                application = apply_accepted_roof_grid_alignment(
                    observation.roof, observation.actual_date, reviewed_fields,
                    reference_sources=(str(reference.resolve()),))
                observation.registered_roof = application.geometry
                observation.image_alignment_status = application.status
                observation.image_alignment_field_id = application.field_id
                observation.image_alignment_cell_id = application.cell_id
                observation.image_alignment_dx_m = application.correction_dx_m
                observation.image_alignment_dy_m = application.correction_dy_m
                observation.image_alignment_positional_uncertainty_m = (
                    application.positional_uncertainty_m)
                observation.image_alignment_review_basis = application.review_basis
                observation.image_alignment_review_metadata = application.review_metadata
                observation.identity_reviewed = True
                observation.confirmed_identity_key = "physical-building-A"
                observation.identity_evidence_type = "CROSS_DATE_MANUAL_REVIEW"
                observation.identity_evidence_ref = "review:physical-building-A"
                observation.identity_evidence_path = str(identity_evidence.resolve())
                observation.identity_evidence_sha256 = sha256(identity_evidence)
                observation.identity_reviewer_id = "reviewer-01"
                observation.identity_reviewed_at = "2026-09-25T10:00:00+08:00"
                observation.identity_evidence_integrity_verified = True
            result = reconstruct(observations)
            output = root / "canonical_products"
            manifest = write_products(output, result, anchor_raster=reference,
                                      alignment_fields=reviewed_fields)
            self.assertEqual(manifest["confirmed_ground_geometry_observations"], 0)
            self.assertEqual(manifest["confirmed_physical_identity_observations"], 2)
            self.assertEqual(manifest["canonical_geometry"]["resolved_count"], 1)
            self.assertTrue(manifest["grid_alignment_receipt_status"][
                "fine_to_10m_nesting_pass"])
            with fiona.open(output / "building_lineage.gpkg",
                            layer="canonical_building_geometry") as layer:
                canonical = list(layer)
            self.assertEqual(len(canonical), 1)
            self.assertIsNotNone(canonical[0]["geometry"])
            self.assertEqual(canonical[0]["properties"]["ground_truth"], 0)
            with rasterio.open(output / "canonical_building_geometry.tif") as raster:
                self.assertGreater(int((raster.read(1) == 1).sum()), 0)
            for prefix in ("registered", "canonical"):
                with rasterio.open(
                        output / f"2018-04-29_to_2018-11-01_{prefix}_change_NON_GROUND.tif") as raster:
                    values = raster.read(1)
                    self.assertGreater(int((values == 1).sum()), 0)
                    self.assertEqual(int((values == 2).sum()), 0)
                    self.assertEqual(int((values == 3).sum()), 0)
            with rasterio.open(output / "2018-04-29_to_2018-11-01_change.tif") as raster:
                self.assertFalse(np.isin(raster.read(1), [1, 2, 3]).any())

    def test_unreviewed_or_wrong_reference_candidates_are_never_applied(self):
        roof = box(10, 10, 20, 20)
        candidate = DisplacementCell(
            "cell", box(0, 0, 30, 30), 3, -2, 0.2,
            "LOCAL_CANDIDATE_NOT_REVIEWED", "TEST", "candidate:cell")
        field = LocalDisplacementField(
            "candidate", None, "2018-04-29", (candidate,),
            reference_label="ref", reference_source="reference:test",
            reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")
        result = apply_accepted_roof_grid_alignment(
            roof, "2018-04-29", [field], reference_sources=("reference:test",))
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "SOURCE_COMPONENT_SCOPE_UNKNOWN")
        self.assertEqual(result.geometry.bounds, roof.bounds)
        other = apply_accepted_roof_grid_alignment(
            roof, "2018-04-29", [field], reference_sources=("different-reference",))
        self.assertFalse(other.applied)
        self.assertEqual(other.status, "NO_MATCHING_REVIEWED_ALIGNMENT_FIELD")

    def test_relative_feature_review_applies_grid_shift_without_ground_accuracy_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "user_base_grid.tif"
            with rasterio.open(reference, "w", driver="GTiff", width=40, height=40,
                               count=1, dtype="uint8", crs="EPSG:32649",
                               transform=from_origin(0, 400, 10, 10)) as sink:
                sink.write(np.zeros((1, 40, 40), dtype="uint8"))
            evidence = root / "reviewed_nonbuilding_ties.txt"
            evidence.write_text(
                "3 FIT and 3 HOLDOUT curb-junction ties; recomputed maxima 0.8 m and 1.1 m.\n",
                encoding="utf-8")
            roof = box(100, 100, 110, 110)
            field = candidate_field("relative", "2018-04-29", roof, reference, 2, -3)
            review_layer = root / "relative_review.gpkg"
            self._write_review_layer(
                review_layer, [field], evidence, reference,
                overrides={"relative": {
                    "independent_ground_evidence": 0,
                    "ground_evidence_type": "",
                    "review_basis": "RELATIVE_REFERENCE_IMAGE_FEATURES",
                    "alignment_evidence_type": "STABLE_NONBUILDING_IMAGE_FEATURE_TIES",
                    "positional_uncertainty_m": 1.1,
                    "positional_uncertainty_semantics": "RELATIVE_IMAGE_GRID_ONLY",
                    "relative_registration_uncertainty_m": 1.1,
                    "nonbuilding_tie_pairs": 3,
                    "holdout_tie_pairs": 3,
                }})

            reviewed_fields, report = attach_reviewed_alignment_cells([field], review_layer)
            reviewed = reviewed_fields[0].cells[0]
            self.assertTrue(reviewed.accepted)
            self.assertFalse(reviewed.independent_ground_evidence)
            self.assertIsNone(reviewed.positional_uncertainty_m)
            self.assertAlmostEqual(reviewed.relative_registration_uncertainty_m, 1.1)
            self.assertAlmostEqual(
                reviewed.metadata["relative_grid_positional_uncertainty_m"], 1.1)
            self.assertEqual(
                reviewed.metadata["relative_grid_positional_uncertainty_semantics"],
                "RELATIVE_IMAGE_GRID_ONLY")
            self.assertEqual(reviewed.metadata["absolute_position_accuracy"], "UNKNOWN")
            self.assertEqual(reviewed.metadata["uncertainty_semantics"],
                             "RELATIVE_IMAGE_GRID_ONLY")
            self.assertAlmostEqual(report["reviewed_cells"][0]["fit_max_adjusted_bound_m"], 0.8)
            self.assertAlmostEqual(report["reviewed_cells"][0]["holdout_max_adjusted_bound_m"], 1.1)
            self.assertEqual(report["absolute_ground_control_cell_count"], 0)
            self.assertEqual(report["relative_reference_feature_cell_count"], 1)
            self.assertFalse(report["absolute_position_accuracy_claimed"])

            application = apply_accepted_roof_grid_alignment(
                roof, "2018-04-29", reviewed_fields,
                reference_sources=(str(reference.resolve()),))
            self.assertTrue(application.applied)
            self.assertEqual(application.geometry.bounds, (102, 97, 112, 107))
            self.assertIsNone(application.positional_uncertainty_m)
            self.assertAlmostEqual(application.relative_registration_uncertainty_m, 1.1)
            self.assertEqual(application.review_basis, "RELATIVE_REFERENCE_IMAGE_FEATURES")

            observation = Observation(
                "relative-roof", "2018", "2018-04-29", None, None, roof,
                "roof:relative", registered_roof=application.geometry,
                image_alignment_status=application.status,
                image_alignment_field_id=application.field_id,
                image_alignment_cell_id=application.cell_id,
                image_alignment_dx_m=application.correction_dx_m,
                image_alignment_dy_m=application.correction_dy_m,
                image_alignment_positional_uncertainty_m=application.positional_uncertainty_m,
                image_alignment_relative_uncertainty_m=application.relative_registration_uncertainty_m,
                image_alignment_review_basis=application.review_basis,
                image_alignment_review_metadata=application.review_metadata)
            feature = _tbia_observation(observation)
            self.assertAlmostEqual(feature.alignment_uncertainty, 1.1)
            self.assertEqual(feature.metadata["alignment_uncertainty_source"],
                             "RELATIVE_IMAGE_TO_BASE_GRID")

            result = reconstruct([observation])
            output = root / "products"
            manifest = write_products(output, result, anchor_raster=reference,
                                      alignment_fields=reviewed_fields)
            self.assertTrue(manifest["image_alignment_field_applied"])
            self.assertEqual(manifest["formal_ground_observations"], 0)
            self.assertIn(
                "ABSOLUTE_GROUND_POSITION_ACCURACY_REMAINS_UNKNOWN",
                manifest["image_alignment_evidence_semantics"][
                    "relative_reference_feature_review"])
            with fiona.open(output / "building_lineage.gpkg",
                            layer="registered_roof_candidates") as layer:
                properties = dict(next(iter(layer))["properties"])
            self.assertEqual(properties["alignment_review_basis"],
                             "RELATIVE_REFERENCE_IMAGE_FEATURES")
            self.assertAlmostEqual(properties["relative_unc_m"], 1.1)
            self.assertIsNone(properties["position_unc_m"])
            self.assertEqual(properties["independent_ground"], 0)
            with rasterio.open(output / "2018-04-29_ground_footprint.tif") as raster:
                values = set(np.unique(raster.read(1)).tolist())
                self.assertNotIn(1, values)
                self.assertIn(2, values)

    def test_relative_sidecar_deterministic_recomputation_and_fail_closed_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "user_base_grid.tif"
            with rasterio.open(reference, "w", driver="GTiff", width=40, height=40,
                               count=1, dtype="uint8", crs="EPSG:32649",
                               transform=from_origin(0, 400, 10, 10)) as sink:
                sink.write(np.zeros((1, 40, 40), dtype="uint8"))
            evidence = root / "stable_ground_feature_evidence.txt"
            evidence.write_text(
                "Synthetic test artifact: curb-corner evidence and localization bounds.\n",
                encoding="utf-8")
            field = candidate_field(
                "relative", "2018-04-29", box(100, 100, 110, 110),
                reference, 2.0, -3.0)
            base_review = {
                "independent_ground_evidence": 0,
                "ground_evidence_type": "",
                "review_basis": "RELATIVE_REFERENCE_IMAGE_FEATURES",
                "alignment_evidence_type": "STABLE_NONBUILDING_IMAGE_FEATURE_TIES",
                "positional_uncertainty_m": 1.1,
                "positional_uncertainty_semantics": "RELATIVE_IMAGE_GRID_ONLY",
                "relative_registration_uncertainty_m": 1.1,
            }

            valid_layer = root / "valid_relative.gpkg"
            self._write_review_layer(
                valid_layer, [field], evidence, reference,
                overrides={"relative": base_review})
            _, valid_report = attach_reviewed_alignment_cells([field], valid_layer)
            self.assertAlmostEqual(
                valid_report["reviewed_cells"][0]["fit_max_adjusted_bound_m"], 0.8)
            self.assertAlmostEqual(
                valid_report["reviewed_cells"][0]["holdout_max_adjusted_bound_m"], 1.1)
            self.assertEqual(
                valid_report["reviewed_cells"][0]["relative_registration_uncertainty_m"],
                valid_report["reviewed_cells"][0]["relative_grid_positional_uncertainty_m"])

            ineligible_fit = {
                "precision_eligible": False,
                "precision_ineligibility_reason": "TIE_NOT_PRECISION_ELIGIBLE",
                "ge_localization_bound_m": None,
                "reference_localization_bound_m": None,
                "adjusted_bound_m": None,
            }
            outside_support = {"ge_x": 500.0}
            duplicate_group = {"independence_group": "synthetic-ground-feature-1"}
            bad_localization = {"localization_basis": "high confidence"}
            bad_artifact_hash = {"evidence_sha256": "0" * 64}
            bad_adjusted = {"adjusted_bound_m": 0.5}
            bad_residual = {"residual_m": 1.5}
            cases = {
                "row_final_uncertainty_mismatch": {
                    "row": {**base_review, "relative_registration_uncertainty_m": 0.8}},
                "positional_uncertainty_m_mismatch": {
                    "row": {**base_review, "positional_uncertainty_m": 0.8}},
                "tie_adjusted_bound_m_mismatch": {
                    "row": base_review, "ties": {0: bad_adjusted}},
                "tie_residual_m_mismatch": {
                    "row": base_review, "ties": {0: bad_residual}},
                "duplicate_fit_holdout_independence_group": {
                    "row": base_review, "ties": {5: duplicate_group}},
                "confidence_label_is_not_localization_basis": {
                    "row": base_review, "ties": {0: bad_localization}},
                "unverified_localization_artifact_hash": {
                    "row": base_review, "ties": {0: bad_artifact_hash}},
                "tie_outside_exact_support": {
                    "row": {**base_review, "relative_ties_within_support_cell": 1},
                    "ties": {0: outside_support}},
                "ineligible_fit_tie_not_counted": {
                    "row": base_review, "ties": {0: ineligible_fit}},
                "sidecar_final_aggregate_mismatch": {
                    "row": base_review,
                    "sidecar": {"fit_max_adjusted_bound_m": 1.4}},
            }
            for name, case in cases.items():
                layer_path = root / f"{name}.gpkg"
                override = dict(case["row"])
                if "ties" in case:
                    override["_tie_overrides"] = case["ties"]
                if "sidecar" in case:
                    override["_sidecar_overrides"] = case["sidecar"]
                self._write_review_layer(
                    layer_path, [field], evidence, reference,
                    overrides={"relative": override})
                with self.subTest(case=name), self.assertRaises(ValueError):
                    attach_reviewed_alignment_cells([field], layer_path)

            over_limit_holdouts = {}
            for index in range(3, 6):
                ge_x = 101.0 + index
                ge_y = 105.0
                candidate_dx = field.cells[0].correction_dx_m_candidate
                candidate_dy = field.cells[0].correction_dy_m_candidate
                over_limit_holdouts[index] = {
                    "reference_x": ge_x + candidate_dx - 1.95,
                    "reference_y": ge_y + candidate_dy,
                    "residual_dx_m": 1.95,
                    "residual_dy_m": 0.0,
                    "residual_m": 1.95,
                    "adjusted_bound_m": 2.15,
                }
            over_limit_layer = root / "over_limit_holdout_bound.gpkg"
            self._write_review_layer(
                over_limit_layer, [field], evidence, reference,
                overrides={"relative": {
                    **base_review,
                    "_tie_overrides": over_limit_holdouts,
                }})
            with self.assertRaisesRegex(ValueError, "exceeds 2"):
                attach_reviewed_alignment_cells([field], over_limit_layer)

    def test_relative_feature_review_rejects_ground_claims_and_weak_ties(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "user_base_grid.tif"
            with rasterio.open(reference, "w", driver="GTiff", width=40, height=40,
                               count=1, dtype="uint8", crs="EPSG:32649",
                               transform=from_origin(0, 400, 10, 10)) as sink:
                sink.write(np.zeros((1, 40, 40), dtype="uint8"))
            evidence = root / "reviewed_nonbuilding_ties.txt"
            evidence.write_text("non-building correspondences and held-out residuals\n",
                                encoding="utf-8")
            roof = box(100, 100, 110, 110)
            field = candidate_field("relative", "2018-04-29", roof, reference, 2, -3)
            invalid_reviews = {
                "claims_independent_ground": {"independent_ground_evidence": 1},
                "claims_absolute_uncertainty": {"positional_uncertainty_m": 0.5},
                "relative_uncertainty_too_large": {"relative_registration_uncertainty_m": 2.1},
                "too_few_nonbuilding_ties": {"nonbuilding_tie_pairs": 2},
                "too_few_holdout_ties": {"holdout_tie_pairs": 2},
                "ties_not_within_cell": {"relative_ties_within_support_cell": 0},
            }
            for name, invalid_override in invalid_reviews.items():
                layer_path = root / f"{name}.gpkg"
                self._write_review_layer(
                    layer_path, [field], evidence, reference,
                    overrides={"relative": {
                        "independent_ground_evidence": 0,
                        "ground_evidence_type": "",
                        "review_basis": "RELATIVE_REFERENCE_IMAGE_FEATURES",
                        "alignment_evidence_type": "STABLE_NONBUILDING_IMAGE_FEATURE_TIES",
                        "positional_uncertainty_m": 1.1,
                        "positional_uncertainty_semantics": "RELATIVE_IMAGE_GRID_ONLY",
                        "relative_registration_uncertainty_m": 1.1,
                        "nonbuilding_tie_pairs": 3,
                        "holdout_tie_pairs": 3,
                        **invalid_override,
                    }})
                with self.subTest(case=name), self.assertRaises(ValueError):
                    attach_reviewed_alignment_cells([field], layer_path)

    def test_spatial_composition_cell_can_only_pass_exact_cell_relative_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "user_base_grid.tif"
            with rasterio.open(reference, "w", driver="GTiff", width=40, height=40,
                               count=1, dtype="uint8", crs="EPSG:32649",
                               transform=from_origin(0, 400, 10, 10)) as sink:
                sink.write(np.zeros((1, 40, 40), dtype="uint8"))
            evidence = root / "composed_cell_ties.txt"
            evidence.write_text(
                "4 non-building fit pairs and 3 held-out pairs, all inside exact composed cell; residuals 0.7m\n",
                encoding="utf-8")
            roof = box(100, 100, 110, 110)
            field = spatial_composition_candidate_field("2018-04-29", roof, reference)
            self.assertFalse(field.cells[0].source_component_scoped)
            self.assertTrue(field.cells[0].relative_cell_review_candidate)
            review_layer = root / "spatial_relative_review.gpkg"
            self._write_review_layer(
                review_layer, [field], evidence, reference,
                overrides={field.field_id: {
                    "independent_ground_evidence": 0,
                    "ground_evidence_type": "",
                    "review_basis": "RELATIVE_REFERENCE_IMAGE_FEATURES",
                    "alignment_evidence_type": "STABLE_NONBUILDING_IMAGE_FEATURE_TIES",
                    "positional_uncertainty_m": 1.1,
                    "positional_uncertainty_semantics": "RELATIVE_IMAGE_GRID_ONLY",
                    "relative_registration_uncertainty_m": 1.1,
                    "nonbuilding_tie_pairs": 3,
                    "holdout_tie_pairs": 3,
                    "relative_ties_within_support_cell": 1,
                }})

            reviewed_fields, report = attach_reviewed_alignment_cells([field], review_layer)
            reviewed = reviewed_fields[0].cells[0]
            self.assertTrue(reviewed.accepted)
            self.assertFalse(reviewed.source_component_scoped)
            self.assertTrue(reviewed.relative_cell_source_scope_review_authorized)
            self.assertEqual(reviewed.metadata["source_component_scope"],
                             "SOURCE_COMPONENT_SCOPE_UNKNOWN")
            self.assertIsNone(reviewed.positional_uncertainty_m)
            self.assertEqual(report["relative_cell_source_scope_authorization_count"], 1)
            self.assertFalse(report["absolute_position_accuracy_claimed"])

            application = apply_accepted_roof_grid_alignment(
                roof, "2018-04-29", reviewed_fields,
                reference_sources=(str(reference.resolve()),))
            self.assertTrue(application.applied)
            self.assertEqual(application.geometry.bounds, (102, 97, 112, 107))
            self.assertIsNone(application.positional_uncertainty_m)
            self.assertEqual(application.review_basis,
                             "RELATIVE_REFERENCE_IMAGE_FEATURES")

            observation = Observation(
                "spatial-relative-roof", "2018", "2018-04-29", None, None,
                roof, "roof:spatial-relative",
                registered_roof=application.geometry,
                image_alignment_status=application.status,
                image_alignment_field_id=application.field_id,
                image_alignment_cell_id=application.cell_id,
                image_alignment_dx_m=application.correction_dx_m,
                image_alignment_dy_m=application.correction_dy_m,
                image_alignment_positional_uncertainty_m=application.positional_uncertainty_m,
                image_alignment_relative_uncertainty_m=(
                    application.relative_registration_uncertainty_m),
                image_alignment_review_basis=application.review_basis,
                image_alignment_review_metadata=application.review_metadata)
            result = reconstruct([observation])
            output = root / "spatial_products"
            product = write_products(output, result, anchor_raster=reference,
                                     alignment_fields=reviewed_fields)
            self.assertTrue(product["image_alignment_field_applied"])
            self.assertEqual(product["formal_ground_observations"], 0)
            self.assertIn("SOURCE_COMPONENT_AND_PER_PIXEL_REFERENCE_PROVENANCE_STAY_UNKNOWN",
                          product["source_component_scope_semantics"])
            from tools.qa_xian_r5_grounded_reconstruction import (
                _valid_unknown_scope_spatial_relative_review,
            )
            with fiona.open(output / "building_lineage.gpkg",
                            layer="image_alignment_candidates") as layer:
                alignment_properties = dict(next(iter(layer))["properties"])
            self.assertTrue(_valid_unknown_scope_spatial_relative_review(
                alignment_properties, product["schema_version"]))
            with fiona.open(output / "building_lineage.gpkg",
                            layer="registered_roof_candidates") as layer:
                properties = dict(next(iter(layer))["properties"])
            self.assertEqual(properties["source_component_scope"],
                             "SOURCE_COMPONENT_SCOPE_UNKNOWN")
            self.assertEqual(properties["alignment_review_basis"],
                             "RELATIVE_REFERENCE_IMAGE_FEATURES")
            self.assertEqual(properties["independent_ground"], 0)
            self.assertIsNone(properties["position_unc_m"])
            self.assertEqual(properties["relative_cell_scope_review_authorized"], 1)
            self.assertEqual(properties["relative_ties_within_support_cell"], 1)
            with rasterio.open(output / "2018-04-29_ground_footprint.tif") as raster:
                self.assertNotIn(1, np.unique(raster.read(1)))

    def test_spatial_composition_unknown_scope_rejects_absolute_or_out_of_cell_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "user_base_grid.tif"
            with rasterio.open(reference, "w", driver="GTiff", width=40, height=40,
                               count=1, dtype="uint8", crs="EPSG:32649",
                               transform=from_origin(0, 400, 10, 10)) as sink:
                sink.write(np.zeros((1, 40, 40), dtype="uint8"))
            evidence = root / "review_evidence.txt"
            evidence.write_text("reviewed evidence\n", encoding="utf-8")
            roof = box(100, 100, 110, 110)
            field = spatial_composition_candidate_field("2018-04-29", roof, reference)
            invalid_reviews = {
                "absolute_control": {
                    "independent_ground_evidence": 1,
                    "ground_evidence_type": "SURVEY_CONTROL",
                    "review_basis": "INDEPENDENT_GROUND_CONTROL",
                    "alignment_evidence_type": "INDEPENDENT_GROUND_CONTROL",
                    "positional_uncertainty_m": 0.5,
                },
                "ties_not_confined_to_cell": {
                    "independent_ground_evidence": 0,
                    "ground_evidence_type": "",
                    "review_basis": "RELATIVE_REFERENCE_IMAGE_FEATURES",
                    "alignment_evidence_type": "STABLE_NONBUILDING_IMAGE_FEATURE_TIES",
                    "positional_uncertainty_m": 1.1,
                    "positional_uncertainty_semantics": "RELATIVE_IMAGE_GRID_ONLY",
                    "relative_registration_uncertainty_m": 1.1,
                    "nonbuilding_tie_pairs": 3,
                    "holdout_tie_pairs": 3,
                    "relative_ties_within_support_cell": 0,
                },
            }
            for name, override in invalid_reviews.items():
                review_path = root / f"{name}.gpkg"
                self._write_review_layer(
                    review_path, [field], evidence, reference,
                    overrides={field.field_id: override})
                with self.subTest(case=name), self.assertRaises(ValueError):
                    attach_reviewed_alignment_cells([field], review_path)

    def test_date_specific_co_gridded_base_field_writes_separate_candidate_products(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchor_2019 = root / "Xian_2019_TrueColor.tif"
            source_2018 = root / "Xian_2018_TrueColor.tif"
            profile = {
                "driver": "GTiff", "width": 40, "height": 40, "count": 1,
                "dtype": "uint8", "crs": "EPSG:32649",
                "transform": from_origin(0, 400, 10, 10),
            }
            for path in (anchor_2019, source_2018):
                with rasterio.open(path, "w", **profile) as sink:
                    sink.write(np.zeros((1, 40, 40), dtype="uint8"))

            roof = box(100, 100, 120, 120)
            field = candidate_field(
                "actual-year-base", "2018-04-29", roof, source_2018, 2.5, -1.25)
            observation = Observation(
                "one-roof", "2018", "2018-04-29", None, None, roof, "roof:one")
            result = reconstruct([observation])
            output = root / "products"

            manifest = write_products(
                output, result, anchor_raster=anchor_2019,
                alignment_fields=(field,),
                alignment_reference_rasters=(source_2018, anchor_2019))

            self.assertEqual(len(manifest["image_grid_roof_candidate_products"]), 1)
            candidate_product = manifest["image_grid_roof_candidate_products"][0]
            self.assertEqual(candidate_product["reference_source"], str(source_2018))
            self.assertEqual(candidate_product["candidate_shift_rendered_count"], 1)
            self.assertFalse(candidate_product["formal_ground_footprint_product"])
            with fiona.open(output / "building_lineage.gpkg",
                            layer="image_grid_roof_candidates") as layer:
                row = dict(next(iter(layer))["properties"])
            self.assertEqual(row["status"], "UNACCEPTED_IMAGE_GRID_SHIFT_ROOF_CANDIDATE")
            self.assertEqual(row["ground_status"], "NOT_A_GROUND_FOOTPRINT")
            with rasterio.open(output / candidate_product["candidate_tif"]) as raster:
                values = set(np.unique(raster.read(1)).tolist())
                self.assertIn(1, values)
                self.assertNotIn(2, values)
            with rasterio.open(output / "2018-04-29_ground_footprint.tif") as raster:
                self.assertIn(2, np.unique(raster.read(1)))
                self.assertNotIn(1, np.unique(raster.read(1)))

    def test_review_adapter_rejects_vector_or_support_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "fixed_grid.tif"
            reference.write_bytes(b"reference raster placeholder")
            evidence = root / "evidence.txt"
            evidence.write_text("control", encoding="utf-8")
            roof = box(10, 10, 20, 20)
            field = candidate_field("field", "2018-04-29", roof, reference, 1, 2)
            review_layer = root / "review.gpkg"
            self._write_review_layer(
                review_layer, [field], evidence, reference,
                overrides={"field": {"correction_dx_m": 999.0}})
            with self.assertRaisesRegex(ValueError, "vector differs"):
                attach_reviewed_alignment_cells([field], review_layer)

    def test_review_adapter_rejects_cells_without_source_component_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "fixed_grid.tif"
            reference.write_bytes(b"reference raster placeholder")
            evidence = root / "survey_control.txt"
            evidence.write_text("independent control record\n", encoding="utf-8")
            roof = box(10, 10, 20, 20)
            scoped = candidate_field("unknown-scope", "2018-04-29", roof,
                                     reference, 1.0, -1.0)
            unscoped_cell = replace(scoped.cells[0], metadata={})
            unscoped = replace(scoped, cells=(unscoped_cell,))
            review_layer = root / "review.gpkg"
            self._write_review_layer(review_layer, [unscoped], evidence, reference)

            with self.assertRaisesRegex(ValueError, "single-source-component support"):
                attach_reviewed_alignment_cells([unscoped], review_layer)


if __name__ == "__main__":
    unittest.main()
