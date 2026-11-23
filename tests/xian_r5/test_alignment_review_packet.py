from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

from shapely.geometry import box, mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xian_r5.alignment.review_packet import (
    DIRECT_METHOD,
    LEGACY_RELATIVE_METHOD,
    MULTI_ANCHOR_METHOD,
    RELATIVE_FIELDS,
    RELATIVE_METHOD,
    REVIEW_LAYER,
    R5_RELATIVE_REVIEW_SCOPE,
    R5_RELATIVE_REVIEW_SCOPE_ID,
    _apply_review_scope_allowlist,
    _write_vector,
    export_alignment_review_packet,
    _read_vector,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_test_png(
    path: Path,
    rgb: tuple[int, int, int],
    *,
    width: int = 1,
    height: int = 1,
) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    scanline = b"\x00" + bytes(rgb) * width
    image_data = zlib.compress(scanline * height)
    payload = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", image_data)
               + chunk(b"IEND", b""))
    path.write_bytes(payload)


def _write_source_gpkg(path: Path, rows: list[dict]) -> None:
    if importlib.util.find_spec("fiona") is not None:
        import fiona
        from shapely.geometry import mapping

        schema = {"geometry": "Polygon", "properties": {
            "field_id": "str:160", "cell_id": "str:100", "reference_date": "str:16",
            "reference_label": "str:80", "reference_source": "str:512",
            "reference_date_status": "str:48", "moving_date": "str:16",
            "status": "str:80", "method": "str:96", "source_ref": "str:1024",
            "correction_dx_m": "float", "correction_dy_m": "float",
            "fit_disp_m": "float", "position_unc_m": "float",
            "reviewed": "int", "independent_ground": "int", "accepted": "int",
            "applied": "int", "reference_sha256": "str:64",
            "scope": "str:48",
        }}
        with fiona.open(path, "w", driver="GPKG", layer="image_alignment_candidates",
                        schema=schema, crs="EPSG:32649") as sink:
            for row in rows:
                geometry = row.pop("geometry")
                sink.write({"geometry": mapping(geometry), "properties": row})
        return

    import geopandas as gpd
    import pyogrio

    records = [{**{key: value for key, value in row.items() if key != "geometry"},
                "geometry": row["geometry"]} for row in rows]
    frame = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:32649")
    pyogrio.write_dataframe(frame, path, layer="image_alignment_candidates",
                            driver="GPKG", geometry_type="Polygon")


class AlignmentReviewPacketTests(unittest.TestCase):
    def test_relative_tie_sidecar_schema_freezes_required_fields_and_rule(self):
        schema_path = ROOT / "schemas" / "xian_r5_relative_alignment_tie_sidecar.v2.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$id"], "xian_r5.relative_alignment_tie_sidecar.v2")
        self.assertEqual(
            schema["properties"]["review_contract_amendment"]["const"],
            "R5_RELATIVE_ALIGNMENT_REVIEW_CONTRACT_AMENDMENT_V2")
        self.assertEqual(schema["properties"]["uncertainty_rule"]["const"],
                         "R5_RELATIVE_ALIGNMENT_UNCERTAINTY_RULE_V1")
        self.assertEqual(schema["properties"]["coordinates_crs"]["const"], "EPSG:32649")
        self.assertEqual(
            schema["properties"]["uncertainty_semantics"]["const"],
            "RELATIVE_IMAGE_GRID_ONLY")
        self.assertEqual(schema["properties"]["absolute_position_accuracy"]["const"],
                         "UNKNOWN")
        self.assertTrue({
            "reference_source", "reference_source_sha256",
            "support_geometry_sha256", "uncertainty_semantics",
        }.issubset(schema["required"]))
        tie_schema = schema["properties"]["ties"]["items"]
        self.assertTrue({
            "tie_id", "field_id", "cell_id", "moving_date", "FIT_or_HOLDOUT",
            "independence_group", "feature_type", "precision_eligible",
            "precision_ineligibility_reason", "ge_x", "ge_y", "reference_x",
            "reference_y", "candidate_dx_m", "candidate_dy_m",
            "residual_dx_m", "residual_dy_m", "residual_m",
            "ge_localization_bound_m", "reference_localization_bound_m",
            "adjusted_bound_m", "localization_basis", "evidence_path",
            "evidence_sha256", "within_exact_support", "independence_group",
        }.issubset(tie_schema["required"]))

    def _make_source_run(self, root: Path) -> tuple[Path, dict[str, str]]:
        run = root / "source_run"
        run.mkdir()
        reference = root / "Xian_2019_TrueColor.tif"
        reference.write_bytes(b"fixed-grid-reference-raster")
        base_review_png = root / "base_review_2018-04-29.png"
        relative_review_png = root / "relative_review_2018-04-29_2018-11-01.png"
        _write_test_png(base_review_png, (32, 96, 160), width=240, height=112)
        _write_test_png(relative_review_png, (160, 96, 32))
        direct_source = root / "base_tie.json"
        direct_source.write_text(json.dumps({
            "schema_version": "xian_r5.base_tie_diagnostic.v1",
            "automatic_registration_promoted": False,
            "epochs": {
                "2018": {
                    "actual_date": "2018-04-29",
                    "review_png": str(base_review_png),
                    "submeter_tiles": {"tile_ids": [{
                        "tile_id": "420000:312000", "actual_date": "2018-04-29",
                        "path": str(base_review_png), "sha256": _sha256(base_review_png),
                        "support_polygon_epsg32649": mapping(box(0, 0, 200, 200)),
                    }]},
                    "base_path": str(reference),
                    "base_sha256": _sha256(reference),
                    "base_reference_year": 2019,
                    "base_grid": {
                        "crs": "EPSG:32649", "resolution": [10.0, 10.0],
                        "window": [0, 0, 20, 20],
                        "transform": [10.0, 0.0, 0.0, 0.0, -10.0, 200.0, 0.0, 0.0, 1.0],
                    },
                    "local_spatial_tie_diagnostic": {
                        "field_interpolated": False, "field_applied": False,
                        "cells": [
                            {"row": 0, "col": 1, "base_pixel_window": [10, 9, 11, 10],
                             "best": {"subpixel_peak": {
                                 "dx_px": -0.125, "dy_px": -0.25,
                                 "status": "SUBPIXEL_CORRELATION_PEAK_CANDIDATE_NOT_APPLIED"}}},
                            {"row": 1, "col": 1, "base_pixel_window": [11, 9, 12, 10],
                             "best": None},
                        ],
                    },
                },
            },
        }, indent=2) + "\n", encoding="utf-8")
        relative_source = root / "lightglue.json"
        relative_source.write_text("{}\n", encoding="utf-8")
        relative_mosaic = root / "mosaic.json"
        relative_mosaic.write_text("{}\n", encoding="utf-8")
        relative_gpkg = root / "relative_source.gpkg"
        relative_gpkg.write_bytes(b"relative-gpkg-source")
        relative_geometry = box(200, 200, 210, 210)
        insufficient_geometry = box(210, 200, 220, 210)
        relative_source.write_text(json.dumps({
            "schema_version": "xian_r5.local_tile_lightglue_field_diagnostic.v1",
            "case_id": "TEST-CASE",
            "scope": "FIXED_CASE_LOCAL_ONLY",
            "registration_promoted": False,
            "registration_applied": False,
            "field_interpolated": False,
            "actual_dates_in_order": ["2018-04-29", "2018-11-01"],
            "input_mosaic_diagnostic": str(relative_mosaic),
            "outputs": {
                "candidate_vector": str(relative_gpkg),
                "match_review_contact_sheets": [{
                    "from_date": "2018-04-29", "to_date": "2018-11-01",
                    "path": str(relative_review_png), "sha256": _sha256(relative_review_png),
                }],
            },
            "mask": {"ground_mask_reviewed": False},
            "pairwise_fields": [{
                "from_date": "2018-04-29", "to_date": "2018-11-01",
                "cell_count": 2, "field_interpolated": False, "field_applied": False,
                "cells": [
                    {
                        "cell_id": "r00c02", "status": "WEAK_LOCAL_CANDIDATE",
                        "matched_features_raw": 284, "eligible_matches": 8,
                        "inliers": 5, "inlier_ratio": 0.625, "coverage_4x4": 0.0625,
                        "median_residual_px": 0.86, "p90_residual_px": 1.41,
                        "candidate_ground_mask_fraction_from": 0.08,
                        "candidate_ground_mask_fraction_to": 0.08,
                        "correction_dx_m_candidate": 0.5,
                        "correction_dy_m_candidate": -1.5,
                        "ground_mask_reviewed": False,
                        "field_interpolated": False,
                        "applied_to_image_or_geometry": False,
                        "polygon_epsg32649": mapping(relative_geometry),
                    },
                    {
                        "cell_id": "r00c01",
                        "status": "INSUFFICIENT_GROUND_CANDIDATE_MATCHES",
                        "matched_features_raw": 300, "eligible_matches": 0,
                        "inliers": 0, "inlier_ratio": 0.0, "coverage_4x4": 0.0,
                        "median_residual_px": None, "p90_residual_px": None,
                        "candidate_ground_mask_fraction_from": 0.02,
                        "candidate_ground_mask_fraction_to": 0.02,
                        "correction_dx_m_candidate": None,
                        "correction_dy_m_candidate": None,
                        "ground_mask_reviewed": False,
                        "field_interpolated": False,
                        "applied_to_image_or_geometry": False,
                        "polygon_epsg32649": mapping(insufficient_geometry),
                    },
                ],
            }],
        }, indent=2) + "\n", encoding="utf-8")

        rows = [
            {
                "field_id": "2018-04-29_to_Xian_2019_TrueColor_DIRECT",
                "cell_id": "r0c1", "reference_date": None,
                "reference_label": "Xian_2019_TrueColor",
                "reference_source": str(reference),
                "reference_date_status": "REFERENCE_ACQUISITION_DATE_UNKNOWN",
                "moving_date": "2018-04-29",
                "status": "DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED",
                "method": DIRECT_METHOD,
                "source_ref": f"{direct_source}#epochs/2018/r0c1",
                "correction_dx_m": 1.25, "correction_dy_m": -2.5,
                "fit_disp_m": 0.8, "position_unc_m": None,
                "reviewed": 0, "independent_ground": 0, "accepted": 0, "applied": 0,
                "reference_sha256": _sha256(reference),
                "scope": "IMAGE_TO_REFERENCE_GRID",
                "geometry": box(100, 100, 110, 110),
            },
            {
                "field_id": "2018-04-29_to_Xian_2019_TrueColor_DIRECT",
                "cell_id": "r1c1", "reference_date": None,
                "reference_label": "Xian_2019_TrueColor",
                "reference_source": str(reference),
                "reference_date_status": "REFERENCE_ACQUISITION_DATE_UNKNOWN",
                "moving_date": "2018-04-29",
                "status": "INSUFFICIENT_MATCHES", "method": DIRECT_METHOD,
                "source_ref": f"{direct_source}#epochs/2018/r1c1",
                "correction_dx_m": None, "correction_dy_m": None,
                "fit_disp_m": None, "position_unc_m": None,
                "reviewed": 0, "independent_ground": 0, "accepted": 0, "applied": 0,
                "reference_sha256": _sha256(reference),
                "scope": "IMAGE_TO_REFERENCE_GRID",
                "geometry": box(110, 100, 120, 110),
            },
            {
                "field_id": "ALIKED_2018-04-29_to_2018-11-01", "cell_id": "r00c02",
                "reference_date": "2018-04-29", "reference_label": "2018-04-29",
                "reference_source": str(relative_mosaic),
                "reference_date_status": "ACTUAL_DATE_KNOWN",
                "moving_date": "2018-11-01",
                "status": "WEAK_LOCAL_CANDIDATE", "method": RELATIVE_METHOD,
                "source_ref": f"{relative_source}#pair:2018-04-29/2018-11-01/cell:r00c02",
                "correction_dx_m": 0.5, "correction_dy_m": -1.5,
                "fit_disp_m": None, "position_unc_m": None,
                "reviewed": 0, "independent_ground": 0, "accepted": 0, "applied": 0,
                "reference_sha256": None, "scope": "IMAGE_TO_REFERENCE_GRID",
                "geometry": box(200, 200, 210, 210),
            },
            {
                "field_id": "ALIKED_2018-04-29_to_2018-11-01", "cell_id": "r00c01",
                "reference_date": "2018-04-29", "reference_label": "2018-04-29",
                "reference_source": str(relative_mosaic),
                "reference_date_status": "ACTUAL_DATE_KNOWN",
                "moving_date": "2018-11-01",
                "status": "INSUFFICIENT_GROUND_CANDIDATE_MATCHES", "method": RELATIVE_METHOD,
                "source_ref": f"{relative_source}#pair:2018-04-29/2018-11-01/cell:r00c01",
                "correction_dx_m": None, "correction_dy_m": None,
                "fit_disp_m": None, "position_unc_m": None,
                "reviewed": 0, "independent_ground": 0, "accepted": 0, "applied": 0,
                "reference_sha256": None, "scope": "IMAGE_TO_REFERENCE_GRID",
                "geometry": box(210, 200, 220, 210),
            },
        ]
        _write_source_gpkg(run / "building_lineage.gpkg", rows)
        qa = {
            "schema_version": "xian_r5.reconstruction_engineering_qa.v1",
            "case_id": "TEST-CASE", "structural_qa": "PASS", "errors": [],
            "r4_status_sha256_unchanged": True,
            "tracked_input_hashes_unchanged": True,
            "pixel_input_hashes_unchanged": True,
            "base_raster_hashes_unchanged": True,
        }
        product = {
            "schema_version": "xian_r5.grounded_products.v7", "crs": "EPSG:32649",
            "image_alignment_field_applied": False,
        }
        receipt = {
            "schema_version": "xian_r5.grounded_fixed_case_run.v1",
            "case_id": "TEST-CASE", "image_grid_alignment_applied_count": 0,
            "direct_base_tie_field_sources": {
                "path": str(direct_source), "sha256": _sha256(direct_source),
            },
            "lightglue_local_field_sources": [
                {"path": str(relative_source), "sha256": _sha256(relative_source)},
                {"path": str(relative_mosaic), "sha256": _sha256(relative_mosaic)},
                {"path": str(relative_gpkg), "sha256": _sha256(relative_gpkg)},
            ],
        }
        for name, data in (("ENGINEERING_QA.json", qa),
                           ("PRODUCT_MANIFEST.json", product),
                           ("EVIDENCE_RECEIPT.json", receipt)):
            (run / name).write_text(json.dumps(data, indent=2), encoding="utf-8")
        before = {
            "base": _sha256(reference),
            "base_review_png": _sha256(base_review_png),
            "relative_review_png": _sha256(relative_review_png),
            "gpkg": _sha256(run / "building_lineage.gpkg"),
            "qa": _sha256(run / "ENGINEERING_QA.json"),
            "manifest": _sha256(run / "PRODUCT_MANIFEST.json"),
            "receipt": _sha256(run / "EVIDENCE_RECEIPT.json"),
        }
        return run, before

    def _add_multi_anchor_candidate(self, root: Path, run: Path) -> None:
        references = {}
        grid = {
            "crs": "EPSG:32649", "transform": [10.0, 0.0, 100.0, 0.0, -10.0, 100.0],
            "width": 20, "height": 20, "resolution": [10.0, 10.0],
        }
        for year in (2017, 2019):
            path = root / f"Xian_{year}_TrueColor.tif"
            if year == 2017:
                path.write_bytes(b"annual-reference-2017")
            else:
                path = root / "Xian_2019_TrueColor.tif"
            digest = _sha256(path)
            references[year] = (path, digest)

        diagnostics = []
        for year in (2017, 2019):
            path = root / f"base_tie_{year}.json"
            path.write_text(json.dumps({"reference_year": year}), encoding="utf-8")
            diagnostics.append({"path": str(path), "sha256": _sha256(path)})

        field_id = "2018-04-29_to_Xian_2019_MULTI_ANCHOR"
        cell_id = "r0c1"
        bounds = [100.0, 90.0, 110.0, 100.0]
        anchor_years = [2017, 2019]
        anchor_summary = {
            "anchor_count": 2,
            "reference_years": anchor_years,
            "anchor_vectors_m": {"2017": [1.0, -2.0], "2019": [1.5, -2.5]},
            "median_correlation": 0.7,
            "median_peak_margin": 0.08,
            "median_radial_deviation_m": 0.35,
            "dispersion_is_positional_uncertainty": False,
        }
        evidence = [{
            "reference_year_label": year,
            "reference_raster_path": str(path),
            "reference_raster_sha256": digest,
            "cell_support_bounds_epsg32649": bounds,
            "reference_grid_transform_local": grid["transform"],
            "reference_component_id_status": "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT",
        } for year, (path, digest) in references.items()]
        integrity = {row["path"]: row["sha256"] for row in diagnostics}
        integrity.update({str(path): digest for path, digest in references.values()})
        report = {
            "schema_version": "xian_r5.multi_anchor_base_tie_candidates.v1",
            "case_id": "TEST-CASE",
            "reference_years": [2017, 2019],
            "canonical_reference_raster": str(references[2019][0]),
            "canonical_reference_sha256": references[2019][1],
            "reference_rasters": [
                {"year": year, "path": str(path), "sha256": digest,
                 "grid": {**grid, "sha256": digest}}
                for year, (path, digest) in references.items()],
            "source_diagnostics": diagnostics,
            "field_cell_summaries": [{
                "field_id": field_id,
                "moving_date": "2018-04-29",
                "cell_id": cell_id,
                "status": "MULTI_ANCHOR_BASE_TIE_CANDIDATE_NOT_REVIEWED",
                "correction_dx_m_candidate": 1.25,
                "correction_dy_m_candidate": -2.5,
                "anchor_summary": anchor_summary,
                "reference_window_evidence": evidence,
            }],
            "field_interpolated": False,
            "field_applied": False,
            "accepted_cells": 0,
            "dispersion_is_positional_uncertainty": False,
            "integrity_files_sha256": integrity,
        }
        receipt_path = run / "EVIDENCE_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["multi_anchor_base_tie_source_report"] = report
        receipt["multi_anchor_base_tie_field_count"] = 1
        receipt["multi_anchor_base_tie_numeric_cell_count"] = 1
        receipt["multi_anchor_base_tie_accepted_cell_count"] = 0
        receipt["multi_anchor_base_tie_field_applied"] = False
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

        source_path = run / "building_lineage.gpkg"
        existing, _ = _read_vector(source_path, "image_alignment_candidates")
        rows = [{**record["properties"], "geometry": record["geometry"]}
                for record in existing]
        rows.append({
            "field_id": field_id,
            "cell_id": cell_id,
            "reference_date": None,
            "reference_label": "Xian_2019_TrueColor_MULTI_ANCHOR",
            "reference_source": str(references[2019][0]),
            "reference_date_status": "REFERENCE_ACQUISITION_DATE_UNKNOWN",
            "moving_date": "2018-04-29",
            "status": "MULTI_ANCHOR_BASE_TIE_CANDIDATE_NOT_REVIEWED",
            "method": MULTI_ANCHOR_METHOD,
            "source_ref": "multi-anchor:TEST-CASE:2018-04-29:r0c1:2017,2019",
            "correction_dx_m": 1.25,
            "correction_dy_m": -2.5,
            "fit_disp_m": 0.35,
            "position_unc_m": None,
            "reviewed": 0,
            "independent_ground": 0,
            "accepted": 0,
            "applied": 0,
            "reference_sha256": references[2019][1],
            "scope": "IMAGE_TO_REFERENCE_GRID",
            "geometry": box(*bounds),
        })
        _write_source_gpkg(source_path, rows)

    def test_exports_separate_pending_absolute_and_relative_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, before = self._make_source_run(root)
            output = root / "review_package"
            manifest = export_alignment_review_packet(source, output)

            self.assertEqual(manifest["absolute_base_review"]["numeric_candidate_count"], 1)
            self.assertEqual(manifest["relative_date_pair_diagnostics"]["numeric_candidate_count"], 1)
            self.assertEqual(manifest["source_hash_recheck"], "PASS")
            review_rows, review_crs = _read_vector(
                output / "absolute_base_review_template.gpkg", REVIEW_LAYER)
            self.assertIn("32649", review_crs)
            self.assertEqual(len(review_rows), 1)
            review = review_rows[0]["properties"]
            self.assertTrue({
                "field_id", "cell_id", "moving_date", "reference_source",
                "reference_source_sha256", "candidate_source_ref", "correction_dx_m",
                "correction_dy_m", "independent_ground_evidence",
                "ground_evidence_type", "review_basis", "alignment_evidence_type",
                "reviewed", "positional_uncertainty_m",
                "positional_uncertainty_semantics", "uncertainty_semantics",
                "relative_registration_uncertainty_m", "nonbuilding_tie_pairs",
                "fit_max_adjusted_bound_m", "holdout_max_adjusted_bound_m",
                "holdout_tie_pairs", "relative_ties_within_support_cell",
                "reviewer_id", "reviewed_at", "evidence_path", "evidence_sha256",
            }.issubset(review))
            self.assertEqual(review["review_state"], "PENDING")
            self.assertEqual(review["independent_ground_evidence"], 0)
            self.assertEqual(review["reviewed"], 0)
            self.assertEqual(review["accepted"], 0)
            self.assertEqual(review["applied"], 0)
            uncertainty = review["positional_uncertainty_m"]
            self.assertTrue(uncertainty is None or math.isnan(uncertainty))
            self.assertEqual(review["correction_dx_m"], 1.25)
            relative_rows, _ = _read_vector(
                output / "relative_date_pair_candidates.gpkg", "relative_date_pair_candidates")
            self.assertEqual(len(relative_rows), 1)
            self.assertEqual(relative_rows[0]["properties"]["review_state"],
                             "RELATIVE_ONLY_NOT_BASE_ANCHORED")
            self.assertEqual(relative_rows[0]["properties"]["matched_features_raw"], 284)
            self.assertEqual(relative_rows[0]["properties"]["eligible_matches"], 8)
            self.assertEqual(relative_rows[0]["properties"]["inliers"], 5)
            self.assertEqual(relative_rows[0]["properties"]["coverage_4x4"], 0.0625)
            self.assertEqual(relative_rows[0]["properties"]["ground_mask_reviewed"], 0)
            self.assertEqual(manifest["schema_version"], "xian_r5.alignment_review_packet.v6")
            self.assertEqual(manifest["base_grid_alignment_review"]["status"],
                             manifest["absolute_base_review"]["status"])
            self.assertTrue(manifest["absolute_base_review"]["legacy_manifest_key"])

            materials = manifest["review_materials"]
            self.assertEqual(materials["context_image_count"], 2)
            self.assertEqual(materials["tie_pairs_template_row_count"], 6)
            self.assertFalse(materials["blank_slots_are_evidence"])
            self.assertEqual(materials["candidate_cell_context_image_count"], 1)
            self.assertEqual(materials["moving_source_component_image_count"], 1)
            cell_context = materials["candidate_cell_context_images"][0]
            self.assertEqual(cell_context["moving_source_component_scope"],
                             "SINGLE_SOURCE_COMPONENT_PER_CANDIDATE_CELL")
            self.assertEqual(cell_context["moving_source_component_tiles"][0]["tile_id"],
                             "420000:312000")
            self.assertEqual(_sha256(output / cell_context["path"]), cell_context["sha256"])
            for record in materials["context_images"]:
                copied = output / record["path"]
                self.assertEqual(_sha256(copied), record["sha256"])
                self.assertEqual(_sha256(Path(record["source_path"])), record["source_sha256"])
            with (output / materials["tie_pairs_template"]).open(
                    encoding="utf-8", newline="") as template_file:
                tie_rows = list(csv.DictReader(template_file))
            self.assertEqual(len(tie_rows), 6)
            self.assertEqual([row["FIT_or_HOLDOUT"] for row in tie_rows],
                             ["FIT"] * 3 + ["HOLDOUT"] * 3)
            self.assertEqual({row["pair_index"] for row in tie_rows}, {"1", "2", "3"})
            self.assertTrue(all(row["field_id"] == review["field_id"]
                                and row["cell_id"] == review["cell_id"]
                                and row["tie_id"] == ""
                                and row["ge_x"] == ""
                                and row["reference_y"] == ""
                                and row["ge_localization_bound_m"] == ""
                                and row["reference_localization_bound_m"] == ""
                                and row["independence_group"] == ""
                                for row in tie_rows))
            self.assertIn("None of these images is a human review verdict",
                          (output / "README.md").read_text(encoding="utf-8"))
            for relative_path, digest in manifest["created_files_sha256"].items():
                self.assertEqual(_sha256(output / relative_path), digest)

            reference = Path(review["reference_source"])
            self.assertEqual(_sha256(reference), review["reference_source_sha256"])
            self.assertEqual(_sha256(source / "building_lineage.gpkg"), before["gpkg"])
            self.assertEqual(_sha256(source / "ENGINEERING_QA.json"), before["qa"])
            self.assertEqual(_sha256(source / "PRODUCT_MANIFEST.json"), before["manifest"])
            self.assertEqual(_sha256(source / "EVIDENCE_RECEIPT.json"), before["receipt"])
            with self.assertRaises(FileExistsError):
                export_alignment_review_packet(source, output)

    def test_frozen_review_scope_is_exact_nine_and_does_not_fall_back(self):
        rows = []
        for index, (cell_id, moving_date) in enumerate(R5_RELATIVE_REVIEW_SCOPE):
            rows.append({
                "feature_id": index,
                "geometry": box(index, 0, index + 1, 1),
                "properties": {
                    "field_id": f"frozen-{index}",
                    "cell_id": cell_id,
                    "moving_date": moving_date,
                },
            })
        for index in range(9):
            rows.append({
                "feature_id": index + 9,
                "geometry": box(index, 2, index + 1, 3),
                "properties": {
                    "field_id": f"out-of-scope-{index}",
                    "cell_id": f"c0007_other{index:02d}",
                    "moving_date": "2018-04-29",
                },
            })

        selected, relative, scope = _apply_review_scope_allowlist(
            rows, [{"properties": {"cell_id": "unrelated"}}],
            R5_RELATIVE_REVIEW_SCOPE_ID)
        self.assertEqual(len(selected), 9)
        self.assertEqual(relative, [])
        self.assertEqual(
            [(row["properties"]["cell_id"], row["properties"]["moving_date"])
             for row in selected],
            list(R5_RELATIVE_REVIEW_SCOPE))
        self.assertEqual(scope["matched_date_cell_count"], 9)
        self.assertFalse(scope["candidate_fallback"])
        self.assertEqual(scope["relative_date_pair_diagnostics"],
                         "OMITTED_BY_REVIEW_SCOPE")

        missing = [
            row for row in rows
            if (row["properties"]["cell_id"], row["properties"]["moving_date"])
            != R5_RELATIVE_REVIEW_SCOPE[-1]
        ]
        with self.assertRaisesRegex(ValueError, "matched 0"):
            _apply_review_scope_allowlist(
                missing, [], R5_RELATIVE_REVIEW_SCOPE_ID)
        duplicated = rows + [dict(rows[0])]
        with self.assertRaisesRegex(ValueError, "matched 2"):
            _apply_review_scope_allowlist(
                duplicated, [], R5_RELATIVE_REVIEW_SCOPE_ID)
        with self.assertRaisesRegex(ValueError, "Unknown alignment review scope"):
            _apply_review_scope_allowlist(rows, [], "UNFROZEN_SCOPE")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self._make_source_run(root)
            output = root / "must_not_be_created"
            with self.assertRaisesRegex(ValueError, "matched 0"):
                export_alignment_review_packet(
                    source, output,
                    review_scope_allowlist=R5_RELATIVE_REVIEW_SCOPE_ID)
            self.assertFalse(output.exists())

    def test_accepts_one_structurally_valid_independent_qa_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self._make_source_run(root)
            canonical_qa = source / "ENGINEERING_QA.json"
            audit_qa = source / "ENGINEERING_QA_INDEPENDENT_AUDIT_20260929.json"
            canonical_qa.rename(audit_qa)

            output = root / "audit_qa_review_package"
            manifest = export_alignment_review_packet(source, output)

            self.assertIn(str(audit_qa.resolve()), manifest["source_files_sha256"])
            self.assertEqual(manifest["source_hash_recheck"], "PASS")
            self.assertEqual(manifest["absolute_base_review"]["accepted_count"], 0)
            self.assertEqual(manifest["absolute_base_review"]["applied_count"], 0)

    def test_rejects_changed_intersecting_source_component_tile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self._make_source_run(root)
            receipt = json.loads((source / "EVIDENCE_RECEIPT.json").read_text(encoding="utf-8"))
            direct_path = Path(receipt["direct_base_tie_field_sources"]["path"])
            diagnostic = json.loads(direct_path.read_text(encoding="utf-8"))
            review_png = Path(diagnostic["epochs"]["2018"]["review_png"])
            _write_test_png(review_png, (240, 40, 80), width=240, height=112)

            output = root / "changed_source_tile_review_package"
            with self.assertRaisesRegex(ValueError, "source tile hash mismatch"):
                export_alignment_review_packet(source, output)
            self.assertFalse(output.exists())

    def test_exports_multi_anchor_cells_as_pending_not_ground_control(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self._make_source_run(root)
            self._add_multi_anchor_candidate(root, source)
            output = root / "multi_anchor_review_package"

            manifest = export_alignment_review_packet(source, output)

            absolute = manifest["absolute_base_review"]
            self.assertEqual(absolute["numeric_candidate_count"], 2)
            self.assertEqual(absolute["direct_single_anchor_candidate_count"], 1)
            self.assertEqual(absolute["multi_anchor_candidate_count"], 1)
            self.assertEqual(absolute["accepted_count"], 0)
            self.assertEqual(absolute["applied_count"], 0)
            self.assertEqual(manifest["multi_anchor_base_tie_source"]["status"], "PROVIDED")
            self.assertFalse(manifest["multi_anchor_base_tie_source"]["dispersion_is_positional_uncertainty"])

            rows, _ = _read_vector(output / absolute["path"], absolute["layer"])
            candidates = [row["properties"] for row in rows
                          if row["properties"]["method"] == MULTI_ANCHOR_METHOD]
            self.assertEqual(len(candidates), 1)
            candidate = candidates[0]
            self.assertEqual(candidate["candidate_source_ref"],
                             "multi-anchor:TEST-CASE:2018-04-29:r0c1:2017,2019")
            self.assertEqual(candidate["anchor_count"], 2)
            self.assertEqual(candidate["anchor_years"], "2017,2019")
            self.assertEqual(candidate["fit_dispersion_kind"],
                             "CROSS_ANCHOR_MEDIAN_RADIAL_DEVIATION_NOT_POSITIONAL_UNCERTAINTY")
            self.assertEqual(candidate["review_state"], "PENDING")
            self.assertEqual(candidate["independent_ground_evidence"], 0)
            self.assertEqual(candidate["accepted"], 0)
            self.assertEqual(candidate["applied"], 0)
            self.assertEqual(manifest["source_hash_recheck"], "PASS")
            readme = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("Cross-anchor dispersion is not positional uncertainty", readme)

    def test_direct_only_packet_marks_relative_diagnostic_not_provided(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self._make_source_run(root)
            receipt_path = source / "EVIDENCE_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt.pop("lightglue_local_field_sources")
            receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            product_path = source / "PRODUCT_MANIFEST.json"
            product = json.loads(product_path.read_text(encoding="utf-8"))
            product["schema_version"] = "xian_r5.grounded_products.v4"
            product_path.write_text(json.dumps(product, indent=2), encoding="utf-8")
            source_rows, _ = _read_vector(source / "building_lineage.gpkg", "image_alignment_candidates")
            direct_only_rows = [
                {**record["properties"], "geometry": record["geometry"]}
                for record in source_rows
                if record["properties"].get("method") == DIRECT_METHOD
            ]
            _write_source_gpkg(source / "building_lineage.gpkg", direct_only_rows)

            output = root / "direct_only_review_package"
            manifest = export_alignment_review_packet(source, output)

            relative = manifest["relative_date_pair_diagnostics"]
            self.assertEqual(relative["status"], "NOT_PROVIDED")
            self.assertEqual(relative["source_file_count"], 0)
            self.assertIsNone(relative["numeric_candidate_count"])
            self.assertIn("NOT_A_ZERO_CANDIDATE_RESULT", relative["semantics"])
            relative_rows, relative_crs = _read_vector(
                output / relative["path"], relative["layer"])
            self.assertEqual(relative_rows, [])
            self.assertIn("32649", relative_crs)
            readme = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("no relative-date diagnostic was supplied", readme)
            self.assertIn("not evidence that the candidate count is zero", readme)
            direct_rows, _ = _read_vector(
                output / "absolute_base_review_template.gpkg", REVIEW_LAYER)
            self.assertEqual(len(direct_rows), 1)
            self.assertEqual(direct_rows[0]["properties"]["review_state"], "PENDING")

    def test_v13_relative_only_packet_preserves_weak_cell_without_base_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self._make_source_run(root)
            product_path = source / "PRODUCT_MANIFEST.json"
            product = json.loads(product_path.read_text(encoding="utf-8"))
            product["schema_version"] = "xian_r5.grounded_products.v13"
            product_path.write_text(json.dumps(product, indent=2), encoding="utf-8")
            receipt_path = source / "EVIDENCE_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt.pop("direct_base_tie_field_sources")
            receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

            source_gpkg = source / "building_lineage.gpkg"
            source_rows, _ = _read_vector(source_gpkg, "image_alignment_candidates")
            relative_only_rows = [
                {**record["properties"], "geometry": record["geometry"]}
                for record in source_rows
                if record["properties"].get("method") == RELATIVE_METHOD
            ]
            self.assertGreaterEqual(len(relative_only_rows), 1)
            for row in relative_only_rows:
                row["method"] = LEGACY_RELATIVE_METHOD
            _write_source_gpkg(source_gpkg, relative_only_rows)

            output = root / "v13_relative_only_review_package"
            manifest = export_alignment_review_packet(source, output)

            self.assertEqual(manifest["schema_version"], "xian_r5.alignment_review_packet.v6")
            self.assertEqual(manifest["base_grid_alignment_review"]["status"],
                             manifest["absolute_base_review"]["status"])
            self.assertTrue(manifest["absolute_base_review"]["legacy_manifest_key"])
            absolute = manifest["absolute_base_review"]
            self.assertEqual(absolute["status"], "NO_NUMERIC_ABSOLUTE_CANDIDATES")
            self.assertEqual(absolute["numeric_candidate_count"], 0)
            self.assertEqual(absolute["reference_rasters_sha256"], {})
            self.assertEqual(manifest["reference_hash_recheck"],
                             "NOT_APPLICABLE_NO_ABSOLUTE_BASE_CANDIDATES")
            relative = manifest["relative_date_pair_diagnostics"]
            self.assertEqual(relative["status"], "PROVIDED")
            self.assertEqual(relative["numeric_candidate_count"], 1)
            rows, _ = _read_vector(output / relative["path"], relative["layer"])
            self.assertEqual(len(rows), 1)
            candidate = rows[0]["properties"]
            self.assertEqual(candidate["candidate_status"], "WEAK_LOCAL_CANDIDATE")
            self.assertEqual(candidate["review_state"], "RELATIVE_ONLY_NOT_BASE_ANCHORED")
            self.assertEqual(candidate["method"], LEGACY_RELATIVE_METHOD)
            self.assertEqual(candidate["inliers"], 5)
            self.assertEqual(candidate["accepted"], 0)
            self.assertEqual(candidate["applied"], 0)
            self.assertEqual(manifest["source_hash_recheck"], "PASS")
            readme = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("cannot independently anchor an observation", readme)
            self.assertIn("Date-pair relative candidates remain unaccepted and unapplied", readme)

    def test_rejects_malformed_optional_relative_source_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self._make_source_run(root)
            receipt_path = source / "EVIDENCE_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["lightglue_local_field_sources"] = [{"path": "missing-hash.json"}]
            receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Malformed candidate source receipt"):
                export_alignment_review_packet(source, root / "invalid_review_package")

    def test_refuses_source_candidates_already_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self._make_source_run(root)
            source_gpkg = source / "building_lineage.gpkg"
            rows, crs = _read_vector(source_gpkg, "image_alignment_candidates")
            self.assertIn("32649", crs)
            # Rebuild this isolated synthetic source with a bypassed apply flag.
            fields = [dict(record["properties"], geometry=record["geometry"])
                      for record in rows]
            fields[0]["accepted"] = 1
            # The original vector's relevant properties are preserved; use its geometry.
            backup = root / "original.gpkg"
            source_gpkg.replace(backup)
            _write_source_gpkg(source_gpkg, fields)
            with self.assertRaisesRegex(ValueError, "already reviewed or applied"):
                export_alignment_review_packet(source, root / "should_not_exist")

    def test_empty_relative_layer_retains_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "empty_relative.gpkg"
            _write_vector(path, "relative_date_pair_candidates", [], RELATIVE_FIELDS)
            records, crs = _read_vector(path, "relative_date_pair_candidates")
            self.assertEqual(records, [])
            self.assertIn("32649", crs)


if __name__ == "__main__":
    unittest.main()
