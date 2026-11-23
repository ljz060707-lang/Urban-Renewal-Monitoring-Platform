"""Integration regressions at existing geometry, identity, state and GT seams."""
import csv
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import fiona
import numpy as np
from affine import Affine
from shapely.geometry import box, mapping, shape

from xian_r5.integration import (
    CONTRACT, adapt_sam_group, attach_single_date_links, build_gt10m_shadow,
    run_r5_state, sha256, verify_artifacts,
)
from xian_r5.label_transfer import Grid, export_raster, load_rasters
from xian_r5.alignment.reviewed_field import recheck_reviewed_alignment_sources
from xian_r5.single_date_identity import REQUIRED_SOURCE_GATES


def observation(key="original_component", day="2018-11-01", geometry=None):
    geometry = geometry if geometry is not None else box(0, 0, 10, 10)
    return {"contract_version": CONTRACT, "observation_id": key, "original_id": key,
            "geometry": mapping(geometry), "geometry_sha256": hashlib.sha256(geometry.wkb).hexdigest(),
            "actual_date": day, "source_path": "original_native_components.gpkg",
            "source_tile_ids": ["1:2:20"], "source_paths": ["original.jpg"],
            "source_hashes": ["a" * 64], "date_status": "REGISTRY_DATE_COHERENT",
            "building_object_ids": ["existing_object"], "physical_id": None,
            "registered_geometry": None, "ground_footprint": None,
            "GT_USED_FOR_GEOMETRY": False}


def a_receipt():
    return {"ACCEPTED_LOCAL_REGIONS": 0, "APPLIED_SHADOW_LOCAL_REGIONS": 0}


class IntegrationContractTests(unittest.TestCase):
    def test_utf8_bom_existing_identity_csv_preserves_source_link_and_competitor(self):
        records = [observation()]
        records[0]["building_object_ids"] = []
        geometry = records[0]["geometry"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            graph = root / "original_graph.gpkg"
            schema = {"geometry": "Polygon", "properties": {
                "node_id": "str", "original_id": "str", "physical_id": "str"}}
            with fiona.open(graph, "w", driver="GPKG", layer="roof_instance_reference",
                            schema=schema, crs="EPSG:32649") as target:
                target.write({"geometry": geometry, "properties": {
                    "node_id": "original_component", "original_id": "original_component", "physical_id": None}})
            with fiona.open(graph, "w", driver="GPKG", layer="building_object_reference",
                            schema=schema, crs="EPSG:32649") as target:
                for key in ("existing_object", "competing_object"):
                    target.write({"geometry": geometry, "properties": {
                        "node_id": key, "original_id": key, "physical_id": None}})
            fields = ["observation_id", "building_object_id", "physical_id", "GT_USED_FOR_GEOMETRY",
                      "source_actual_date", "link_level"]
            ledger = root / "original_links.csv"
            with ledger.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for key, level in (("existing_object", "TRACEABLE_SINGLE_DATE_LINK"),
                                   ("competing_object", "CANDIDATE_ONLY")):
                    writer.writerow(dict(zip(fields, ["original_component", key, "", "False", "2018-11-01", level])))
            proof = {"observation_id": "original_component", "building_object_id": "existing_object",
                     "link_level": "TRACEABLE_SINGLE_DATE_LINK", "physical_id": None,
                     "source_actual_date": "2018-11-01", "GT_USED_FOR_IDENTITY": True,
                     "GT_USED_FOR_GEOMETRY": False,
                     "source_object_records": [], "geometry_lineage": "original native replay",
                     "proof_scope": "SINGLE_DATE_ONLY",
                     "evidence": {**dict.fromkeys(REQUIRED_SOURCE_GATES, True),
                                  "native_exact_object_count": 1, "source_conflicts": []}}
            links, count = attach_single_date_links(records, ledger, graph, [{"records": [proof]}])
            self.assertEqual(count, 1)
            self.assertEqual(len(links), 2)
            self.assertEqual(records[0]["building_object_ids"], ["existing_object"])
            self.assertTrue(all(r["physical_id"] is None for r in links))
            self.assertTrue(links[0]["GT_USED_FOR_IDENTITY"])

    def test_single_date_link_is_not_physical_identity_or_stable(self):
        rows = [observation()]
        state, _ = run_r5_state(rows, a_receipt(), ["reference.tif"])
        self.assertEqual(rows[0]["building_object_ids"], ["existing_object"])
        self.assertIsNone(state.presence[0]["physical_id"])
        self.assertEqual(state.presence[0]["presence"], "U")
        self.assertEqual(state.actual_date_order, ["2018-11-01"])
        self.assertEqual(state.transitions, [])

    def test_missing_detection_does_not_generate_absence(self):
        rows = [observation("before"), observation("other", "2019-12-02", box(500, 500, 510, 510))]
        state, _ = run_r5_state(rows, a_receipt(), ["reference.tif"])
        self.assertTrue(all(r["presence"] == "U" for r in state.presence))
        self.assertFalse(any(r.get("state") == "DEMOLITION" for r in state.transitions))

    def test_unknown_date_is_not_replaced_by_nominal_year(self):
        rows = [observation(day=None)]
        state, _ = run_r5_state(rows, a_receipt(), ["reference.tif"])
        self.assertIsNone(state.presence[0]["actual_date"])
        self.assertEqual(state.actual_date_order, [])

    def test_unverified_zero_transform_stays_null_and_geometry_unchanged(self):
        rows = [observation()]
        before = shape(rows[0]["geometry"]).wkb
        run_r5_state(rows, a_receipt(), ["reference.tif"])
        self.assertIsNone(rows[0]["registered_geometry"])
        self.assertIsNone(rows[0]["applied_vector_m"])
        self.assertEqual(shape(rows[0]["geometry"]).wkb, before)

    def test_nonzero_acceptance_flag_does_not_authorize_application(self):
        with self.assertRaisesRegex(ValueError, "reviewed-field"):
            run_r5_state([observation()], {**a_receipt(), "ACCEPTED_LOCAL_REGIONS": 1}, ["reference.tif"])

    def test_string_or_bool_counter_is_not_verified_zero(self):
        for value in (False, "0", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_r5_state([observation()], {**a_receipt(), "ACCEPTED_LOCAL_REGIONS": value}, ["reference.tif"])

    def test_historical_track_cannot_fill_physical_identity(self):
        row = observation()
        row["physical_id"] = "TRK_R4X_old"
        with self.assertRaises(ValueError):
            run_r5_state([row], a_receipt(), ["reference.tif"])

    def test_gt_geometry_or_prealigned_geometry_cannot_enter_native_handoff(self):
        for name, value in (("GT_USED_FOR_GEOMETRY", True), ("registered_geometry", mapping(box(0, 0, 1, 1))),
                            ("ground_footprint", mapping(box(0, 0, 1, 1)))):
            row = observation()
            row[name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                run_r5_state([row], a_receipt(), ["reference.tif"])

    def test_geometry_hash_disagreement_stops_state_engine(self):
        row = observation()
        row["geometry"] = mapping(box(1, 1, 11, 11))
        with self.assertRaises(ValueError):
            run_r5_state([row], a_receipt(), ["reference.tif"])

    def test_duplicate_original_observation_is_not_silently_deduplicated(self):
        with self.assertRaises(ValueError):
            run_r5_state([observation(), observation()], a_receipt(), ["reference.tif"])

    def test_manual_gt_and_student_predictions_rejected_as_automatic_inputs(self):
        for role in ("MANUAL_ROOF_REFERENCE", "STUDENT_PREDICTION", "PREDICTED_CHANGE"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                verify_artifacts({"forbidden": {"path": str(Path("never_read").resolve()),
                                                "sha256": "a" * 64, "role": role}})

    def test_input_mutation_cannot_resume_with_same_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            path.write_text("original", encoding="utf-8")
            pins = {"input": {"path": str(path), "sha256": sha256(path), "role": "MODULE_A_RECEIPT"}}
            verify_artifacts(pins)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_artifacts(pins)

    def test_real_fraction_path_retains_unknown_background_and_full_denominator(self):
        rows = [observation(geometry=box(0, 0, 10, 10))]
        state, _ = run_r5_state(rows, a_receipt(), ["reference.tif"])
        grid = Grid(Affine(10, 0, 0, 0, -10, 10), 2, 1, "EPSG:32649")
        packets = [{"region_id": "original_region", "source_observed_date": "2018-11-01",
                    "exact_support": mapping(box(0, 0, 20, 10))}]
        fractions, eligible, supervised = build_gt10m_shadow(rows, packets, box(0, 0, 5, 10), grid, state)
        self.assertAlmostEqual(fractions["raw_fraction"][0, 0], .5)
        self.assertAlmostEqual(fractions["scope_fraction"][0, 0, 0], .5)
        self.assertEqual(fractions["raw_fraction"][0, 1], -9999)
        self.assertEqual(eligible["counts"]["eligible"], 0)
        np.testing.assert_array_equal(supervised["unknown"], 1)
        np.testing.assert_array_equal(supervised["weight"], 0)
        self.assertTrue((fractions["candidate_fraction"] == -9999).all())

    def test_moved_geometry_requires_qualified_handoff(self):
        rows = [observation()]
        state, _ = run_r5_state(rows, a_receipt(), ["reference.tif"])
        rows[0]["applied_vector_m"] = [0, -10]
        with self.assertRaises(ValueError):
            build_gt10m_shadow(rows, [], box(0, 0, 10, 10), Grid(Affine.identity(), 1, 1, "EPSG:32649"), state)

    def test_affine_order_and_dataloader_grid_are_checked(self):
        grid = Grid(Affine(10, 0, 308460, 0, -10, 3794750), 2, 2, "EPSG:32649")
        self.assertEqual(grid.affine.to_gdal(), (308460, 10, 0, 3794750, 0, -10))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            export_raster(root / "a.tif", np.zeros((1, 2, 2), np.uint8), grid, ["ignore"], root=root)
            other = Grid(Affine(10, 0, 308461, 0, -10, 3794750), 2, 2, "EPSG:32649")
            export_raster(root / "b.tif", np.zeros((1, 2, 2), np.uint8), other, ["ignore"], root=root)
            with self.assertRaises(ValueError):
                load_rasters([root / "a.tif", root / "b.tif"])

    def test_existing_v2_review_to_subpixel_roof_to_fraction_keeps_identity_unresolved(self):
        # Reuse existing explicitly synthetic review fixtures. These are only
        # temporary unit-test artifacts, never date-cell review deliverables.
        fixture_path = Path(__file__).parent / "xian_r5/test_reviewed_image_alignment.py"
        spec = importlib.util.spec_from_file_location("synthetic_review_fixture", fixture_path)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "synthetic_reference.tif"
            import rasterio
            with rasterio.open(reference, "w", driver="GTiff", width=4, height=4,
                               count=1, dtype="uint8", crs="EPSG:32649",
                               transform=Affine(10, 0, 90, 0, -10, 120)) as target:
                target.write(np.zeros((1, 4, 4), np.uint8))
            evidence = root / "SYNTHETIC_UNIT_TEST_EVIDENCE.txt"
            evidence.write_text("SYNTHETIC UNIT TEST ONLY", encoding="utf-8")
            roof = box(100, 100, 110, 110)
            field = fixture.candidate_field("synthetic", "2018-11-01", roof, reference, 2.5, -1.25)
            review = root / "synthetic_review.gpkg"
            fixture.ReviewedImageAlignmentTests()._write_review_layer(
                review, [field], evidence, reference, overrides={"synthetic": {
                    "independent_ground_evidence": 0, "ground_evidence_type": "",
                    "review_basis": "RELATIVE_REFERENCE_IMAGE_FEATURES",
                    "alignment_evidence_type": "STABLE_NONBUILDING_IMAGE_FEATURE_TIES"}})
            row = observation(geometry=roof)
            row["source_support_complete"] = True
            records = [row]
            state, report = run_r5_state(records, {**a_receipt(), "ACCEPTED_LOCAL_REGIONS": 1},
                                        [str(reference)], candidate_fields=[field], reviewed_layer=review)
            self.assertEqual(records[0]["applied_vector_m"], [2.5, -1.25])
            self.assertEqual(shape(records[0]["geometry"]).wkb, roof.wkb)
            self.assertIsNone(state.observations[0].ground_footprint)
            self.assertIsNone(state.observations[0].physical_id)
            self.assertEqual(state.presence[0]["presence"], "U")
            self.assertAlmostEqual(report["reviewed_cells"][0]["relative_registration_uncertainty_m"], 1.1)
            self.assertFalse(report["absolute_position_accuracy_claimed"])
            grid = Grid(Affine(10, 0, 90, 0, -10, 120), 4, 4, "EPSG:32649")
            packets = [{"region_id": "synthetic_region", "source_observed_date": "2018-11-01",
                        "exact_support": mapping(box(90, 80, 130, 120))}]
            result, eligibility, supervised = build_gt10m_shadow(records, packets, box(90, 80, 130, 120), grid, state)
            self.assertAlmostEqual(np.maximum(result["candidate_fraction"], 0).sum(), 1.)
            self.assertEqual(eligibility["counts"]["eligible"], 0)
            self.assertFalse(supervised["weight"].any())
            partial = observation(geometry=roof)
            partial["source_support_complete"] = False
            partial_state, _ = run_r5_state([partial], {**a_receipt(), "ACCEPTED_LOCAL_REGIONS": 1},
                                            [str(reference)], candidate_fields=[field], reviewed_layer=review)
            self.assertIsNone(partial_state.observations[0].registered_roof)
            self.assertIsNone(partial["applied_vector_m"])
            evidence.write_text("CHANGED SYNTHETIC EVIDENCE", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                recheck_reviewed_alignment_sources(report)

    def test_preattached_accepted_cell_cannot_bypass_review_recomputation(self):
        from xian_r5.alignment.field import DisplacementCell, LocalDisplacementField
        cell = DisplacementCell("synthetic", box(-10, -10, 20, 20), 1., 0., None,
                                "LOCAL_CANDIDATE_NOT_REVIEWED", "SYNTHETIC_TEST", "test:source", reviewed=True)
        field = LocalDisplacementField("synthetic", None, "2018-11-01", (cell,),
                                        reference_label="test", reference_source="reference.tif",
                                        reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            run_r5_state([observation()], a_receipt(), ["reference.tif"], candidate_fields=[field])

    def sam(self):
        mask = np.zeros((3, 4, 4), np.uint8)
        mask[:, 1:3, 1:3] = 1
        scores = np.array([.9, .8, .7])
        packet = {"source_parent_component": "original_region", "image_sha256": "a" * 64,
                  "source_window": [100, 200, 4, 4], "local_image_shape": [4, 4],
                  "source_raster_affine": [0.000002, 0, 108.7, 0, -0.000002, 34.3],
                  "source_raster_crs": "EPSG:4326", "native_resampling": "NONE",
                  "source_observed_date": "2018-11-01", "experiment_year": 2019}
        prompt = {"component_key": "original_component", "group_id": "original_group",
                  "source_geometry_wkb_sha256": "c" * 64,
                  "point_coords_pixel_index_xy": [[1, 1]], "point_labels": [1]}
        receipt = {"group_id": "original_group", "component_key": "original_component",
                   "source_geometry_wkb_sha256": "c" * 64,
                   "image_sha256": "a" * 64, "source_window": packet["source_window"],
                   "selected_mask_index": None, "roof_or_full_instance_verified": False,
                   "physical_id": None, "original_mask_shape": [3, 4, 4], "mask_pixels": [4, 4, 4],
                   "predicted_mask_quality_diagnostic_only": scores.tolist()}
        return packet, prompt, receipt, {"masks": mask, "predicted_mask_quality": scores}

    def test_sam_keeps_three_native_alternatives_without_parent_identity(self):
        packet, prompt, receipt, arrays = self.sam()
        records = adapt_sam_group(packet, prompt, receipt, arrays, "b" * 64)
        self.assertEqual([r["original_id"] for r in records], ["original_group_MASK_" + str(i) for i in range(3)])
        self.assertTrue(all(not r["selected"] and r["physical_id"] is None
                            and r["building_object_id"] is None for r in records))
        self.assertTrue(all(r["source_observed_date"] == "2018-11-01" for r in records))
        self.assertTrue(all(r["candidate_displacement_m"] is None for r in records))

    def test_sam_score_cannot_promote_candidate(self):
        packet, prompt, receipt, arrays = self.sam()
        receipt["selected_mask_index"] = 0
        with self.assertRaises(ValueError):
            adapt_sam_group(packet, prompt, receipt, arrays, "b" * 64)

    def test_sam_original_native_pixels_checked_against_receipt(self):
        packet, prompt, receipt, arrays = self.sam()
        arrays["masks"][0, 0, 0] = 1
        with self.assertRaises(ValueError):
            adapt_sam_group(packet, prompt, receipt, arrays, "b" * 64)


if __name__ == "__main__":
    unittest.main()
