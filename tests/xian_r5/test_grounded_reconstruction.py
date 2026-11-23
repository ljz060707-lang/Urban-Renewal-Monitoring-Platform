"""Interface-level tests; synthetic geometry is not Xi'an accuracy evidence."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
import fiona
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, shape
from shapely.ops import transform

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.alignment.field import DisplacementCell, LocalDisplacementField
from xian_r5.alignment.field_io import paired_source_component_scope_for_support
from xian_r5.reconstruction import (AbsenceEvidence, FootprintEvidence, Observation,
                                    attach_overlapping_external_candidates, reconstruct)
from xian_r5.reconstruction.products import (
    _coarsen_change, _coarsen_ids, _coarsen_quality, write_products,
)
from xian_r5.reconstruction.registration import estimate_tile_registration
from tools.run_xian_r5_grounded_reconstruction import (_registration_support_polygon,
                                                       _registration_source_support,
                                                       _validate_alignment_reference_scopes,
                                                       _actual_date_pairs,
                                                       _date_coherent_mosaic_centers,
                                                       _tiles_grouped_by_actual_date,
                                                       _footprint_candidate_record)


def observed(uid: str, epoch: str, actual_date: str, roof_dx: float, *,
             verified: bool = True, key: str | None = "building-a") -> Observation:
    ground = box(315200, 3799200, 315215, 3799210)
    roof = box(315200 + roof_dx, 3799200, 315215 + roof_dx, 3799210)
    evidence = FootprintEvidence(
        ground, f"survey:{uid}", "SURVEY", verified, verified, 0.5,
        reviewer_id="synthetic-test", reviewed_at="2026-09-01T00:00:00+00:00",
        evidence_path=f"synthetic-evidence:{uid}", evidence_sha256="a" * 64,
        evidence_integrity_verified=True)
    return Observation(uid, epoch, actual_date, None, None, roof, f"roof:{uid}",
                       footprint_evidence=evidence, confirmed_identity_key=key,
                       identity_reviewed=verified)


def source_component_metadata(*actual_dates: str, tile_id: str = "420796:312173"):
    return {
        "source_component_scope": "SINGLE_SOURCE_COMPONENT_PER_CELL",
        "source_component_ids": [{
            "actual_date": actual_date, "tile_id": tile_id,
            "path": f"synthetic/{actual_date}/{tile_id}.jpg",
            "sha256": ("a" if index % 2 == 0 else "b") * 64,
        } for index, actual_date in enumerate(actual_dates)],
        "source_component_coverage_fraction": 1.0,
    }


class GroundedReconstructionTests(unittest.TestCase):
    def test_alignment_reference_validation_separates_base_and_date_pair_fields(self):
        cell = DisplacementCell(
            "cell", box(315100, 3799100, 315200, 3799200), None, None,
            None, "INSUFFICIENT_MATCHES", "SYNTHETIC", "receipt:test")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base_raster = root / "Xian_2019_TrueColor.tif"
            date_specific_base = root / "Xian_2018_TrueColor.tif"
            pair_receipt = root / "mosaic-registration.json"
            base_field = LocalDisplacementField(
                "base-tie", "2018-04-29", "2018-11-01", (cell,),
                reference_source=str(base_raster))
            date_specific_field = LocalDisplacementField(
                "date-specific-base-tie", None, "2018-04-29", (cell,),
                reference_label="Xian_2018 annual raster",
                reference_source=str(date_specific_base),
                reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")
            date_pair_field = LocalDisplacementField(
                "date-pair", "2018-04-29", "2018-11-01", (cell,),
                reference_source=str(pair_receipt))

            _validate_alignment_reference_scopes(
                base_reference_fields=(base_field, date_specific_field),
                date_pair_fields=(date_pair_field,),
                base_raster=base_raster,
                verified_base_raster_sources=(date_specific_base,),
                date_pair_reference_source=pair_receipt)

            wrong_base = LocalDisplacementField(
                "wrong-base", "2018-04-29", "2018-11-01", (cell,),
                reference_source=str(pair_receipt))
            with self.assertRaisesRegex(ValueError, "verified co-gridded"):
                _validate_alignment_reference_scopes(
                    base_reference_fields=(wrong_base,),
                    date_pair_fields=(),
                    base_raster=base_raster,
                    verified_base_raster_sources=(date_specific_base,),
                    date_pair_reference_source=None)
            with self.assertRaisesRegex(ValueError, "validated mosaic diagnostic"):
                _validate_alignment_reference_scopes(
                    base_reference_fields=(),
                    date_pair_fields=(date_pair_field,),
                    base_raster=base_raster,
                    verified_base_raster_sources=(date_specific_base,),
                    date_pair_reference_source=root / "other-mosaic.json")

    def test_registration_support_polygon_scales_with_tile_span(self):
        tile = _registration_support_polygon(420875, 312123, 1)
        mosaic = _registration_support_polygon(420875, 312123, 3)

        self.assertTrue(tile.is_valid)
        self.assertTrue(mosaic.covers(tile.centroid))
        self.assertAlmostEqual(mosaic.area / tile.area, 9.0, delta=0.1)

    def test_mosaic_registration_support_uses_exact_source_tile_union(self):
        tile_x, tile_y = 420875, 312123
        fallback = _registration_support_polygon(tile_x, tile_y, 3)
        dates = ("2018-04-29", "2018-11-01")
        entries_by_date = {}
        for date_index, actual_date in enumerate(dates):
            entries = []
            for row in range(tile_y - 1, tile_y + 2):
                for col in range(tile_x - 1, tile_x + 2):
                    tile = _registration_support_polygon(col, row, 1)
                    entries.append({
                        "tile_id": f"{col}:{row}", "actual_date": actual_date,
                        "path": f"synthetic/{actual_date}/{col}_{row}.jpg",
                        "sha256": chr(ord("a") + date_index) * 64,
                        "support_polygon_epsg32649": mapping(tile),
                    })
            entries_by_date[actual_date] = entries

        support = _registration_source_support(fallback, entries_by_date)
        scope = paired_source_component_scope_for_support(
            support, entries_by_date, dates)

        self.assertTrue(support.is_valid)
        self.assertEqual(scope["source_component_scope"],
                         "MULTIPLE_SOURCE_COMPONENTS_PER_CELL")
        self.assertAlmostEqual(scope["source_component_coverage_fraction"], 1.0, places=12)
        self.assertEqual(len(scope["source_component_ids"]), 18)

    def test_registration_pairs_cover_all_unique_actual_dates(self):
        self.assertEqual(
            _actual_date_pairs(["2019-12-02", "2018-11-01", "2018-04-29",
                                "2018-11-01"]),
            [("2018-04-29", "2018-11-01"),
             ("2018-04-29", "2019-12-02"),
             ("2018-11-01", "2019-12-02")],
        )

    def test_mosaic_candidates_use_shared_geography_not_shared_target_tile_keys(self):
        left_date, right_date = "2018-04-29", "2019-12-02"
        date_index = {}
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                date_index[("2018", 10 + dx, 20 + dy)] = left_date
                date_index[("2020", 10 + dx, 20 + dy)] = right_date
                date_index[("2018", 11 + dx, 20 + dy)] = left_date
                date_index[("2020", 11 + dx, 20 + dy)] = right_date
        # One absent exact-date source makes the adjacent window unusable.
        date_index.pop(("2020", 12, 21))

        centers = {(10, 20), (11, 20)}
        candidates = _date_coherent_mosaic_centers(
            centers, "2018", "2020", left_date, right_date, date_index)

        self.assertEqual(candidates, [(10, 20)])

    def test_shifted_roofs_same_ground_produce_only_stable(self):
        observations = [observed("a", "2018", "2018-04-29", 0),
                        observed("b", "2019", "2018-11-01", 4),
                        observed("c", "2020", "2019-12-02", -3)]
        result = reconstruct(observations)
        self.assertEqual([t["state"] for t in result.transitions], ["STABLE", "STABLE"])
        self.assertEqual(len({o.physical_id for o in result.observations}), 1)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "products"
            write_products(output, result)
            with rasterio.open(output / "2018-04-29_ground_footprint.tif") as first, rasterio.open(output / "2018-11-01_ground_footprint.tif") as second:
                self.assertEqual(first.crs, second.crs)
                self.assertEqual(first.transform, second.transform)
                np.testing.assert_array_equal(first.read(1), second.read(1))
            with rasterio.open(output / "2018-04-29_to_2018-11-01_change.tif") as change:
                array = change.read(1)
                self.assertGreater(int((array == 1).sum()), 0)
                self.assertEqual(int((array == 2).sum()), 0)
                self.assertEqual(int((array == 3).sum()), 0)
                self.assertEqual(int((array == 4).sum()), 0)
                self.assertEqual(int((array == 255).sum()), 0)
            with self.assertRaises(FileExistsError):
                write_products(output, result)

    def test_mixed_nominal_epoch_splits_by_actual_date_and_collapses_duplicate_acquisition(self):
        observations = [
            observed("early", "2018", "2018-04-29", 0),
            observed("middle", "2018", "2018-11-01", 4),
            observed("middle-copy", "2019", "2018-11-01", 4),
            observed("late", "2019", "2019-12-02", -3),
        ]

        result = reconstruct(observations)

        self.assertEqual(result.actual_date_order,
                         ["2018-04-29", "2018-11-01", "2019-12-02"])
        self.assertEqual([row["state"] for row in result.transitions],
                         ["STABLE", "STABLE"])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "products"
            manifest = write_products(output, result)
            self.assertEqual(manifest["actual_dates_by_epoch"]["2018"],
                             ["2018-04-29", "2018-11-01"])
            self.assertEqual(manifest["epochs"], result.actual_date_order)
            self.assertTrue((output / "2018-04-29_ground_footprint.tif").is_file())
            self.assertTrue((output / "2018-11-01_ground_footprint.tif").is_file())
            self.assertTrue((output / "2019-12-02_ground_footprint.tif").is_file())
            self.assertTrue((output / "2018-04-29_to_2018-11-01_change.tif").is_file())
            self.assertTrue((output / "2018-11-01_to_2019-12-02_change.tif").is_file())
            self.assertFalse((output / "2018_ground_footprint.tif").exists())

    def test_unknown_dates_are_exported_as_unordered_unresolved_groups(self):
        result = reconstruct([
            observed("unknown", "2018", None, 0),
            observed("known", "2019", "2018-11-01", 2),
        ])

        self.assertEqual(result.actual_date_order, ["2018-11-01"])
        self.assertEqual(result.transitions, [])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "products"
            manifest = write_products(output, result)
            unknown_group = "DATE_UNKNOWN_2018"
            self.assertEqual(manifest["epochs"], ["2018-11-01", unknown_group])
            self.assertEqual(manifest["output_group_metadata"][unknown_group]["actual_date"], None)
            with rasterio.open(output / f"{unknown_group}_ground_footprint.tif") as raster:
                tags = raster.tags()
                self.assertEqual(tags["ACTUAL_DATE"], "UNKNOWN")
                self.assertEqual(tags["ACTUAL_DATE_STATUS"], "ACTUAL_DATE_UNRESOLVED")
                self.assertEqual(tags["EPOCH_LABELS_JSON"], '["2018"]')
            self.assertFalse(any("_to_" in path.name and "_change.tif" in path.name
                                 for path in output.glob("*.tif")))

    def test_unverified_translation_is_not_ground(self):
        result = reconstruct([observed("a", "2018", "2018-04-29", 3, verified=False)])
        self.assertIsNone(result.observations[0].ground_footprint)
        self.assertIsNone(result.observations[0].physical_id)
        self.assertEqual(result.observations[0].footprint_status, "GROUND_CANDIDATE_UNVERIFIED")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "products"
            write_products(output, result)
            with rasterio.open(output / "2018-04-29_ground_footprint.tif") as raster:
                values = set(np.unique(raster.read(1)).tolist())
                self.assertEqual(values, {0, 2})
                self.assertEqual(raster.nodata, 255)

    def test_candidate_track_id_rasters_keep_source_labels_unpromoted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchor = root / "base10m.tif"
            with rasterio.open(anchor, "w", driver="GTiff", width=8, height=8,
                               count=1, dtype="uint8", crs="EPSG:32649",
                               transform=from_origin(0, 80, 10, 10)) as sink:
                sink.write(np.zeros((1, 8, 8), dtype="uint8"))

            observations = [
                Observation("a-2018", "2018", "2018-04-29", None, None,
                            box(10, 60, 20, 70), "roof:a-2018",
                            candidate_identity="track-a"),
                Observation("b-2018", "2018", "2018-04-29", None, None,
                            box(16, 60, 26, 70), "roof:b-2018",
                            candidate_identity="track-b"),
                Observation("duplicate-1", "2018", "2018-04-29", None, None,
                            box(30, 60, 34, 64), "roof:duplicate-1",
                            candidate_identity="track-duplicate"),
                Observation("duplicate-2", "2018", "2018-04-29", None, None,
                            box(36, 60, 40, 64), "roof:duplicate-2",
                            candidate_identity="track-duplicate"),
                Observation("missing-label", "2018", "2018-04-29", None, None,
                            box(44, 60, 48, 64), "roof:missing-label"),
                Observation("a-2019", "2019", "2018-11-01", None, None,
                            box(11, 60, 21, 70), "roof:a-2019",
                            candidate_identity="track-a"),
            ]
            output = root / "products"
            manifest = write_products(
                output, reconstruct(observations), anchor_raster=anchor)

            self.assertEqual(manifest["schema_version"],
                             "xian_r5.grounded_products.v17")
            self.assertFalse(manifest["candidate_track_ids_promoted_to_physical_ids"])
            codebook_path = output / manifest["candidate_track_id_codebook"]["path"]
            self.assertEqual(
                hashlib.sha256(codebook_path.read_bytes()).hexdigest(),
                manifest["candidate_track_id_codebook"]["sha256"])
            codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
            codes = {label: int(code) for code, label in
                     codebook["code_to_source_candidate_identity"].items()}
            self.assertEqual(set(codes), {"track-a", "track-b", "track-duplicate"})
            self.assertEqual(manifest["candidate_track_id_codebook"][
                "candidate_identity_label_count"], len(codes))
            products = {item["output_group"]: item for item in
                        manifest["candidate_track_id_products"]}
            unknown = np.uint32(4294967295)

            first = products["2018-04-29"]
            with rasterio.open(output / first["candidate_tif"]) as track_ds, \
                    rasterio.open(output / first["quality_tif"]) as quality_ds:
                track = track_ds.read(1)
                quality = quality_ds.read(1)
                unique_row, unique_col = track_ds.index(13.5, 65.5)
                overlap_row, overlap_col = track_ds.index(18.5, 65.5)
                duplicate_row, duplicate_col = track_ds.index(32.5, 62.5)
                missing_row, missing_col = track_ds.index(46.5, 62.5)
                self.assertEqual(track[unique_row, unique_col], codes["track-a"])
                self.assertEqual(quality[unique_row, unique_col], 1)
                self.assertEqual(track[overlap_row, overlap_col], unknown)
                self.assertEqual(quality[overlap_row, overlap_col], 4)
                self.assertEqual(track[duplicate_row, duplicate_col], unknown)
                self.assertEqual(quality[duplicate_row, duplicate_col], 3)
                self.assertEqual(track[missing_row, missing_col], unknown)
                self.assertEqual(quality[missing_row, missing_col], 2)
            self.assertEqual(first["repeated_source_track_labels"],
                             ["track-duplicate"])

            second = products["2018-11-01"]
            with rasterio.open(output / second["candidate_tif"]) as track_ds, \
                    rasterio.open(output / second["quality_tif"]) as quality_ds:
                row, col = track_ds.index(13.5, 65.5)
                self.assertEqual(track_ds.read(1)[row, col], codes["track-a"])
                self.assertEqual(quality_ds.read(1)[row, col], 1)

            for group_id in products:
                with rasterio.open(output / f"{group_id}_physical_id.tif") as ids:
                    self.assertTrue(set(np.unique(ids.read(1)).tolist())
                                    <= {0, int(unknown)})
                with rasterio.open(output / f"{group_id}_candidate_track_id_ROOF_PROXY_BASE10M.tif") as base:
                    self.assertEqual(base.tags()["GRID_LEVEL"], "REFERENCE_10M")
                self.assertFalse(products[group_id]["formal_physical_id_product"])
                self.assertFalse(products[group_id]["change_use"])

            with rasterio.open(
                    output / "2018-04-29_to_2018-11-01_change.tif") as change:
                self.assertFalse(np.isin(change.read(1), [1, 2, 3]).any())

    def test_external_polygon_is_preserved_only_as_unverified_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "external.gpkg"
            roof = box(315200, 3799200, 315215, 3799210)
            external_utm = box(315203, 3799200, 315218, 3799210)
            to_wgs = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
            external_wgs = transform(to_wgs.transform, external_utm)
            anchor = root / "base10m.tif"
            with rasterio.open(anchor, "w", driver="GTiff", width=100, height=100,
                               count=1, dtype="uint8", crs="EPSG:32649",
                               transform=from_origin(315000, 3799400, 10, 10)) as sink:
                sink.write(np.zeros((1, 100, 100), dtype="uint8"))
            with fiona.open(source, "w", driver="GPKG", layer="external",
                            schema={"geometry": "Polygon", "properties": {"Floor": "int"}},
                            crs="EPSG:4326") as sink:
                sink.write({"geometry": mapping(external_wgs), "properties": {"Floor": 6}})
            observation = Observation("candidate", "2018", "2018-04-29", None, None,
                                      roof, "roof:candidate")
            report = attach_overlapping_external_candidates(
                [observation], source, layer="external", source_epoch_label="2019",
                source_geometry_semantics="ROOF_BOUNDARY_FROM_IMAGE")
            self.assertEqual(report["source_capture_date"], None)
            self.assertEqual(report["source_epoch_label"], "2019")
            self.assertEqual(report["source_epoch_label_provenance"],
                             "USER_REPORTED_EPOCH_LABEL_NOT_EXACT_ACQUISITION_DATE")
            self.assertEqual(report["source_geometry_semantics"],
                             "ROOF_BOUNDARY_FROM_IMAGE")
            self.assertEqual(report["source_geometry_semantics_provenance"],
                             "USER_REPORTED_SEMANTICS_NOT_INDEPENDENTLY_VERIFIED")
            self.assertIsNone(report["source_accuracy_m"])
            self.assertFalse(report["ground_truth"])
            self.assertEqual(report["source_feature_count"], 1)
            self.assertEqual(report["source_crs"], "EPSG:4326")
            self.assertEqual(report["source_schema"]["geometry"], "Polygon")
            self.assertEqual(report["source_schema"]["properties"], {"Floor": "int"})
            self.assertEqual(report["observations_with_candidates"], 1)
            self.assertEqual(report["source_hash_recheck"], "PASS")
            self.assertEqual(report["source_component_sha256"], report["source_component_sha256_after"])
            self.assertEqual(len(report["source_component_sha256"]["external.gpkg"]), 64)
            self.assertEqual(report["schema_version"], "xian_r5.external_footprint_candidates.v5")
            self.assertEqual(len(observation.footprint_candidates), 1)
            candidate = observation.footprint_candidates[0]
            self.assertEqual(candidate.source_epoch_label, "2019")
            self.assertEqual(candidate.source_geometry_semantics,
                             "ROOF_BOUNDARY_FROM_IMAGE")
            self.assertIsNone(candidate.source_capture_date)
            self.assertEqual(candidate.source_date_status, "NOT_ASSOCIATED")
            self.assertEqual(candidate.source_date_candidates, ())
            self.assertEqual(candidate.source_tile_actual_dates, ())
            self.assertEqual(candidate.source_scene_linkage_status, "UNVERIFIED")
            self.assertAlmostEqual(candidate.intersection_area_m2, 120.0, delta=0.01)
            self.assertAlmostEqual(candidate.roof_overlap_fraction, 0.8, delta=1e-5)
            self.assertAlmostEqual(candidate.candidate_overlap_fraction, 0.8, delta=1e-5)
            self.assertAlmostEqual(candidate.candidate_centroid_dx_from_roof_m, 3.0, delta=0.01)
            self.assertAlmostEqual(candidate.candidate_centroid_dy_from_roof_m, 0.0, delta=0.01)
            self.assertAlmostEqual(candidate.centroid_distance_m, 3.0, delta=0.01)
            self.assertAlmostEqual(candidate.candidate_polygon_distance_m, 0.0, delta=0.01)
            self.assertIsNone(candidate.candidate_search_radius_m)
            self.assertAlmostEqual(candidate.centroid_aligned_roof_iou, 1.0, delta=1e-5)
            self.assertAlmostEqual(candidate.centroid_aligned_iou_gain, 1.0 - 2.0 / 3.0, delta=1e-5)
            result = reconstruct([observation])
            self.assertIsNone(observation.ground_footprint)
            self.assertIsNone(observation.physical_id)
            self.assertEqual(observation.footprint_status, "GROUND_CANDIDATE_UNVERIFIED")
            output = root / "products"
            manifest = write_products(output, result, anchor_raster=anchor)
            self.assertEqual(manifest["schema_version"], "xian_r5.grounded_products.v17")
            self.assertEqual(manifest["external_candidate_source_epoch_labels"], ["2019"])
            self.assertEqual(manifest["external_candidate_source_geometry_semantics"],
                             ["ROOF_BOUNDARY_FROM_IMAGE"])
            self.assertEqual(manifest["external_candidate_source_date_status_counts"],
                             {"NOT_ASSOCIATED": 1})
            self.assertEqual(manifest["external_candidate_source_date_candidates"], [])
            self.assertFalse(manifest["external_candidate_source_scene_linkage_verified"])
            with fiona.open(output / "building_lineage.gpkg", layer="footprint_candidates") as layer:
                rows = list(layer)
                self.assertEqual(len(rows), 1)
                props = dict(rows[0]["properties"])
                self.assertEqual(props["candidate_id"], "1")
                self.assertEqual(props["source_date"], None)
                self.assertEqual(props["source_date_status"], "NOT_ASSOCIATED")
                self.assertEqual(json.loads(props["source_date_candidates_json"]), [])
                self.assertEqual(json.loads(props["source_tile_actual_dates_json"]), [])
                self.assertEqual(props["source_scene_linkage_status"], "UNVERIFIED")
                self.assertEqual(props["source_epoch_label"], "2019")
                self.assertEqual(props["source_geometry_semantics"],
                                 "ROOF_BOUNDARY_FROM_IMAGE")
                self.assertEqual(props["source_geometry_semantics_provenance"],
                                 "USER_REPORTED_SEMANTICS_NOT_INDEPENDENTLY_VERIFIED")
                self.assertEqual(props["source_accuracy_m"], None)
                self.assertEqual(json.loads(props["source_properties_json"]), {"Floor": 6})
                self.assertAlmostEqual(props["intersection_area_m2"], 120.0, delta=0.01)
                self.assertAlmostEqual(props["roof_overlap_fraction"], 0.8, delta=1e-5)
                self.assertAlmostEqual(props["candidate_overlap_fraction"], 0.8, delta=1e-5)
                self.assertAlmostEqual(props["candidate_centroid_dx_from_roof_m"], 3.0, delta=0.01)
                self.assertAlmostEqual(props["candidate_centroid_dy_from_roof_m"], 0.0, delta=0.01)
                self.assertAlmostEqual(props["centroid_distance_m"], 3.0, delta=0.01)
                self.assertAlmostEqual(props["candidate_polygon_distance_m"], 0.0, delta=0.01)
                self.assertIsNone(props["candidate_search_radius_m"])
                self.assertAlmostEqual(props["centroid_aligned_roof_iou"], 1.0, delta=1e-5)
                self.assertAlmostEqual(props["centroid_aligned_iou_gain"], 1.0 - 2.0 / 3.0, delta=1e-5)
                self.assertEqual(props["status"], "UNVERIFIED_CANDIDATE")
                candidate_feature_geometry = rows[0]["geometry"]
            candidate_products = manifest["ground_footprint_candidate_products"]
            self.assertEqual(len(candidate_products), 1)
            candidate_product = candidate_products[0]
            self.assertEqual(candidate_product["candidate_feature_count"], 1)
            self.assertEqual(candidate_product["candidate_observation_count"], 1)
            self.assertFalse(candidate_product["formal_ground_footprint_product"])
            self.assertFalse(candidate_product["position_accuracy_claimed"])
            with rasterio.open(output / candidate_product["candidate_tif"]) as raster:
                candidate_mask = raster.read(1)
                expected_mask = rasterize(
                    [(candidate_feature_geometry, 1)], out_shape=candidate_mask.shape,
                    transform=raster.transform, fill=0, dtype="uint8")
                np.testing.assert_array_equal(candidate_mask, expected_mask)
                self.assertIn(1, np.unique(candidate_mask))
                self.assertEqual(raster.tags()["PRODUCT_KIND"],
                                 "UNVERIFIED_GROUND_FOOTPRINT_CANDIDATE")
                self.assertEqual(raster.tags()["FORMAL_GROUND_FOOTPRINT_PRODUCT"], "NO")
            with rasterio.open(output / candidate_product["candidate_base10m_tif"]) as raster:
                self.assertIn(1, np.unique(raster.read(1)))
                self.assertEqual(raster.tags()["GRID_LEVEL"], "REFERENCE_10M")
            with rasterio.open(output / "2018-04-29_ground_footprint.tif") as raster:
                self.assertIn(2, np.unique(raster.read(1)))
                self.assertNotIn(1, np.unique(raster.read(1)))
                self.assertEqual(raster.nodata, 255)
            with rasterio.open(output / "2018-04-29_spatial_quality.tif") as raster:
                quality = raster.read(1)
                roof_only_row, roof_only_col = raster.index(315201, 3799205)
                candidate_row, candidate_col = raster.index(315217, 3799205)
                self.assertEqual(int(quality[roof_only_row, roof_only_col]), 2)
                self.assertEqual(int(quality[candidate_row, candidate_col]), 3)
            with rasterio.open(output / "2018-04-29_physical_id.tif") as raster:
                self.assertIn(np.uint32(4294967295), np.unique(raster.read(1)))
                self.assertIsNone(raster.nodata)

            changed_hashes = [{"external.gpkg": "before"}, {"external.gpkg": "after"}]
            with patch("xian_r5.reconstruction.external_candidates._source_component_hashes",
                       side_effect=changed_hashes):
                with self.assertRaisesRegex(RuntimeError, "source changed while being read"):
                    attach_overlapping_external_candidates([
                        Observation("changed", "2018", "2018-04-29", None, None,
                                    roof, "roof:changed")], source, layer="external")

    def test_image_grid_outline_comparison_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchor = root / "base10m.tif"
            with rasterio.open(anchor, "w", driver="GTiff", width=100, height=100,
                               count=1, dtype="uint8", crs="EPSG:32649",
                               transform=from_origin(315000, 3799400, 10, 10)) as sink:
                sink.write(np.zeros((1, 100, 100), dtype="uint8"))
            roof = box(315200, 3799200, 315215, 3799210)
            outline = box(315203, 3799198, 315218, 3799208)
            source = root / "external.gpkg"
            with fiona.open(source, "w", driver="GPKG", layer="external",
                            schema={"geometry": "Polygon", "properties": {"Floor": "int"}},
                            crs="EPSG:32649") as sink:
                sink.write({"geometry": mapping(outline), "properties": {"Floor": 6}})
            observation = Observation(
                "outline-diagnostic", "2018", "2018-04-29", None, None,
                roof, "roof:outline-diagnostic")
            attach_overlapping_external_candidates([observation], source, layer="external")
            support = box(315150, 3799150, 315250, 3799250)
            field = LocalDisplacementField(
                "base-field", None, "2018-04-29",
                (DisplacementCell(
                    "base-cell", support, 3.0, -2.0, None,
                    "DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED", "SYNTHETIC",
                    "field:base-cell",
                    metadata=source_component_metadata("2018-04-29")),),
                reference_label="base10m", reference_source=str(anchor.resolve()),
                reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")

            output = root / "products"
            manifest = write_products(
                output, reconstruct([observation]), anchor_raster=anchor,
                alignment_fields=[field])

            self.assertEqual(manifest["schema_version"], "xian_r5.grounded_products.v17")
            self.assertFalse(manifest["formal_ground_products_use_candidate_roof_geometry"])
            diagnostic = manifest["image_grid_roof_outline_diagnostics"]
            self.assertEqual(diagnostic["row_count"], 1)
            self.assertFalse(diagnostic["formal_ground_use"])
            self.assertFalse(diagnostic["formal_change_use"])
            with fiona.open(output / "building_lineage.gpkg",
                            layer="image_grid_roof_outline_diagnostics") as layer:
                rows = list(layer)
            self.assertEqual(len(rows), 1)
            props = dict(rows[0]["properties"])
            self.assertEqual(props["ground_status"], "NOT_VERIFIED")
            self.assertEqual(props["formal_ground_use"], 0)
            self.assertEqual(props["formal_change_use"], 0)
            self.assertEqual(props["candidate_shift"], 1)
            self.assertEqual(props["candidate_dx_m"], 3.0)
            self.assertEqual(props["candidate_dy_m"], -2.0)
            self.assertEqual(props["roof_source_ref"], observation.source_ref)
            self.assertIsNone(props["candidate_source_date"])
            self.assertIsNone(props["candidate_source_accuracy_m"])
            self.assertAlmostEqual(props["image_grid_iou"], 1.0, places=6)
            self.assertGreater(props["image_grid_iou_gain"], 0.45)
            self.assertTrue(shape(rows[0]["geometry"]).equals(outline))
            with rasterio.open(output / "2018-04-29_ground_footprint.tif") as raster:
                self.assertNotIn(1, np.unique(raster.read(1)))

    def test_external_bbox_scan_continues_when_driver_cannot_report_layer_extent(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "external.gpkg"
            source.write_bytes(b"stable mocked vector source")
            roof = box(315200, 3799200, 315215, 3799210)
            to_wgs = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
            candidate_wgs = transform(to_wgs.transform, roof)
            feature = {"id": "7", "geometry": mapping(candidate_wgs),
                       "properties": {"Floor": 3}}

            class SourceWithoutExtent:
                crs_wkt = "EPSG:4326"
                crs = {}
                schema = {"geometry": "Polygon", "properties": {"Floor": "int"}}

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def __len__(self):
                    return 1

                @property
                def bounds(self):
                    raise fiona.errors.DriverError("Unable to calculate layer extent")

                def filter(self, *, bbox):
                    self.query_bbox = bbox
                    return iter([feature])

            external_source = SourceWithoutExtent()
            observation = Observation("extent-error", "2018", "2018-04-29",
                                      None, None, roof, "roof:extent-error")
            with patch("xian_r5.reconstruction.external_candidates.fiona.open",
                       return_value=external_source):
                report = attach_overlapping_external_candidates([observation], source)

            self.assertEqual(report["source_bounds"], None)
            self.assertEqual(report["source_bounds_status"], "UNAVAILABLE_DRIVER_ERROR")
            self.assertGreater(len(external_source.query_bbox), 0)
            self.assertEqual(report["features_examined_in_local_bbox"], 1)
            self.assertEqual(report["observations_with_candidates"], 1)
            self.assertEqual(len(observation.footprint_candidates), 1)

    def test_nearby_nonoverlapping_external_polygon_is_kept_as_candidate_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "nearby.gpkg"
            roof = box(315200, 3799200, 315215, 3799210)
            nearby_utm = box(315220, 3799200, 315235, 3799210)
            outside_utm = box(315241, 3799200, 315256, 3799210)
            to_wgs = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
            features = [transform(to_wgs.transform, geom)
                        for geom in (nearby_utm, outside_utm)]
            with fiona.open(source, "w", driver="GPKG", layer="external",
                            schema={"geometry": "Polygon", "properties": {"Floor": "int"}},
                            crs="EPSG:4326") as sink:
                for index, geometry in enumerate(features, start=1):
                    sink.write({"geometry": mapping(geometry), "properties": {"Floor": index}})
            observation = Observation("nearby", "2018", "2018-04-29", None, None,
                                      roof, "roof:nearby")
            report = attach_overlapping_external_candidates(
                [observation], source, layer="external", max_candidate_distance_m=20.0)
            self.assertEqual(report["schema_version"], "xian_r5.external_footprint_candidates.v5")
            self.assertEqual(report["association_policy"],
                             "POSITIVE_AREA_OVERLAP_OR_NEARBY_WITHIN_DISTANCE")
            self.assertEqual(len(observation.footprint_candidates), 1)
            candidate = observation.footprint_candidates[0]
            self.assertEqual(candidate.association_method, "NEARBY_WITHIN_SEARCH_RADIUS")
            self.assertAlmostEqual(candidate.match_iou, 0.0, delta=1e-7)
            self.assertAlmostEqual(candidate.candidate_polygon_distance_m, 5.0, delta=0.02)
            self.assertAlmostEqual(candidate.candidate_search_radius_m, 20.0)
            self.assertAlmostEqual(candidate.candidate_centroid_dx_from_roof_m, 20.0, delta=0.02)
            self.assertAlmostEqual(candidate.centroid_aligned_roof_iou, 1.0, delta=1e-5)
            self.assertAlmostEqual(candidate.centroid_aligned_iou_gain, 1.0, delta=1e-5)
            receipt_candidate = _footprint_candidate_record(candidate)
            self.assertEqual(receipt_candidate["association"], candidate.association_method)
            self.assertEqual(receipt_candidate["candidate_id"], candidate.candidate_id)
            for name in ("candidate_polygon_distance_m", "candidate_search_radius_m",
                         "centroid_aligned_roof_iou", "centroid_aligned_iou_gain",
                         "intersection_area_m2", "roof_overlap_fraction",
                         "candidate_overlap_fraction", "candidate_centroid_dx_from_roof_m",
                         "candidate_centroid_dy_from_roof_m", "centroid_distance_m"):
                self.assertEqual(receipt_candidate[name], getattr(candidate, name))
            self.assertEqual(receipt_candidate["status"], "UNVERIFIED_CANDIDATE")

            result = reconstruct([observation])
            self.assertIsNone(observation.ground_footprint)
            self.assertEqual(observation.footprint_status, "GROUND_CANDIDATE_UNVERIFIED")
            output = root / "products"
            manifest = write_products(output, result)
            self.assertEqual(manifest["confirmed_ground_geometry_observations"], 0)
            with fiona.open(output / "building_lineage.gpkg", layer="footprint_candidates") as layer:
                rows = list(layer)
            self.assertEqual(len(rows), 1)
            properties = dict(rows[0]["properties"])
            self.assertEqual(properties["association"], "NEARBY_WITHIN_SEARCH_RADIUS")
            self.assertEqual(properties["status"], "UNVERIFIED_CANDIDATE")
            self.assertAlmostEqual(properties["candidate_polygon_distance_m"], 5.0, delta=0.02)
            self.assertAlmostEqual(properties["centroid_aligned_roof_iou"], 1.0, delta=1e-5)

    def test_absence_requires_review_and_date_order(self):
        first = observed("a", "2018", "2018-04-29", 0)
        absence = AbsenceEvidence(
            "building-a", "2019", "2018-11-01", "image:2019", True, True,
            reviewer_id="synthetic-test", reviewed_at="2026-09-01T00:00:00+00:00",
            evidence_path="synthetic-absence-review", evidence_sha256="b" * 64,
            evidence_integrity_verified=True)
        result = reconstruct([first, observed("other", "2019", "2018-11-01", 25, verified=False, key=None)], absence_evidence=[absence])
        self.assertEqual(result.transitions[0]["state"], "DEMOLITION")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "products"
            write_products(output, result)
            with rasterio.open(output / "2018-04-29_to_2018-11-01_demolition.tif") as raster:
                self.assertGreater(int((raster.read(1) == 1).sum()), 0)
        duplicate = reconstruct([observed("a", "2018", "2018-04-29", 0), observed("b", "2019", "2018-04-29", 1)])
        self.assertEqual(duplicate.actual_date_order, ["2018-04-29"])
        self.assertEqual(duplicate.transitions, [])
        unreviewed = AbsenceEvidence(
            "building-a", "2019", "2018-11-01", "image:2019", True, False,
            reviewer_id="synthetic-test", reviewed_at="2026-09-01T00:00:00+00:00",
            evidence_path="synthetic-absence-review", evidence_sha256="b" * 64,
            evidence_integrity_verified=True)
        unresolved = reconstruct([first, observed("other2", "2019", "2018-11-01", 25, verified=False, key=None)], absence_evidence=[unreviewed])
        self.assertEqual(unresolved.transitions[0]["state"], "UNRESOLVED")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "products"
            write_products(output, unresolved)
            with rasterio.open(output / "2018-04-29_to_2018-11-01_change.tif") as raster:
                self.assertIn(4, np.unique(raster.read(1)))
                self.assertEqual(raster.nodata, 255)

    def test_matching_and_products_follow_actual_date_not_nominal_label(self):
        later_nominal_but_earlier_actual = observed("early", "2019", "2018-04-29", 0)
        earlier_nominal_but_later_actual = observed("late", "2018", "2019-05-10", 5)

        result = reconstruct([earlier_nominal_but_later_actual, later_nominal_but_earlier_actual])

        self.assertEqual(result.epoch_order, ["2019", "2018"])
        self.assertEqual(result.actual_date_order, ["2018-04-29", "2019-05-10"])
        self.assertEqual(result.transitions[0]["from_date"], "2018-04-29")
        self.assertEqual(result.transitions[0]["to_date"], "2019-05-10")
        self.assertEqual(result.transitions[0]["state"], "STABLE")
        self.assertEqual(result.candidate_edges[0]["from_uid"], "early")
        self.assertEqual(result.candidate_edges[0]["to_uid"], "late")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "products"
            manifest = write_products(output, result)
            self.assertEqual(manifest["epochs"], ["2018-04-29", "2019-05-10"])
            self.assertEqual(manifest["actual_dates_by_epoch"], {
                "2019": ["2018-04-29"], "2018": ["2019-05-10"]})
            self.assertTrue((output / "2018-04-29_to_2019-05-10_change.tif").is_file())

    def test_candidate_identity_graph_keeps_one_to_many_links_separate_from_ids(self):
        previous = observed("prior", "2018", "2018-04-29", 0,
                            verified=False, key=None)
        current_a = observed("next-a", "2019", "2018-11-01", 0,
                             verified=False, key=None)
        current_b = observed("next-b", "2019", "2018-11-01", 0,
                             verified=False, key=None)

        result = reconstruct([previous, current_a, current_b])

        self.assertEqual(len(result.candidate_edges), 2)
        self.assertEqual({edge["relation"] for edge in result.candidate_edges}, {"1:N"})
        self.assertEqual({edge["status"] for edge in result.candidate_edges}, {"CANDIDATE_ONLY"})
        self.assertTrue(all(edge["from_uid"] == "prior" for edge in result.candidate_edges))
        self.assertTrue(all(observation.physical_id is None for observation in result.observations))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "products"
            manifest = write_products(output, result)
            self.assertEqual(manifest["candidate_identity_edge_count"], 2)
            self.assertFalse(manifest["candidate_edges_promoted_to_physical_identity"])
            radius_semantics = manifest["candidate_identity_edge_search_radius_semantics"]
            self.assertFalse(radius_semantics["unknown_search_buffer_is_positional_accuracy"])
            self.assertEqual(
                radius_semantics["SHARED_UNVERIFIED_EXTERNAL_OUTLINE_REFERENCE"],
                "NULL_NOT_APPLICABLE")
            self.assertTrue(all(edge["search_radius_m"] is not None
                                for edge in result.candidate_edges))
            with fiona.open(output / "building_lineage.gpkg",
                            layer="candidate_identity_edges") as layer:
                edges = list(layer)
                self.assertEqual(len(edges), 2)
                self.assertNotIn("physical_id", layer.schema["properties"])
                self.assertEqual({feature["properties"]["relation"] for feature in edges}, {"1:N"})
                self.assertEqual({feature["properties"]["status"] for feature in edges},
                                 {"CANDIDATE_ONLY"})
                self.assertEqual(
                    {feature["properties"]["search_radius_m"] for feature in edges},
                    {edge["search_radius_m"] for edge in result.candidate_edges})
            with fiona.open(output / "building_lineage.gpkg",
                            layer="identity_graph_nodes") as layer:
                nodes = list(layer)
                self.assertEqual(len(nodes), 3)
                self.assertTrue(all(node["properties"]["physical_id"] is None for node in nodes))

    def test_shared_unverified_outline_reference_adds_candidate_split_edges_only(self):
        previous = observed("prior", "2018", "2018-04-29", 0,
                            verified=False, key=None)
        current_a = observed("next-a", "2019", "2018-11-01", 25,
                             verified=False, key=None)
        current_b = observed("next-b", "2019", "2018-11-01", 26,
                             verified=False, key=None)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "external.gpkg"
            with fiona.open(source, "w", driver="GPKG", layer="external",
                            schema={"geometry": "Polygon", "properties": {"Id": "int"}},
                            crs="EPSG:32649") as sink:
                sink.write({"geometry": mapping(box(315190, 3799190, 315260, 3799230)),
                            "properties": {"Id": 77}})

            report = attach_overlapping_external_candidates(
                [previous, current_a, current_b], source, layer="external",
                source_epoch_label="2019",
                source_geometry_semantics="ROOF_BOUNDARY_FROM_IMAGE")
            result = reconstruct([previous, current_a, current_b])
            outline_edges = [
                edge for edge in result.candidate_edges
                if edge.get("edge_basis") == "SHARED_UNVERIFIED_EXTERNAL_OUTLINE_REFERENCE"]

            self.assertFalse(report["ground_truth"])
            self.assertEqual(report["source_epoch_label"], "2019")
            self.assertEqual(len(outline_edges), 2)
            self.assertEqual({edge["relation"] for edge in outline_edges}, {"1:N"})
            self.assertTrue(all(edge["status"] == "CANDIDATE_ONLY" for edge in outline_edges))
            self.assertTrue(all(edge["affinity"] is None for edge in outline_edges))
            self.assertTrue(all(edge["candidate_count"] == 1 for edge in outline_edges))
            self.assertTrue(all(edge["candidate_references"][0]["source_capture_date"] is None
                                for edge in outline_edges))
            self.assertTrue(all(edge["candidate_references"][0]["source_epoch_label"] == "2019"
                                for edge in outline_edges))
            self.assertTrue(all(
                edge["candidate_references"][0]["source_geometry_semantics"]
                == "ROOF_BOUNDARY_FROM_IMAGE" for edge in outline_edges))
            self.assertTrue(all(edge["candidate_references"][0]["source_accuracy_m"] is None
                                for edge in outline_edges))
            self.assertTrue(all(item.physical_id is None for item in result.observations))
            self.assertTrue(all(item["presence"] == "U" for item in result.presence))

            output = root / "products"
            manifest = write_products(output, result)
            with fiona.open(output / "building_lineage.gpkg",
                            layer="candidate_identity_edges") as layer:
                props = list(layer)
                self.assertIn("candidate_reference_json", layer.schema["properties"])
                self.assertNotIn("physical_id", layer.schema["properties"])
            written_outline_edges = [
                item for item in props
                if item["properties"]["edge_basis"] ==
                "SHARED_UNVERIFIED_EXTERNAL_OUTLINE_REFERENCE"]
            self.assertEqual(len(written_outline_edges), 2)
            self.assertTrue(all(
                json.loads(item["properties"]["candidate_reference_json"])[0][
                    "verification_status"] == "UNVERIFIED_CANDIDATE"
                for item in written_outline_edges))
            self.assertFalse(manifest["candidate_edges_promoted_to_physical_identity"])
            self.assertEqual(
                manifest["candidate_identity_edge_basis_counts"][
                    "SHARED_UNVERIFIED_EXTERNAL_OUTLINE_REFERENCE"], 2)

    def test_shared_outline_reference_bridges_a_missed_intermediate_observation(self):
        previous = observed("prior", "2018", "2018-04-29", 0,
                            verified=False, key=None)
        unrelated_middle = observed("middle-other", "2019", "2018-11-01", 300,
                                    verified=False, key=None)
        current = observed("later", "2020", "2019-12-02", 25,
                           verified=False, key=None)

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "external.gpkg"
            with fiona.open(source, "w", driver="GPKG", layer="external",
                            schema={"geometry": "Polygon", "properties": {"Id": "int"}},
                            crs="EPSG:32649") as sink:
                sink.write({"geometry": mapping(box(315190, 3799190, 315260, 3799230)),
                            "properties": {"Id": 77}})

            report = attach_overlapping_external_candidates(
                [previous, unrelated_middle, current], source, layer="external")
            result = reconstruct([previous, unrelated_middle, current])
            bridges = [
                edge for edge in result.candidate_edges
                if edge.get("edge_basis") ==
                "SHARED_UNVERIFIED_EXTERNAL_OUTLINE_REFERENCE"
                and edge["from_uid"] == "prior" and edge["to_uid"] == "later"]

            self.assertFalse(report["ground_truth"])
            self.assertEqual(len(bridges), 1)
            self.assertEqual(bridges[0]["from_date"], "2018-04-29")
            self.assertEqual(bridges[0]["to_date"], "2019-12-02")
            self.assertTrue(bridges[0]["temporal_order_valid"])
            self.assertEqual(bridges[0]["status"], "CANDIDATE_ONLY")
            self.assertEqual(bridges[0]["candidate_references"][0]["candidate_id"], "1")
            self.assertTrue(all(item.physical_id is None for item in result.observations))
            self.assertTrue(all(item["presence"] == "U" for item in result.presence))

    def test_tile_sources_collapse_only_byte_identical_duplicate_acquisitions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            same_a, same_b, conflict_a, conflict_b, later = [root / name for name in (
                "same-a.jpg", "same-b.jpg", "conflict-a.jpg", "conflict-b.jpg", "later.jpg")]
            same_a.write_bytes(b"identical")
            same_b.write_bytes(b"identical")
            conflict_a.write_bytes(b"first")
            conflict_b.write_bytes(b"second")
            later.write_bytes(b"later")
            tile_paths = {
                ("2018", 10, 20): same_a,
                ("2019", 10, 20): same_b,
                ("2018", 11, 20): conflict_a,
                ("2019", 11, 20): conflict_b,
                ("2020", 10, 20): later,
            }
            date_index = {
                ("2018", 10, 20): "2018-04-29",
                ("2019", 10, 20): "2018-04-29",
                ("2018", 11, 20): "2018-11-01",
                ("2019", 11, 20): "2018-11-01",
                ("2020", 10, 20): "2019-12-02",
            }

            grouped, duplicates = _tiles_grouped_by_actual_date(tile_paths, date_index)

            self.assertEqual(grouped["2018-04-29"][(10, 20)][0], "2018")
            self.assertNotIn((11, 20), grouped.get("2018-11-01", {}))
            self.assertEqual(grouped["2019-12-02"][(10, 20)][0], "2020")
            self.assertEqual([item["status"] for item in duplicates], [
                "DUPLICATE_ACTUAL_DATE_IDENTICAL_SOURCE_COLLAPSED",
                "DUPLICATE_ACTUAL_DATE_SOURCE_CONFLICT_SKIPPED",
            ])

    def test_masked_registration_is_candidate_without_ground_review(self):
        import cv2
        rng = np.random.default_rng(42)
        gray = rng.integers(50, 210, (256, 256), dtype=np.uint8)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        left = np.repeat(gray[:, :, None], 3, axis=2)
        right = cv2.warpAffine(left, np.float32([[1, 0, 4], [0, 1, -2]]), (256, 256))
        result = estimate_tile_registration(None, None, tile_x=420875, tile_y=312123,
                                            from_date="2018-04-29", to_date="2018-11-01",
                                            left_image=left, right_image=right)
        self.assertEqual(result.status, "GROUND_MASKED_CANDIDATE_NOT_REVIEWED")
        self.assertGreater(result.n_inliers, 12)
        self.assertLess(result.dx_m, 0)

    def test_products_are_nested_in_reference_10m_grid(self):
        observation = observed("a", "2018", "2018-04-29", 0, verified=False, key=None)
        result = reconstruct([observation])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "base10m.tif"
            with rasterio.open(reference, "w", driver="GTiff", width=100, height=100,
                               count=3, dtype="uint8", crs="EPSG:32649",
                               transform=from_origin(315000, 3799500, 10, 10)) as dst:
                dst.write(np.zeros((3, 100, 100), dtype=np.uint8))
            output = root / "products"
            write_products(output, result, anchor_raster=reference)
            with rasterio.open(output / "2018-04-29_ground_footprint.tif") as fine, rasterio.open(output / "2018-04-29_ground_footprint_BASE10M.tif") as base:
                self.assertEqual(fine.crs, base.crs)
                self.assertEqual(fine.bounds, base.bounds)
                self.assertEqual(fine.width, base.width * 10)
                self.assertEqual(fine.height, base.height * 10)
                self.assertEqual(base.res, (10.0, 10.0))
                self.assertEqual(base.transform.c % 10, 0)
                self.assertEqual(base.transform.f % 10, 0)
                self.assertIn(2, np.unique(base.read(1)))
                self.assertEqual(base.nodata, 255)

    def test_local_alignment_candidates_export_without_becoming_corrections(self):
        observation = observed("a", "2018", "2018-04-29", 0, verified=False, key=None)
        result = reconstruct([observation])
        alignment_cell = DisplacementCell(
            "tile-1-2", box(315150, 3799150, 315250, 3799250),
            1.25, -0.75, 0.4, "GROUND_MASKED_CANDIDATE_NOT_REVIEWED",
            "SYNTHETIC_TEST_TIE", "registration:2018-04-29:2018-11-01:tile-1-2",
            metadata={"maximum_masked_samples_available": 128,
                      "minimum_samples_required": 100,
                      "sample_support_status": "NUMERIC_PEAK_AVAILABLE",
                      "reference_window_evidence": [
                          {"reference_year_label": 2017,
                           "reference_raster_path": "base-2017.tif",
                           "reference_raster_sha256": "f" * 64,
                           "reference_grid_window_full_raster": [100, 200, 300, 400],
                           "cell_window_within_reference_grid": [2, 3, 4, 5],
                           "cell_window_full_reference_raster": [102, 203, 104, 205],
                           "reference_component_id_status": "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"},
                          {"reference_year_label": 2019,
                           "reference_raster_path": "base-2019.tif",
                           "reference_raster_sha256": "e" * 64,
                           "reference_grid_window_full_raster": [100, 200, 300, 400],
                           "cell_window_within_reference_grid": [2, 3, 4, 5],
                           "cell_window_full_reference_raster": [102, 203, 104, 205],
                           "reference_component_id_status": "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"}],
                      "source_tile_grid": {
                          "mosaic_diagnostic_path": "mosaic.json",
                          "mosaic_diagnostic_sha256": "a" * 64,
                          "tile_window": {"x0": 420796, "y0": 312172,
                                          "nx": 2, "ny": 2},
                          "tile_pixel_size": [256, 256],
                          "pixel_transform": [1, 0, 0, 0, -1, 10, 0, 0, 1],
                          "tile_col0": 1,
                          "tile_row0_north_to_south": 0,
                          "patch_tiles": 1,
                          "pixel_window": [256, 0, 512, 256],
                          "tiles_by_actual_date": {
                              "2018-04-29": {
                                  "expected_tile_ids": ["420797:312173"],
                                  "present_tiles": [{
                                      "tile_id": "420797:312173",
                                      "path": "tile-2018.jpg",
                                      "sha256": "b" * 64,
                                      "stitched_mosaic_grid": {
                                          "source_component_window_verified": True,
                                          "nominal_task_date": "2018-04-29",
                                          "mosaic_path": "mosaic-2018.tif",
                                          "pixel_window_col_row_width_height": [256, 0, 256, 256]}}],
                                  "missing_tile_ids": [], "coverage_fraction": 1.0},
                              "2018-11-01": {
                                  "expected_tile_ids": ["420797:312173"],
                                  "present_tiles": [{
                                      "tile_id": "420797:312173",
                                      "path": "tile-2019.jpg",
                                      "sha256": "c" * 64,
                                      "stitched_mosaic_grid": {
                                          "source_component_window_verified": True,
                                          "nominal_task_date": "2019-01-01",
                                          "mosaic_path": "mosaic-2019.tif",
                                          "pixel_window_col_row_width_height": [256, 0, 256, 256]}}],
                                      "missing_tile_ids": [], "coverage_fraction": 1.0}}}})
        alignment_cell.metadata.update(source_component_metadata(
            "2018-04-29", "2018-11-01"))
        alignment_field = LocalDisplacementField(
            "2018-04-29_to_2018-11-01", "2018-04-29", "2018-11-01",
            (alignment_cell,))
        annual_base_field = LocalDisplacementField(
            "2018-11-01_to_Xian_2019_TrueColor", None, "2018-11-01",
            (DisplacementCell(
                "base-cell", box(315150, 3799150, 315250, 3799250),
                -2.0, 1.0, None, "COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY",
                "COMPOSED_TEST", "composed:test",
                metadata=source_component_metadata("2018-11-01")),),
            reference_label="Xian_2019_TrueColor", reference_source="base:test",
            reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")

        def composed_field(field_id, status):
            return LocalDisplacementField(
                field_id, None, "2018-11-01",
                (DisplacementCell("base-cell", box(315150, 3799150, 315250, 3799250),
                                  -2.0, 1.0, None, status, "COMPOSED_TEST",
                                  f"composed:{field_id}",
                                  metadata=source_component_metadata("2018-11-01")),),
                reference_label="Xian_2019_TrueColor", reference_source="base:test",
                reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")

        failed_cycle_field = composed_field(
            "failed-cycle", "COMPOSED_CANDIDATE_CYCLE_SCREEN_FAILED")
        unchecked_cycle_field = composed_field(
            "unchecked-cycle", "COMPOSED_CANDIDATE_TREE_HAS_NO_CYCLE_CHECK")
        unknown_anchor_field = composed_field(
            "unknown-anchor", "COMPOSED_CANDIDATE_ANCHOR_BASE_DATE_UNKNOWN")
        spatial_cycle_field = composed_field(
            "spatial-cycle", "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY")
        spatial_failed_cycle_field = composed_field(
            "spatial-failed-cycle", "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_FAILED")
        spatial_unchecked_cycle_field = composed_field(
            "spatial-unchecked-cycle", "SPATIALLY_COMPOSED_CANDIDATE_TREE_NO_CYCLE_CHECK")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "base10m.tif"
            with rasterio.open(reference, "w", driver="GTiff", width=100, height=100,
                               count=3, dtype="uint8", crs="EPSG:32649",
                               transform=from_origin(315000, 3799500, 10, 10)) as dst:
                dst.write(np.zeros((3, 100, 100), dtype="uint8"))
            output = root / "products"
            manifest = write_products(
                output, result, anchor_raster=reference,
                alignment_fields=[alignment_field, annual_base_field,
                                  failed_cycle_field, unchecked_cycle_field,
                                  unknown_anchor_field, spatial_cycle_field,
                                  spatial_failed_cycle_field, spatial_unchecked_cycle_field])

            with rasterio.open(output / "2018-04-29_to_2018-11-01_alignment_quality.tif") as quality:
                self.assertEqual(quality.res, (1.0, 1.0))
                self.assertIn(2, np.unique(quality.read(1)))
                self.assertNotIn(1, np.unique(quality.read(1)))
            with rasterio.open(output / "2018-04-29_to_2018-11-01_alignment_quality_BASE10M.tif") as quality:
                self.assertEqual(quality.res, (10.0, 10.0))
                self.assertIn(2, np.unique(quality.read(1)))
            with rasterio.open(output / "2018-11-01_to_Xian_2019_TrueColor_alignment_quality.tif") as quality:
                self.assertIn(4, np.unique(quality.read(1)))
                self.assertNotIn(2, np.unique(quality.read(1)))
                self.assertNotIn(255, np.unique(quality.read(1)))
            with rasterio.open(output / "2018-11-01_to_Xian_2019_TrueColor_alignment_quality_BASE10M.tif") as quality:
                self.assertEqual(quality.res, (10.0, 10.0))
                self.assertIn(4, np.unique(quality.read(1)))
            for field_id, expected_code in (("failed-cycle", 5),
                                            ("unchecked-cycle", 6),
                                            ("unknown-anchor", 7),
                                            ("spatial-cycle", 4),
                                            ("spatial-failed-cycle", 5),
                                            ("spatial-unchecked-cycle", 6)):
                with rasterio.open(output / f"{field_id}_alignment_quality.tif") as quality:
                    self.assertIn(expected_code, np.unique(quality.read(1)))
            self.assertEqual(manifest["image_alignment_quality_codes"][
                "composed_cycle_screen_only_unreviewed_candidate"], 4)
            self.assertEqual(manifest["image_alignment_quality_codes"][
                "composed_anchor_acquisition_date_unknown_candidate"], 7)
            with fiona.open(output / "building_lineage.gpkg", layer="image_alignment_candidates") as layer:
                feature = next(iter(layer))
                properties = dict(feature["properties"])
                self.assertEqual(properties["accepted"], 0)
                self.assertEqual(properties["applied"], 0)
                self.assertEqual(properties["scope"], "IMAGE_TO_REFERENCE_GRID")
                self.assertEqual(properties["max_masked_n"], 128)
                self.assertEqual(properties["min_required_n"], 100)
                self.assertEqual(properties["sample_support"], "NUMERIC_PEAK_AVAILABLE")
                self.assertEqual(properties["tile_col0"], 1)
                self.assertEqual(properties["tile_mosaic_sha256"], "a" * 64)
                tile_provenance = json.loads(properties["tile_prov_json"])
                self.assertEqual(
                    tile_provenance["tiles_by_actual_date"]["2018-04-29"]
                    ["present_tiles"][0]["tile_id"], "420797:312173")
                component_grid = tile_provenance["tiles_by_actual_date"][
                    "2018-04-29"]["present_tiles"][0]["stitched_mosaic_grid"]
                self.assertEqual(component_grid["nominal_task_date"], "2018-04-29")
                self.assertEqual(component_grid["pixel_window_col_row_width_height"],
                                 [256, 0, 256, 256])
                nominal_source_grid = tile_provenance["tiles_by_actual_date"][
                    "2018-11-01"]["present_tiles"][0]["stitched_mosaic_grid"]
                self.assertEqual(nominal_source_grid["nominal_task_date"], "2019-01-01")
                self.assertNotEqual("2018-11-01", nominal_source_grid["nominal_task_date"])
                self.assertLess(len(properties["tile_prov_json"]), 32767)
                self.assertEqual(properties["reference_anchor_count"], 2)
                self.assertEqual(properties["reference_anchor_years"], "2017,2019")
                reference_windows = json.loads(
                    properties["reference_window_evidence_json"])
                self.assertEqual(
                    reference_windows[0]["cell_window_full_reference_raster"],
                    [102, 203, 104, 205])
            field_summary = next(item for item in manifest["image_alignment_fields"]
                                 if item["field_id"] == alignment_field.field_id)
            self.assertEqual(field_summary["source_tile_counts_by_actual_date"],
                             {"2018-04-29": 1, "2018-11-01": 1})
            self.assertEqual(
                field_summary["stitched_mosaic_component_window_verified_count"], 2)
            self.assertEqual(field_summary["reference_anchor_year_labels"], [2017, 2019])
            self.assertEqual(field_summary["reference_window_evidence_cell_count"], 1)

    def test_base_alignment_paths_write_separate_roof_candidate_tifs_only(self):
        observation = observed("a", "2018", "2018-04-29", 0, verified=False, key=None)
        result = reconstruct([observation])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "base10m.tif"
            with rasterio.open(reference, "w", driver="GTiff", width=100, height=100,
                               count=3, dtype="uint8", crs="EPSG:32649",
                               transform=from_origin(315000, 3799500, 10, 10)) as dst:
                dst.write(np.zeros((3, 100, 100), dtype=np.uint8))

            support = box(315150, 3799150, 315250, 3799250)
            def base_field(field_id, dx, dy, status, sample_support_status=None):
                tile_grid_metadata = ({
                    "source_tile_grid": {
                        "mosaic_diagnostic_path": "fixed-mosaic.json",
                        "mosaic_diagnostic_sha256": "d" * 64,
                        "tile_window": {"x0": 420796, "y0": 312172,
                                        "nx": 2, "ny": 2},
                        "tile_pixel_size": [256, 256],
                        "pixel_transform": [1, 0, 0, 0, -1, 10, 0, 0, 1],
                        "tile_col0": 0,
                        "tile_row0_north_to_south": 0,
                        "patch_tiles": 1,
                        "pixel_window": [0, 0, 256, 256],
                        "tiles_by_actual_date": {
                            "2018-04-29": {
                                "expected_tile_ids": ["420796:312173"],
                                "present_tiles": [{
                                    "tile_id": "420796:312173",
                                    "path": "fixed-2018.jpg",
                                    "sha256": "e" * 64}],
                                "missing_tile_ids": [], "coverage_fraction": 1.0}}}}
                    if field_id == "direct" else {})
                if sample_support_status is not None:
                    tile_grid_metadata["sample_support_status"] = sample_support_status
                tile_grid_metadata.update({
                    "source_component_scope": "SINGLE_SOURCE_COMPONENT_PER_CELL",
                    "source_component_ids": [{
                        "actual_date": "2018-04-29", "tile_id": "420796:312173",
                        "path": "fixed-2018.jpg", "sha256": "e" * 64,
                    }],
                    "source_component_coverage_fraction": 1.0,
                })
                candidate_cell = DisplacementCell(
                    "cell", support, dx, dy, None, status, "SYNTHETIC_BASE_TIE",
                    f"field:{field_id}:cell", reviewed=False,
                    independent_ground_evidence=False, positional_uncertainty_m=None,
                    metadata=tile_grid_metadata)
                return LocalDisplacementField(
                    field_id, None, "2018-04-29", (candidate_cell,),
                    reference_label="Xian_2019_TrueColor",
                    reference_source=str(reference.resolve()),
                    reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")

            direct = base_field("direct", 5.0, -3.0,
                                "DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED")
            composed = base_field("composed", -2.0, 1.0,
                                  "COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY")
            unresolved = base_field("no-tie", None, None,
                                    "DIRECT_BASE_TIE_NO_VALID_TIE",
                                    "INSUFFICIENT_MASKED_SAMPLES")
            no_correlation_peak = base_field(
                "no-peak", None, None, "DIRECT_BASE_TIE_NO_VALID_TIE",
                "NO_VALID_CORRELATION_SCORE")
            insufficient_anchors = base_field(
                "no-anchors", None, None, "MULTI_ANCHOR_BASE_TIE_INSUFFICIENT_SUPPORT",
                "INSUFFICIENT_ANCHOR_CANDIDATES")
            dominant_cells = (
                DisplacementCell(
                    "dominant", box(315200, 3799200, 315213.5, 3799210),
                    2.0, -1.0, None, "DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED",
                    "SYNTHETIC_BASE_TIE", "field:dominant:cell", reviewed=False,
                    independent_ground_evidence=False, positional_uncertainty_m=None,
                    metadata={
                        "source_component_scope": "SINGLE_SOURCE_COMPONENT_PER_CELL",
                        "source_component_ids": [{
                            "actual_date": "2018-04-29", "tile_id": "420796:312173",
                            "path": "fixed-2018.jpg", "sha256": "e" * 64,
                        }],
                        "source_component_coverage_fraction": 1.0,
                    }),
                DisplacementCell(
                    "remainder", box(315213.5, 3799200, 315215, 3799210),
                    None, None, None, "DIRECT_BASE_TIE_NO_VALID_TIE",
                    "SYNTHETIC_BASE_TIE", "field:dominant:remainder", reviewed=False,
                    independent_ground_evidence=False, positional_uncertainty_m=None,
                    metadata={
                        "source_component_scope": "SINGLE_SOURCE_COMPONENT_PER_CELL",
                        "source_component_ids": [{
                            "actual_date": "2018-04-29", "tile_id": "420796:312173",
                            "path": "fixed-2018.jpg", "sha256": "e" * 64,
                        }],
                        "source_component_coverage_fraction": 1.0,
                    }),
            )
            dominant = LocalDisplacementField(
                "dominant", None, "2018-04-29", dominant_cells,
                reference_label="Xian_2019_TrueColor",
                reference_source=str(reference.resolve()),
                reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")
            output = root / "products"
            manifest = write_products(
                output, result, anchor_raster=reference,
                alignment_fields=[direct, composed, unresolved, no_correlation_peak,
                                  insufficient_anchors, dominant])

            products = {item["field_id"]: item
                        for item in manifest["image_grid_roof_candidate_products"]}
            self.assertEqual(set(products), {"direct", "composed", "no-tie", "no-peak",
                                             "no-anchors", "dominant"})
            self.assertFalse(manifest["formal_ground_products_use_candidate_roof_geometry"])
            self.assertEqual(manifest["formal_ground_observations"], 0)
            with fiona.open(output / "building_lineage.gpkg",
                            layer="image_grid_roof_candidates") as layer:
                rows = {dict(feature["properties"])["field_id"]: feature
                        for feature in layer}
            self.assertEqual(set(rows), set(products))
            direct_row = dict(rows["direct"]["properties"])
            self.assertEqual(direct_row["ground_status"], "NOT_A_GROUND_FOOTPRINT")
            self.assertEqual(direct_row["candidate_shift"], 1)
            self.assertEqual(direct_row["candidate_dx_m"], 5.0)
            self.assertIsNone(direct_row["grid_unc_m"])
            self.assertEqual(direct_row["cell_support_fraction"], 1.0)
            dominant_row = dict(rows["dominant"]["properties"])
            self.assertEqual(
                dominant_row["status"],
                "UNACCEPTED_DOMINANT_CELL_IMAGE_GRID_SHIFT_ROOF_CANDIDATE")
            self.assertEqual(dominant_row["candidate_shift"], 1)
            self.assertAlmostEqual(dominant_row["cell_support_fraction"], 0.9)
            self.assertEqual(
                manifest["image_grid_roof_candidate_policy"]["dominant_cell_candidate_only"]["minimum_fraction"],
                0.8)
            direct_provenance = json.loads(direct_row["tile_prov_json"])
            self.assertEqual(
                direct_provenance["tiles_by_actual_date"]["2018-04-29"]
                ["present_tiles"][0]["tile_id"], "420796:312173")
            self.assertEqual(dict(rows["no-tie"]["properties"])["candidate_shift"], 0)

            with fiona.open(output / "building_lineage.gpkg", layer="roof_candidates") as layer:
                raw = next(iter(layer))
            self.assertEqual(shape(raw["geometry"]).bounds, observation.roof.bounds)
            with fiona.open(output / "building_lineage.gpkg", layer="image_grid_roof_candidates") as layer:
                direct_feature = next(feature for feature in layer
                                      if feature["properties"]["field_id"] == "direct")
            self.assertEqual(shape(direct_feature["geometry"]).bounds,
                             (observation.roof.bounds[0] + 5.0,
                              observation.roof.bounds[1] - 3.0,
                              observation.roof.bounds[2] + 5.0,
                              observation.roof.bounds[3] - 3.0))

            direct_tif = output / products["direct"]["candidate_tif"]
            with rasterio.open(direct_tif) as raster:
                direct_tags = raster.tags()
                actual = raster.read(1)
                expected = rasterize(
                    [(mapping(shape(direct_feature["geometry"])), 1)],
                    out_shape=actual.shape, transform=raster.transform,
                    fill=0, dtype="uint8")
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(direct_tags["PRODUCT_SCHEMA_VERSION"],
                             "xian_r5.grounded_products.v17")
            self.assertEqual(direct_tags["BASE_GRID_PIXEL_PROVENANCE_STATUS"],
                             "UNKNOWN_PER_PIXEL_SOURCE_COMPONENT_DATE_AND_VIEW_ANGLE")
            self.assertEqual(direct_tags["MOVING_ACTUAL_DATE"], "2018-04-29")
            self.assertEqual(direct_tags["PRODUCT_KIND"],
                             "IMAGE_GRID_ROOF_ALIGNMENT_CANDIDATE")
            self.assertEqual(direct_tags["ROOF_TO_GROUND_CORRECTION"], "NO")
            with rasterio.open(output / products["direct"]["quality_tif"]) as quality:
                self.assertEqual(quality.tags()["MOVING_ACTUAL_DATE"], "2018-04-29")
                self.assertEqual(quality.tags()["POSITIONAL_ACCURACY_CLAIMED"], "NO")
            with rasterio.open(output / products["direct"]["quality_tif"]) as quality:
                self.assertIn(2, np.unique(quality.read(1)))
            with rasterio.open(output / products["no-tie"]["candidate_tif"]) as raster:
                self.assertIn(2, np.unique(raster.read(1)))
                self.assertEqual(raster.nodata, 255)
            with rasterio.open(output / products["no-tie"]["quality_tif"]) as quality:
                self.assertIn(8, np.unique(quality.read(1)))
            for field_id, expected_code in (("no-tie", 8), ("no-peak", 9),
                                            ("no-anchors", 10)):
                with rasterio.open(output / f"{field_id}_alignment_quality.tif") as quality:
                    self.assertIn(expected_code, np.unique(quality.read(1)))
                with rasterio.open(output / products[field_id]["quality_tif"]) as quality:
                    self.assertIn(expected_code, np.unique(quality.read(1)))
                with rasterio.open(output / products[field_id]["quality_base10m_tif"]) as quality:
                    self.assertIn(expected_code, np.unique(quality.read(1)))
                    self.assertEqual(quality.tags()["AGGREGATION_POLICY"],
                                     "MAX_FINE_QUALITY_CODE")
            self.assertEqual(manifest["image_alignment_quality_codes"][
                "insufficient_masked_tie_samples"], 8)
            self.assertEqual(manifest["image_alignment_quality_codes"][
                "no_valid_correlation_peak"], 9)
            self.assertEqual(manifest["image_alignment_quality_codes"][
                "insufficient_multi_anchor_support"], 10)
            self.assertGreater(products["dominant"]["quality_pixel_counts"]["5"], 0)
            with rasterio.open(output / products["dominant"]["quality_tif"]) as quality:
                self.assertIn(5, np.unique(quality.read(1)))
                self.assertEqual(
                    quality.tags()["ROOF_CELL_SUPPORT_POLICY"],
                    "SINGLE_SOURCE_COMPONENT_REQUIRED;FULL_CELL_OR_UNACCEPTED_DOMINANT_CELL_AREA_FRACTION_GE_0.80")
            with rasterio.open(output / "2018-04-29_ground_footprint.tif") as raster:
                self.assertEqual(set(np.unique(raster.read(1)).tolist()), {0, 2})
                self.assertEqual(raster.tags()["ACTUAL_DATES_JSON"],
                                 '["2018-04-29"]')
                self.assertEqual(raster.tags()["REFERENCE_ACQUISITION_DATE_STATUS"],
                                 "UNKNOWN_ANNUAL_FILE_LABEL_ONLY")
                self.assertEqual(raster.tags()["BASE_GRID_PIXEL_PROVENANCE_STATUS"],
                                 "UNKNOWN_PER_PIXEL_SOURCE_COMPONENT_DATE_AND_VIEW_ANGLE")
            self.assertEqual(
                manifest["tif_metadata_contract"]["base_grid_pixel_provenance_status"],
                "UNKNOWN_PER_PIXEL_SOURCE_COMPONENT_DATE_AND_VIEW_ANGLE")

    def test_base_id_coarsening_preserves_single_identity_and_marks_conflict_unknown(self):
        unknown = np.uint32(4294967295)
        fine = np.array([[0, 17, 0, 18, unknown, 19],
                         [0, 0, 0, 20, 0, 0]], dtype=np.uint32)

        coarse = _coarsen_ids(fine, 2)

        np.testing.assert_array_equal(coarse, np.array([[17, unknown, unknown]], dtype=np.uint32))

    def test_base_change_coarsening_preserves_partial_single_state_and_marks_mixed_unknown(self):
        fine = np.array([[0, 1, 1, 2, 0, 4],
                         [0, 0, 0, 0, 0, 0]], dtype=np.uint8)

        coarse = _coarsen_change(fine, 2)

        np.testing.assert_array_equal(coarse, np.array([[1, 4, 4]], dtype=np.uint8))

    def test_quality_coarsening_preserves_extended_alignment_codes(self):
        fine = np.array([[0, 8, 0, 9, 0, 10],
                         [0, 0, 0, 0, 0, 0]], dtype=np.uint8)

        coarse = _coarsen_quality(fine, 2)

        np.testing.assert_array_equal(coarse, np.array([[8, 9, 10]], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
