#!/usr/bin/env python3
"""Read Stage-2 fixed-case imagery and emit isolated R5 reconstruction products.

This is an engineering run, not model validation.  R4 candidate polygons are
roof candidates; the default run has no reviewed ground or identity evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import replace
from itertools import combinations
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import fiona
import numpy as np
from PIL import Image
import rasterio
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xian_r5.alignment.field import DisplacementCell, LocalDisplacementField, group_fields
from xian_r5.alignment.roof_candidate import apply_accepted_roof_grid_alignment
from xian_r5.alignment.reviewed_field import (attach_reviewed_alignment_cells,
                                              recheck_reviewed_alignment_sources)
from xian_r5.alignment.field_io import (read_composed_alignment_fields,
                                        read_direct_base_tie_fields,
                                        read_component_tie_consensus_fields,
                                        read_lightglue_local_field_candidates,
                                        read_multi_anchor_base_tie_fields,
                                        read_spatial_alignment_composition_fields,
                                        paired_source_component_scope_for_support)
from xian_r5.alignment.comparison import compare_alignment_fields
from xian_r5.alignment.mosaic_grid import load_mosaic_grids_by_source_epoch
from xian_r5.reconstruction.core import Observation, reconstruct
from xian_r5.reconstruction.external_candidates import attach_overlapping_external_candidates
from xian_r5.reconstruction.products import write_products
from xian_r5.reconstruction.rbf_sam_candidates import attach_rbf_sam_footprint_candidates
from xian_r5.reconstruction.rbf_sam_handoff import prepare_rbf_sam_handoff
from xian_r5.reconstruction.reviewed_footprints import (attach_reviewed_ground_footprints,
                                                        recheck_review_sources)
from xian_r5.reconstruction.reviewed_candidates import (
    attach_reviewed_ground_footprint_candidates,
    recheck_reviewed_candidate_sources,
)
from xian_r5.reconstruction.reviewed_absences import load_reviewed_absences
from xian_r5.reconstruction.reviewed_identities import (
    attach_reviewed_physical_identities,
    recheck_reviewed_identity_sources,
)
from xian_r5.reconstruction.registration import (ORIGIN_LAT, ORIGIN_LON, ORIGIN_X, ORIGIN_Y,
                                                 STEP_DEG, TO_UTM, TO_WGS,
                                                 estimate_tile_registration,
                                                 registration_backend_status)
from xian_r5.reconstruction.registration_worker import SubprocessRegistrationBackend
from xian_r5.reconstruction.source_dates import associate_source_tile_dates


STAGE2 = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\17_stage2")
MAP = STAGE2 / "03_fixed_cases" / "R5_FIXED_CASE_TILE_MAP.csv"
GEOMETRY = STAGE2 / "03_fixed_cases" / "R5_FIXED_CASE_GEOMETRY.gpkg"
DEFAULT_OUTPUT = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\18_reconstruction\B03-TEX04_basegrid_01")
R4_STATUS = Path(r"F:\xian\西安_当前最佳候选交付_20260917\08_cloud_v2_candidate\RUN_STATUS.json")
BASE_DIR = Path(r"F:\xian\xian底图\xian搴曞浘")
DEFAULT_BASE_RASTER = BASE_DIR / "Xian_2019_TrueColor.tif"
BASE_RASTERS = [BASE_DIR / f"Xian_{year}_TrueColor.tif" for year in range(2017, 2023)]
RAW_DATE_CSV = Path(r"F:\xian\xian\date_audit\tables\tile_actual_date.csv")
SCENES = {"2018": "xa180429", "2019": "xa190101", "2020": "xa201231"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _footprint_candidate_record(candidate) -> dict:
    """Serialize candidate provenance and diagnostics into the evidence receipt."""
    return {
        "candidate_id": candidate.candidate_id,
        "source_ref": candidate.source_ref,
        "source_crs": candidate.source_crs,
        "source_date": candidate.source_capture_date,
        "source_date_status": candidate.source_date_status,
        "source_date_candidates": list(candidate.source_date_candidates),
        "source_tile_ids": list(candidate.source_tile_ids),
        "source_tile_actual_dates": [
            {"tile_id": tile_id, "actual_date": actual_date}
            for tile_id, actual_date in candidate.source_tile_actual_dates
        ],
        "source_scene_key": candidate.source_scene_key,
        "source_scene_linkage_status": candidate.source_scene_linkage_status,
        "source_date_provenance": candidate.source_date_provenance,
        "source_epoch_label": candidate.source_epoch_label,
        "source_geometry_semantics": candidate.source_geometry_semantics,
        "source_geometry_semantics_provenance": candidate.source_geometry_semantics_provenance,
        "source_accuracy_m": candidate.source_accuracy_m,
        "association": candidate.association_method,
        "match_iou": candidate.match_iou,
        "intersection_area_m2": candidate.intersection_area_m2,
        "roof_overlap_fraction": candidate.roof_overlap_fraction,
        "candidate_overlap_fraction": candidate.candidate_overlap_fraction,
        "candidate_centroid_dx_from_roof_m": candidate.candidate_centroid_dx_from_roof_m,
        "candidate_centroid_dy_from_roof_m": candidate.candidate_centroid_dy_from_roof_m,
        "centroid_distance_m": candidate.centroid_distance_m,
        "candidate_polygon_distance_m": candidate.candidate_polygon_distance_m,
        "candidate_search_radius_m": candidate.candidate_search_radius_m,
        "centroid_aligned_roof_iou": candidate.centroid_aligned_roof_iou,
        "centroid_aligned_iou_gain": candidate.centroid_aligned_iou_gain,
        "source_properties": candidate.source_properties,
        "area_m2": candidate.geometry.area,
        "status": "UNVERIFIED_CANDIDATE",
    }


def _resolved_path_key(path: Path | str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").casefold()


def _validate_alignment_reference_scopes(
    *,
    base_reference_fields: tuple[LocalDisplacementField, ...],
    date_pair_fields: tuple[LocalDisplacementField, ...],
    base_raster: Path,
    verified_base_raster_sources: tuple[Path | str, ...] = (),
    date_pair_reference_source: Path | None,
) -> None:
    """Keep image-to-base ties distinct from inter-date image ties.

    Base-tie fields may target any raster in the already verified, exactly
    co-gridded 10 m stack. LightGlue date-pair fields instead target the
    date-pair mosaic diagnostic that records their source images; that receipt
    is validated independently by the field reader.
    """
    base_keys = {_resolved_path_key(base_raster)}
    base_keys.update(_resolved_path_key(path) for path in verified_base_raster_sources)
    for field in base_reference_fields:
        if (not field.reference_source
                or _resolved_path_key(field.reference_source) not in base_keys):
            raise ValueError(
                f"Base-tie field {field.field_id} is outside the verified co-gridded 10 m raster stack")

    if date_pair_fields and date_pair_reference_source is None:
        raise ValueError("Date-pair alignment fields are missing their mosaic diagnostic")
    pair_key = (_resolved_path_key(date_pair_reference_source)
                if date_pair_reference_source is not None else None)
    for field in date_pair_fields:
        if (not field.reference_source
                or _resolved_path_key(field.reference_source) != pair_key):
            raise ValueError(
                f"Date-pair field {field.field_id} does not target its validated mosaic diagnostic")


def _date_coherent_neighborhoods(centers: set[tuple[int, int]], epochs: set[str]) -> dict:
    wanted = {(x + dx, y + dy) for x, y in centers for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    found = {}
    with RAW_DATE_CSV.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["task_name"] not in {SCENES[e] for e in epochs}:
                continue
            xy = int(row["tile_x"]), int(row["tile_y"])
            if xy in wanted and row["tile_present"] == "1" and row["actual_capture_date"]:
                epoch = next(e for e in epochs if SCENES[e] == row["task_name"])
                pixel = Path(r"F:\xian\xian") / SCENES[epoch] / "20" / str(xy[0]) / f"gesh_{xy[0]}_{xy[1]}_20.jpg"
                key = (epoch, *xy)
                if not pixel.is_file():
                    found[key] = None
                elif key in found and found[key] != row["actual_capture_date"]:
                    found[key] = None
                else:
                    found[key] = row["actual_capture_date"]
    return found


def _date_coherent_mosaic_centers(
    centers: set[tuple[int, int]], left_epoch: str, right_epoch: str,
    from_date: str, to_date: str, date_index: dict,
) -> list[tuple[int, int]]:
    """Return bounded common geographic windows backed by nine exact-date tiles.

    The center is a case-local geographic anchor, not a requirement that either
    acquisition has a mapped target tile at that exact coordinate. This lets
    adjacent dates register where their target-tile sets differ, while rejecting
    incomplete or date-mixed 3x3 source mosaics.
    """
    offsets = [(dx, dy) for dy in (1, 0, -1) for dx in (-1, 0, 1)]
    result = []
    for center_x, center_y in sorted(centers):
        left_complete = all(
            date_index.get((left_epoch, center_x + dx, center_y + dy)) == from_date
            for dx, dy in offsets)
        right_complete = all(
            date_index.get((right_epoch, center_x + dx, center_y + dy)) == to_date
            for dx, dy in offsets)
        if left_complete and right_complete:
            result.append((center_x, center_y))
    return result


def _mosaic(epoch: str, center: tuple[int, int], expected_date: str, date_index: dict) -> tuple[np.ndarray, list[dict]] | None:
    x, y = center
    images = []
    input_records = []
    for row_y in (y + 1, y, y - 1):
        line = []
        for col_x in (x - 1, x, x + 1):
            if date_index.get((epoch, col_x, row_y)) != expected_date:
                return None
            path = Path(r"F:\xian\xian") / SCENES[epoch] / "20" / str(col_x) / f"gesh_{col_x}_{row_y}_20.jpg"
            if not path.is_file():
                return None
            digest_before = sha256(path)
            with Image.open(path) as source:
                image = np.asarray(source.convert("RGB"))
            if image.shape != (256, 256, 3):
                return None
            tile_support = _registration_support_polygon(col_x, row_y, 1)
            input_records.append({
                "tile_id": f"{col_x}:{row_y}",
                "actual_date": expected_date,
                "path": str(path),
                "sha256": digest_before,
                "sha256_before": digest_before,
                "support_polygon_epsg32649": mapping(tile_support),
            })
            line.append(image)
        images.append(np.concatenate(line, axis=1))
    return np.concatenate(images, axis=0), input_records


def _covered_tiles(geometry) -> list[tuple[int, int]]:
    minx, miny, maxx, maxy = geometry.bounds
    lon0, lat0 = TO_WGS.transform(minx, miny)
    lon1, lat1 = TO_WGS.transform(maxx, maxy)
    x0 = ORIGIN_X + math.floor((min(lon0, lon1) - ORIGIN_LON) / STEP_DEG) - 1
    x1 = ORIGIN_X + math.floor((max(lon0, lon1) - ORIGIN_LON) / STEP_DEG) + 1
    y0 = ORIGIN_Y + math.floor((min(lat0, lat1) - ORIGIN_LAT) / STEP_DEG) - 1
    y1 = ORIGIN_Y + math.floor((max(lat0, lat1) - ORIGIN_LAT) / STEP_DEG) + 1
    covered = []
    for tile_x in range(x0, x1 + 1):
        for tile_y in range(y0, y1 + 1):
            west = ORIGIN_LON + (tile_x - ORIGIN_X) * STEP_DEG
            south = ORIGIN_LAT + (tile_y - ORIGIN_Y) * STEP_DEG
            lonlat = [(west, south), (west + STEP_DEG, south),
                      (west + STEP_DEG, south + STEP_DEG), (west, south + STEP_DEG)]
            tile = Polygon([TO_UTM.transform(lon, lat) for lon, lat in lonlat])
            if geometry.intersection(tile).area > 0.1:
                covered.append((tile_x, tile_y))
    return covered


def _registration_support_polygon(tile_x: int, tile_y: int, tile_span: int) -> Polygon:
    """Projected support cell for a same-grid tile or date-coherent mosaic tie."""
    if tile_span < 1:
        raise ValueError("Registration support span must be positive")
    west = ORIGIN_LON + (tile_x - ORIGIN_X) * STEP_DEG
    south = ORIGIN_LAT + (tile_y - ORIGIN_Y) * STEP_DEG
    east, north = west + STEP_DEG * tile_span, south + STEP_DEG * tile_span
    return Polygon([TO_UTM.transform(west, south), TO_UTM.transform(east, south),
                    TO_UTM.transform(east, north), TO_UTM.transform(west, north)])


def _registration_source_support(
    fallback: Polygon,
    source_tiles_by_date: dict[str, list[dict]],
) -> Polygon | MultiPolygon:
    """Use the exact projected source-tile union as the fitted field support."""
    geometries = []
    for entries in source_tiles_by_date.values():
        for entry in entries:
            polygon = entry.get("support_polygon_epsg32649")
            if not isinstance(polygon, dict):
                continue
            geometry = shape(polygon)
            if (geometry.is_empty or not geometry.is_valid
                    or geometry.geom_type not in {"Polygon", "MultiPolygon"}):
                continue
            geometries.append(geometry)
    if not geometries:
        return fallback
    support = unary_union(geometries)
    if (support.is_empty or not support.is_valid
            or support.geom_type not in {"Polygon", "MultiPolygon"}):
        raise ValueError("Registration source tiles do not form a valid polygon support")
    return support


def _associate_external_candidate_dates(observations: list[Observation],
                                        date_index: dict) -> dict:
    """Bind candidate outlines to exact source-tile dates, not verified capture dates."""
    cache = {}
    status_counts = defaultdict(int)
    date_counts = defaultdict(int)
    unique_candidate_count = 0
    for observation in observations:
        for candidate_index, candidate in enumerate(observation.footprint_candidates):
            epoch_label = candidate.source_epoch_label
            if epoch_label is None:
                continue
            scene_key = SCENES.get(epoch_label)
            cache_key = (candidate.source_ref, epoch_label)
            if cache_key not in cache:
                unique_candidate_count += 1
                tile_coordinates = _covered_tiles(candidate.geometry)
                date_lookup = {
                    tile: date_index.get((epoch_label, tile[0], tile[1]))
                    for tile in tile_coordinates
                }
                association = associate_source_tile_dates(
                    tile_coordinates, date_lookup)
                cache[cache_key] = (scene_key, association)
            scene_key, association = cache[cache_key]
            observation.footprint_candidates[candidate_index] = replace(
                candidate,
                source_date_status=association.status,
                source_date_candidates=association.candidate_dates,
                source_tile_ids=association.tile_ids,
                source_tile_actual_dates=association.tile_actual_dates,
                source_scene_key=scene_key,
                source_scene_linkage_status=(
                "EPOCH_LABEL_TO_SCENE_MAPPING_ONLY_SOURCE_VECTOR_NOT_HASH_BOUND"
                    if scene_key is not None else "NO_SCENE_MAPPING_FOR_USER_EPOCH_LABEL"),
                source_date_provenance=(
                    f"{RAW_DATE_CSV};scene={scene_key};"
                    "candidate geometry intersects these source tiles; source-vector linkage unverified"
                    if scene_key is not None else
                    "No scene mapping for user-reported source epoch label"),
            )
            status_counts[association.status] += 1
            for source_date in association.candidate_dates:
                date_counts[source_date] += 1

    return {
        "schema_version": "xian_r5.external_candidate_source_tile_dates.v1",
        "source_date_registry": str(RAW_DATE_CSV),
        "source_date_registry_sha256": sha256(RAW_DATE_CSV),
        "unique_candidate_geometry_count": unique_candidate_count,
        "observation_candidate_association_count": sum(
            len(observation.footprint_candidates) for observation in observations),
        "candidate_status_counts_by_observation_association": dict(
            sorted(status_counts.items())),
        "candidate_date_counts_by_observation_association": dict(
            sorted(date_counts.items())),
        "scene_mapping": dict(SCENES),
        "source_vector_linkage_status": (
            "USER_REPORTED_EPOCH_LABEL_MAPPED_TO_SOURCE_SCENE;"
            "VECTOR_SOURCE_RASTER_HASH_NOT_VERIFIED"),
        "capture_date_field_promoted": False,
        "neighbor_date_inference_used": False,
    }


def _tiles_grouped_by_actual_date(tile_paths: dict, date_index: dict) -> tuple[dict, list[dict]]:
    """Group tile sources by real acquisition date and collapse only byte-identical duplicates."""
    candidates = defaultdict(list)
    for (epoch, tile_x, tile_y), path in tile_paths.items():
        actual_date = date_index.get((epoch, tile_x, tile_y))
        if actual_date and path.is_file():
            candidates[(actual_date, tile_x, tile_y)].append((epoch, path))
    grouped = defaultdict(dict)
    duplicate_receipts = []
    for (actual_date, tile_x, tile_y), sources in sorted(candidates.items()):
        ordered_sources = sorted(sources, key=lambda item: item[0])
        hashes = {sha256(path) for _, path in ordered_sources}
        if len(hashes) > 1:
            duplicate_receipts.append({
                "actual_date": actual_date, "tile_x": tile_x, "tile_y": tile_y,
                "source_epochs": [epoch for epoch, _ in ordered_sources],
                "source_paths": [str(path) for _, path in ordered_sources],
                "status": "DUPLICATE_ACTUAL_DATE_SOURCE_CONFLICT_SKIPPED",
            })
            continue
        epoch, path = ordered_sources[0]
        grouped[actual_date][(tile_x, tile_y)] = (epoch, path)
        if len(ordered_sources) > 1:
            duplicate_receipts.append({
                "actual_date": actual_date, "tile_x": tile_x, "tile_y": tile_y,
                "source_epochs": [item[0] for item in ordered_sources],
                "source_paths": [str(item[1]) for item in ordered_sources],
                "selected_epoch": epoch,
                "status": "DUPLICATE_ACTUAL_DATE_IDENTICAL_SOURCE_COLLAPSED",
            })
    return grouped, duplicate_receipts


def _actual_date_pairs(actual_dates) -> list[tuple[str, str]]:
    """Return all ordered real-date pairs after collapsing duplicate dates."""
    return list(combinations(sorted(set(actual_dates)), 2))


def inspect_base_stack(reference_path: Path) -> dict:
    profiles = []
    reference = None
    for path in BASE_RASTERS:
        if not path.is_file():
            raise FileNotFoundError(path)
        with rasterio.open(path) as raster:
            current = {"crs": str(raster.crs), "transform": list(raster.transform),
                       "width": raster.width, "height": raster.height, "count": raster.count,
                       "resolution": list(raster.res), "nodata": raster.nodata,
                       "bounds": list(raster.bounds), "source_tags": raster.tags()}
        if current["crs"] != "EPSG:32649" or current["count"] < 3 or current["resolution"] != [10.0, 10.0]:
            raise ValueError(f"Unexpected 10m base profile: {path}")
        grid = (current["crs"], tuple(current["transform"]), current["width"], current["height"])
        if reference is None:
            reference = grid
        elif grid != reference:
            raise ValueError("The six 10m base rasters do not share one exact pixel grid")
        profiles.append({"path": str(path), "sha256": sha256(path), **current})
    if reference_path not in BASE_RASTERS:
        raise ValueError("Reference raster must be one of the six verified user-provided 10m layers")
    return {"rasters": profiles, "shared_grid_verified": True,
            "reference_path": str(reference_path),
            "reference_is_spatial_anchor_only": True,
            "annual_layer_label_is_not_exact_acquisition_date": True}


def run(case_id: str, output: Path, base_raster: Path = DEFAULT_BASE_RASTER,
        candidate_footprint_layer: Path | None = None,
        composed_alignment_gpkg: Path | None = None,
        composed_alignment_manifest: Path | None = None,
        direct_base_tie_manifest: Path | None = None,
        lightglue_local_field_diagnostic: Path | None = None,
        reviewed_ground_footprint_layer: Path | None = None,
        reviewed_absence_csv: Path | None = None,
        reviewed_image_alignment_layer: Path | None = None,
        multi_anchor_base_tie_manifests: list[Path] | None = None,
        registration_worker_python: Path | None = None,
        rbf_sam_prediction_bundle: Path | None = None,
        rbf_sam_mosaic_diagnostic: Path | None = None,
        candidate_footprint_source_epoch_label: str | None = None,
        candidate_footprint_source_geometry_semantics: str | None = None,
        component_tie_consensus_manifest: Path | None = None,
        rbf_sam_native_output: Path | None = None,
        rbf_sam_inference_manifest: Path | None = None,
        reviewed_ground_footprint_candidate_layer: Path | None = None,
        spatial_alignment_composition_manifest: Path | None = None,
        reviewed_identity_layer: Path | None = None,
        freeze_registration_candidates: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {output}")
    base_stack = inspect_base_stack(base_raster)
    rbf_sam_prediction_bundle, rbf_sam_native_bridge_report = prepare_rbf_sam_handoff(
        output,
        prediction_bundle=rbf_sam_prediction_bundle,
        mosaic_diagnostic=rbf_sam_mosaic_diagnostic,
        native_output=rbf_sam_native_output,
        inference_manifest=rbf_sam_inference_manifest,
    )
    expected_base_key = str(Path(base_raster).resolve()).casefold()
    expected_base_sha256 = next(
        item["sha256"] for item in base_stack["rasters"]
        if str(Path(item["path"]).resolve()).casefold() == expected_base_key)
    if (composed_alignment_gpkg is None) != (composed_alignment_manifest is None):
        raise ValueError("Composed alignment GeoPackage and manifest must be supplied together")
    if direct_base_tie_manifest is not None and multi_anchor_base_tie_manifests:
        raise ValueError("Use either a single direct base-tie receipt or multi-anchor receipts, not both")
    rbf_sam_source_grids = None
    if rbf_sam_mosaic_diagnostic is not None:
        rbf_sam_source_grids = load_mosaic_grids_by_source_epoch(
            rbf_sam_mosaic_diagnostic,
            source_root=Path(r"F:\xian\xian"),
            scenes_by_epoch=SCENES,
        )
    composed_alignment_fields = []
    composed_alignment_source_hashes = {}
    if composed_alignment_gpkg is not None and composed_alignment_manifest is not None:
        composed_alignment_source_hashes = {
            str(path.resolve()): sha256(path.resolve())
            for path in (composed_alignment_gpkg, composed_alignment_manifest)}
        composed_alignment_fields = read_composed_alignment_fields(
            composed_alignment_gpkg, composed_alignment_manifest)
        composed_manifest = json.loads(
            Path(composed_alignment_manifest).read_text(encoding="utf-8"))
        source_diagnostics = composed_manifest.get("source_diagnostics") or {}
        for prefix in ("base_pair", "base_tie"):
            source_path_value = source_diagnostics.get(f"{prefix}_path")
            if source_path_value:
                source_path = Path(source_path_value).resolve()
                expected_digest = source_diagnostics.get(f"{prefix}_sha256_after")
                actual_digest = sha256(source_path)
                if actual_digest != expected_digest:
                    raise RuntimeError(
                        f"Composed alignment {prefix} diagnostic changed after validation")
                composed_alignment_source_hashes[str(source_path)] = actual_digest
    direct_alignment_fields = []
    direct_alignment_source_hashes = {}
    if direct_base_tie_manifest is not None:
        direct_path = direct_base_tie_manifest.resolve()
        direct_alignment_source_hashes[str(direct_path)] = sha256(direct_path)
        verified_base_hashes = {
            _resolved_path_key(item["path"]): item["sha256"]
            for item in base_stack["rasters"]
        }
        direct_alignment_fields = read_direct_base_tie_fields(
            direct_path,
            verified_reference_raster_hashes=verified_base_hashes,
        )
        if sha256(direct_path) != direct_alignment_source_hashes[str(direct_path)]:
            raise RuntimeError("Direct base-tie diagnostic changed while being loaded")
    multi_anchor_alignment_fields = []
    multi_anchor_alignment_report = None
    multi_anchor_alignment_source_hashes = {}
    if multi_anchor_base_tie_manifests:
        multi_anchor_alignment_fields, multi_anchor_alignment_report = (
            read_multi_anchor_base_tie_fields(
                multi_anchor_base_tie_manifests,
                canonical_reference_raster=base_raster))
        if multi_anchor_alignment_report.get("case_id") != case_id:
            raise ValueError("Multi-anchor base-tie diagnostics belong to a different fixed case")
        multi_anchor_alignment_source_hashes = dict(
            multi_anchor_alignment_report["integrity_files_sha256"])
    component_tie_alignment_fields = []
    component_tie_alignment_report = None
    component_tie_alignment_source_hashes = {}
    if component_tie_consensus_manifest is not None:
        component_tie_alignment_fields, component_tie_alignment_report = (
            read_component_tie_consensus_fields(
                component_tie_consensus_manifest,
                canonical_reference_raster=base_raster))
        if component_tie_alignment_report.get("case_id") != case_id:
            raise ValueError("Component-tie consensus belongs to a different fixed case")
        component_tie_alignment_source_hashes = dict(
            component_tie_alignment_report["integrity_files_sha256"])
    lightglue_alignment_fields = []
    lightglue_alignment_source_hashes = {}
    lightglue_mosaic_diagnostic_path = None
    if lightglue_local_field_diagnostic is not None:
        lightglue_path = Path(lightglue_local_field_diagnostic).resolve()
        lightglue_report = json.loads(lightglue_path.read_text(encoding="utf-8"))
        if lightglue_report.get("case_id") != case_id:
            raise ValueError("LightGlue local-field diagnostic belongs to a different fixed case")
        lightglue_mosaic_diagnostic_path = Path(
            lightglue_report["input_mosaic_diagnostic"]).resolve()
        source_mosaic_grids = load_mosaic_grids_by_source_epoch(
            lightglue_mosaic_diagnostic_path,
            source_root=Path(r"F:\xian\xian"),
            scenes_by_epoch=SCENES,
        )
        lightglue_alignment_fields = read_lightglue_local_field_candidates(
            lightglue_path, expected_base_raster=base_raster,
            expected_base_sha256=expected_base_sha256,
            mosaic_grids_by_source_epoch=source_mosaic_grids)
        source_paths = [
            lightglue_path,
            Path(lightglue_report["outputs"]["candidate_vector"]),
            lightglue_mosaic_diagnostic_path,
            Path(lightglue_report["model"]["receipt_path"]),
            *(Path(path) for path in lightglue_report["model"]["weights_sha256"]),
        ]
        for field in lightglue_alignment_fields:
            for cell in field.cells:
                tile_grid = cell.metadata.get("source_tile_grid", {})
                for date_receipt in tile_grid.get("tiles_by_actual_date", {}).values():
                    source_paths.extend(
                        Path(tile["path"]) for tile in date_receipt.get("present_tiles", [])
                    )
        lightglue_alignment_source_hashes = {
            str(path.resolve()): sha256(path.resolve()) for path in source_paths}
    spatial_composed_alignment_fields = []
    spatial_alignment_composition_report = None
    spatial_alignment_tracked_paths = {}
    if spatial_alignment_composition_manifest is not None:
        spatial_manifest_path = Path(spatial_alignment_composition_manifest).resolve()
        spatial_alignment_composition_report = json.loads(
            spatial_manifest_path.read_text(encoding="utf-8"))
        spatial_composed_alignment_fields = read_spatial_alignment_composition_fields(
            spatial_manifest_path,
            expected_base_raster=base_raster,
            expected_base_sha256=expected_base_sha256,
            expected_case_id=case_id,
        )
        spatial_products = spatial_alignment_composition_report.get("products") or {}
        spatial_source_inputs = spatial_alignment_composition_report.get("source_inputs") or {}
        spatial_gpkg_path = (spatial_manifest_path.parent /
                             str(spatial_products.get("geopackage") or "")).resolve()
        spatial_source_paths = [
            ("manifest", spatial_manifest_path),
            ("geopackage", spatial_gpkg_path),
            ("direct_base_tie", Path(
                spatial_source_inputs["direct_base_tie_diagnostic"]["path"]).resolve()),
        ]
        spatial_source_paths.extend(
            (f"relative_diagnostic_{index:02d}", Path(record["path"]).resolve())
            for index, record in enumerate(spatial_source_inputs.get("relative_diagnostics", []))
        )
        spatial_alignment_tracked_paths = {
            f"spatial_alignment_{label}": path
            for label, path in spatial_source_paths
        }
    _validate_alignment_reference_scopes(
        base_reference_fields=tuple((
            *composed_alignment_fields,
            *direct_alignment_fields,
            *multi_anchor_alignment_fields,
            *component_tie_alignment_fields,
            *spatial_composed_alignment_fields,
        )),
        date_pair_fields=tuple(lightglue_alignment_fields),
        base_raster=base_raster,
        verified_base_raster_sources=tuple(
            Path(item["path"]) for item in base_stack["rasters"]),
        date_pair_reference_source=lightglue_mosaic_diagnostic_path,
    )
    tracked_sources = {"fixed_case_map": MAP, "fixed_case_geometry": GEOMETRY,
                       "actual_date_registry": RAW_DATE_CSV, "r4_status": R4_STATUS}
    if rbf_sam_prediction_bundle is not None:
        tracked_sources["rbf_sam_prediction_bundle"] = Path(rbf_sam_prediction_bundle).resolve()
        tracked_sources["rbf_sam_mosaic_diagnostic"] = Path(
            rbf_sam_mosaic_diagnostic).resolve()
    if rbf_sam_native_bridge_report is not None:
        tracked_sources["rbf_sam_native_output"] = Path(
            rbf_sam_native_output).resolve()
        tracked_sources["rbf_sam_inference_manifest"] = Path(
            rbf_sam_inference_manifest).resolve()
    if reviewed_ground_footprint_candidate_layer is not None:
        tracked_sources["reviewed_ground_footprint_candidate_layer"] = Path(
            reviewed_ground_footprint_candidate_layer).resolve()
    if reviewed_identity_layer is not None:
        tracked_sources["reviewed_identity_layer"] = Path(reviewed_identity_layer).resolve()
    tracked_sources.update(spatial_alignment_tracked_paths)
    source_hashes_before = {name: sha256(path) for name, path in tracked_sources.items()}
    if rbf_sam_native_bridge_report is not None:
        if (source_hashes_before["rbf_sam_native_output"]
                != rbf_sam_native_bridge_report["native_output_sha256"]
                or source_hashes_before["rbf_sam_inference_manifest"]
                != rbf_sam_native_bridge_report["binding_manifest_sha256"]):
            raise RuntimeError("RBF-SAM native inputs changed between conversion and run tracking")
    r4_before = source_hashes_before["r4_status"]
    with MAP.open("r", encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["case_id"] == case_id]
    if not rows:
        raise ValueError(f"No fixed-case rows for {case_id}")
    geometries = {}
    roofs_by_epoch = defaultdict(list)
    with fiona.open(GEOMETRY, layer="fixed_case_polygons") as source:
        if source.crs.to_epsg() != 32649:
            raise ValueError("Fixed-case geometry is not EPSG:32649")
        for feature in source:
            props = dict(feature["properties"])
            if props["case_id"] != case_id:
                continue
            geom = shape(feature["geometry"])
            epoch = str(props["epoch_label"])
            roofs_by_epoch[epoch].append(geom)
            if props["role"] == "TARGET":
                geometries[(epoch, str(props["object_id"]))] = geom
    observations = []
    unique = set()
    for row in rows:
        object_id = row.get("r4_polygon_object_id")
        if not object_id or (row["epoch_label"], object_id) in unique:
            continue
        unique.add((row["epoch_label"], object_id))
        geom = geometries.get((row["epoch_label"], object_id))
        if geom is None:
            raise ValueError(f"Missing source polygon: {row['epoch_label']}/{object_id}")
        path = Path(row["pixel_tile_path"]) if row.get("pixel_tile_path") else None
        if path is not None and not path.is_file():
            path = None
        observations.append(Observation(
            uid=f"{case_id}:{row['epoch_label']}:{object_id}",
            epoch_label=row["epoch_label"],
            actual_date=row["actual_date"] if row["actual_date"][:4].isdigit() else None,
            tile_id=row.get("actual_date_tile_id") or None,
            image_path=str(path) if path else None,
            roof=geom,
            source_ref=f"{GEOMETRY}#fixed_case_polygons/{case_id}/{row['epoch_label']}/{object_id}",
            candidate_identity=row.get("physical_track_id") or row.get("case_track_id") or None,
        ))
    if not observations:
        raise ValueError(f"No mapped target polygons for {case_id}")
    local_registration_backend = registration_backend_status() if not freeze_registration_candidates else {
        "available": False, "status": "SKIPPED_BY_RECONSTRUCTION_FREEZE",
        "import_error": None,
    }
    registration_worker = None
    if not freeze_registration_candidates and not local_registration_backend["available"] and registration_worker_python is not None:
        registration_worker = SubprocessRegistrationBackend(
            registration_worker_python, ROOT / "tools" / "xian_r5_registration_worker.py")
        registration_backend = registration_worker.status()
        registration_backend["in_process_import_error"] = local_registration_backend["import_error"]
    else:
        registration_backend = {
            **local_registration_backend,
            "adapter": "FROZEN_NOT_RUN" if freeze_registration_candidates else "IN_PROCESS",
        }
    if not registration_backend["available"]:
        for obs in observations:
            obs.registration_status = registration_backend["status"]

    def estimate_registration(*args, **kwargs):
        if registration_worker is not None and registration_backend["available"]:
            return registration_worker.estimate_tile_registration(*args, **kwargs)
        return estimate_tile_registration(*args, **kwargs)
    # Same-coordinate tile paths are taken from the actual-date map rows, not
    # inferred from nominal year.  Different dates never share one tile pair.
    tile_paths = {}
    for row in rows:
        if row.get("pixel_tile_path") and row.get("tile_x") and row.get("tile_y"):
            key = (row["epoch_label"], int(row["tile_x"]), int(row["tile_y"]))
            path = Path(row["pixel_tile_path"])
            if path.is_file():
                tile_paths[key] = path
    pixel_hashes_before = {str(path): sha256(path) for path in sorted(set(tile_paths.values()))}
    centers = {(x, y) for _, x, y in tile_paths}
    centers.update(xy for obs in observations for xy in _covered_tiles(obs.roof))
    date_index = _date_coherent_neighborhoods(centers, {o.epoch_label for o in observations})
    for obs in observations:
        covered = _covered_tiles(obs.roof)
        obs.covered_tiles = tuple(f"MAIN_Z20:{x}:{y}" for x, y in covered)
        source_paths = [tile_paths.get((obs.epoch_label, x, y)) for x, y in covered]
        obs.source_tile_ids = obs.covered_tiles
        obs.source_image_paths = tuple(str(path.resolve()) if path is not None else ""
                                       for path in source_paths)
        obs.source_hashes = tuple(pixel_hashes_before.get(str(path), "")
                                  if path is not None else "" for path in source_paths)
        dates = {date_index.get((obs.epoch_label, x, y)) for x, y in covered}
        obs.date_candidates = tuple(sorted(value for value in dates if value))
        if not covered or None in dates:
            obs.date_status = "COVERAGE_OR_DATE_UNKNOWN"
            obs.actual_date = None
        elif len(dates) > 1:
            obs.date_status = "DATE_HETEROGENEITY"
            obs.actual_date = None
        elif obs.actual_date != next(iter(dates)):
            obs.date_status = "CENTROID_DATE_CONFLICT"
            obs.actual_date = None
        else:
            obs.date_status = "DATE_COHERENT_FULL_ROOF_COVERAGE"
    reviewed_ground_footprint_report = None
    if reviewed_ground_footprint_layer is not None:
        reviewed_ground_footprint_report = attach_reviewed_ground_footprints(
            observations, reviewed_ground_footprint_layer)
    reviewed_candidate_report = None
    if reviewed_ground_footprint_candidate_layer is not None:
        reviewed_candidate_report = attach_reviewed_ground_footprint_candidates(
            observations, reviewed_ground_footprint_candidate_layer)
    reviewed_identity_report = None
    if reviewed_identity_layer is not None:
        reviewed_identity_report = attach_reviewed_physical_identities(
            observations, reviewed_identity_layer)
    reviewed_absence_evidence = []
    reviewed_absence_report = None
    if reviewed_absence_csv is not None:
        reviewed_absence_evidence, reviewed_absence_report = load_reviewed_absences(
            observations, reviewed_absence_csv)
    candidate_source_report = None
    if candidate_footprint_layer is not None:
        candidate_source_report = attach_overlapping_external_candidates(
            observations, candidate_footprint_layer,
            source_epoch_label=candidate_footprint_source_epoch_label,
            source_geometry_semantics=candidate_footprint_source_geometry_semantics)
        candidate_source_report["source_tile_date_association"] = (
            _associate_external_candidate_dates(observations, date_index))
    rbf_sam_candidate_report = None
    if rbf_sam_prediction_bundle is not None:
        rbf_sam_candidate_report = attach_rbf_sam_footprint_candidates(
            observations,
            rbf_sam_prediction_bundle,
            mosaic_diagnostic_path=rbf_sam_mosaic_diagnostic,
            mosaic_grids_by_source_epoch=rbf_sam_source_grids,
        )
    registrations = []
    alignment_cell_rows = []
    observation_registration_rank = {
        obs.uid: (0 if not registration_backend["available"] else -1)
        for obs in observations
    }

    def record_observation_registration(obs, reg) -> None:
        if reg.dx_m is not None and reg.dy_m is not None:
            rank = 3
        elif reg.n_inliers > 0:
            rank = 2
        else:
            rank = 1
        if rank <= observation_registration_rank[obs.uid]:
            return
        observation_registration_rank[obs.uid] = rank
        obs.registration_dx_m = reg.dx_m
        obs.registration_dy_m = reg.dy_m
        obs.registration_sigma_m = reg.sigma_m
        obs.registration_status = reg.status

    def retain_registration_field(
        reg, *, scope: str, tile_span: int, cell_id: str,
        source_tiles_by_date: dict[str, list[dict]],
    ) -> Polygon | MultiPolygon:
        support = _registration_source_support(
            _registration_support_polygon(reg.tile_x, reg.tile_y, tile_span),
            source_tiles_by_date,
        )
        registrations[-1]["support_polygon_epsg32649"] = mapping(support)
        registrations[-1]["correction_scope"] = "IMAGE_TO_REFERENCE_GRID"
        source_component_metadata = paired_source_component_scope_for_support(
            support, source_tiles_by_date, (reg.from_date, reg.to_date))
        alignment_cell_rows.append((reg.from_date, reg.to_date, DisplacementCell(
            cell_id=cell_id,
            support_geometry=support,
            correction_dx_m_candidate=reg.dx_m,
            correction_dy_m_candidate=reg.dy_m,
            fit_dispersion_m=reg.sigma_m,
            status=reg.status,
            method="OPENCV_SIFT_GROUND_MASKED_TRANSLATION",
            source_ref=f"{scope}:{reg.from_date}:{reg.to_date}:{reg.tile_x}:{reg.tile_y}",
            reviewed=bool(reg.anchor_reviewed),
            independent_ground_evidence=False,
            positional_uncertainty_m=None,
            metadata=source_component_metadata,
        )))
        return support

    dated_tiles, duplicate_date_sources = _tiles_grouped_by_actual_date(tile_paths, date_index)
    acquisition_dates = sorted(dated_tiles)
    if lightglue_alignment_fields:
        available_dates = set(acquisition_dates)
        for field in lightglue_alignment_fields:
            if field.reference_date not in available_dates or field.moving_date not in available_dates:
                raise ValueError(f"LightGlue field dates are outside this case's acquisitions: {field.field_id}")
    if (direct_alignment_fields
            and {field.moving_date for field in direct_alignment_fields} != set(acquisition_dates)):
        raise ValueError("Direct base-tie dates do not exactly match this fixed case's observations")
    date_pairs_to_estimate = ([] if freeze_registration_candidates
                              else _actual_date_pairs(acquisition_dates))
    for from_date, to_date in date_pairs_to_estimate:
        xy = sorted(set(dated_tiles[from_date]) & set(dated_tiles[to_date]))
        same_tile_registrations = {}
        for tile_x, tile_y in xy:
            left_epoch, left_path = dated_tiles[from_date][(tile_x, tile_y)]
            right_epoch, right_path = dated_tiles[to_date][(tile_x, tile_y)]
            reg = estimate_registration(
                left_path, right_path, tile_x=tile_x, tile_y=tile_y,
                from_date=from_date, to_date=to_date,
                left_roofs=roofs_by_epoch[left_epoch], right_roofs=roofs_by_epoch[right_epoch])
            reg_data = reg.as_dict()
            reg_data.update({"from_epoch_label": left_epoch, "to_epoch_label": right_epoch,
                             "scope": "SAME_TILE_ACTUAL_DATE_PAIR_CANDIDATE"})
            registrations.append(reg_data)
            same_tile_registrations[(tile_x, tile_y, left_epoch, right_epoch)] = reg
            tile_support = _registration_support_polygon(tile_x, tile_y, 1)
            retain_registration_field(
                reg, scope="SAME_TILE", tile_span=1,
                cell_id=f"same:{tile_x}:{tile_y}",
                source_tiles_by_date={
                    from_date: [{
                        "tile_id": f"{tile_x}:{tile_y}", "actual_date": from_date,
                        "path": str(left_path), "sha256": pixel_hashes_before[str(left_path)],
                        "support_polygon_epsg32649": mapping(tile_support),
                    }],
                    to_date: [{
                        "tile_id": f"{tile_x}:{tile_y}", "actual_date": to_date,
                        "path": str(right_path), "sha256": pixel_hashes_before[str(right_path)],
                        "support_polygon_epsg32649": mapping(tile_support),
                    }],
                })
            for obs in observations:
                if (obs.epoch_label == right_epoch and obs.actual_date == to_date
                        and obs.tile_id == f"MAIN_Z20:{tile_x}:{tile_y}"):
                    record_observation_registration(obs, reg)

        # Same-tile calls above emit explicit BLOCKED rows without opening
        # imagery. Do not assemble/hash 3x3 mosaics when the backend is absent.
        if not registration_backend["available"]:
            continue

        weak_statuses = {"INSUFFICIENT_MATCHES", "INSUFFICIENT_GROUND_CANDIDATE_FEATURES",
                         "REJECTED_WEAK_GROUND_CANDIDATE"}
        from_epochs = sorted({epoch for epoch, _ in dated_tiles[from_date].values()})
        to_epochs = sorted({epoch for epoch, _ in dated_tiles[to_date].values()})
        for left_epoch in from_epochs:
            for right_epoch in to_epochs:
                mosaic_centers = _date_coherent_mosaic_centers(
                    centers, left_epoch, right_epoch, from_date, to_date, date_index)
                for center_x, center_y in mosaic_centers:
                    center = (center_x, center_y)
                    same_tile_reg = same_tile_registrations.get(
                        (center_x, center_y, left_epoch, right_epoch))
                    if same_tile_reg is not None and same_tile_reg.status not in weak_statuses:
                        continue
                    left_mosaic = _mosaic(left_epoch, center, from_date, date_index)
                    right_mosaic = _mosaic(right_epoch, center, to_date, date_index)
                    if left_mosaic is None or right_mosaic is None:
                        continue
                    left_image, left_inputs = left_mosaic
                    right_image, right_inputs = right_mosaic
                    for item in [*left_inputs, *right_inputs]:
                        prior_hash = pixel_hashes_before.setdefault(item["path"], item["sha256_before"])
                        if prior_hash != item["sha256_before"]:
                            raise RuntimeError(f"Image tile changed while being read: {item['path']}")
                    mosaic_reg = estimate_registration(
                        None, None, tile_x=center_x - 1, tile_y=center_y - 1,
                        from_date=from_date, to_date=to_date,
                        left_roofs=roofs_by_epoch[left_epoch], right_roofs=roofs_by_epoch[right_epoch],
                        left_image=left_image, right_image=right_image, tile_span=3)
                    mosaic_data = mosaic_reg.as_dict()
                    mosaic_data.update({"scope": "DATE_COHERENT_3X3_MOSAIC_CANDIDATE",
                                        "center_tile": f"MAIN_Z20:{center_x}:{center_y}",
                                        "from_epoch_label": left_epoch,
                                        "to_epoch_label": right_epoch})
                    registrations.append(mosaic_data)
                    support = retain_registration_field(
                        mosaic_reg, scope="DATE_COHERENT_3X3_MOSAIC", tile_span=3,
                        cell_id=f"mosaic:{center_x - 1}:{center_y - 1}:3",
                        source_tiles_by_date={
                            from_date: left_inputs,
                            to_date: right_inputs,
                        })
                    for obs in observations:
                        if (obs.epoch_label == right_epoch and obs.actual_date == to_date
                                and support.covers(obs.roof)):
                            record_observation_registration(obs, mosaic_reg)
    alignment_fields = [*group_fields(alignment_cell_rows), *composed_alignment_fields,
                        *direct_alignment_fields, *lightglue_alignment_fields,
                        *multi_anchor_alignment_fields, *component_tie_alignment_fields,
                        *spatial_composed_alignment_fields]
    reviewed_alignment_report = None
    if reviewed_image_alignment_layer is not None:
        alignment_fields, reviewed_alignment_report = attach_reviewed_alignment_cells(
            alignment_fields, reviewed_image_alignment_layer)
    alignment_applications = []
    for obs in observations:
        if obs.ground_footprint is not None:
            obs.image_alignment_status = "GROUND_FOOTPRINT_ALREADY_IN_PROJECTED_REFERENCE_CRS"
            application = None
        else:
            application = apply_accepted_roof_grid_alignment(
                obs.roof, obs.actual_date, alignment_fields,
                reference_sources=tuple(
                    item["path"] for item in base_stack["rasters"]))
            obs.image_alignment_status = application.status
            if application.applied:
                obs.registered_roof = application.geometry
                obs.image_alignment_field_id = application.field_id
                obs.image_alignment_cell_id = application.cell_id
                obs.image_alignment_dx_m = application.correction_dx_m
                obs.image_alignment_dy_m = application.correction_dy_m
                obs.image_alignment_positional_uncertainty_m = application.positional_uncertainty_m
                obs.image_alignment_relative_uncertainty_m = (
                    application.relative_registration_uncertainty_m)
                obs.image_alignment_review_basis = application.review_basis
                obs.image_alignment_review_metadata = application.review_metadata
        alignment_applications.append({
            "uid": obs.uid, "actual_date": obs.actual_date,
            "status": obs.image_alignment_status,
            "applied": bool(application and application.applied),
            "field_id": application.field_id if application else None,
            "cell_id": application.cell_id if application else None,
            "correction_dx_m": application.correction_dx_m if application else None,
            "correction_dy_m": application.correction_dy_m if application else None,
            "positional_uncertainty_m": application.positional_uncertainty_m if application else None,
            "relative_registration_uncertainty_m": (
                application.relative_registration_uncertainty_m if application else None),
            "review_basis": application.review_basis if application else None,
            "geometry_basis": ("IMAGE_GRID_ALIGNED_ROOF_PROXY_NOT_GROUND"
                               if application and application.applied else "RAW_ROOF_PROXY"),
        })
    result = reconstruct(observations, absence_evidence=reviewed_absence_evidence)
    applied_field_cells = {
        (obs.image_alignment_field_id, obs.image_alignment_cell_id)
        for obs in observations if obs.registered_roof is not None
    }
    alignment_path_comparisons = []
    for direct_field in direct_alignment_fields:
        matches = [field for field in composed_alignment_fields
                   if field.moving_date == direct_field.moving_date
                   and field.reference_label == direct_field.reference_label]
        if len(matches) > 1:
            raise ValueError(f"Multiple composed fields match direct field {direct_field.field_id}")
        if matches:
            alignment_path_comparisons.append(compare_alignment_fields(direct_field, matches[0]))
    if composed_alignment_source_hashes:
        after = {path: sha256(Path(path)) for path in composed_alignment_source_hashes}
        if after != composed_alignment_source_hashes:
            raise RuntimeError("Composed alignment source changed before reconstruction products were written")
    if direct_alignment_source_hashes:
        after = {path: sha256(Path(path)) for path in direct_alignment_source_hashes}
        if after != direct_alignment_source_hashes:
            raise RuntimeError("Direct base-tie source changed before reconstruction products were written")
    if multi_anchor_alignment_source_hashes:
        after = {path: sha256(Path(path)) for path in multi_anchor_alignment_source_hashes}
        if after != multi_anchor_alignment_source_hashes:
            raise RuntimeError("Multi-anchor base-tie sources changed before products were written")
    if component_tie_alignment_source_hashes:
        after = {path: sha256(Path(path))
                 for path in component_tie_alignment_source_hashes}
        if after != component_tie_alignment_source_hashes:
            raise RuntimeError("Component-tie consensus sources changed before products were written")
    if lightglue_alignment_source_hashes:
        after = {path: sha256(Path(path)) for path in lightglue_alignment_source_hashes}
        if after != lightglue_alignment_source_hashes:
            raise RuntimeError("LightGlue local-field sources changed before products were written")
    if spatial_alignment_tracked_paths:
        expected_spatial_hashes = {
            str(path.resolve()): source_hashes_before[name]
            for name, path in spatial_alignment_tracked_paths.items()
        }
        after_spatial_hashes = {
            path: sha256(Path(path)) for path in expected_spatial_hashes
        }
        if after_spatial_hashes != expected_spatial_hashes:
            raise RuntimeError(
                "Spatial-composition inputs changed before products were written")
    product_alignment_fields = alignment_fields
    manifest = write_products(output, result, anchor_raster=base_raster,
                              alignment_fields=product_alignment_fields,
                              alignment_reference_rasters=tuple(
                                  Path(item["path"]) for item in base_stack["rasters"]))
    def record(obs: Observation) -> dict:
        return {
            "uid": obs.uid, "epoch_label": obs.epoch_label, "actual_date": obs.actual_date,
            "tile_id": obs.tile_id, "covered_tiles": obs.covered_tiles,
            "source_tile_ids": obs.source_tile_ids,
            "source_image_paths": obs.source_image_paths,
            "source_hashes": obs.source_hashes,
            "date_status": obs.date_status, "date_candidates": obs.date_candidates,
            "image_path": obs.image_path, "roof_source": obs.source_ref,
            "roof_area_m2": obs.roof.area, "footprint_status": obs.footprint_status,
            "ground_footprint_source": obs.footprint_evidence.source_ref if obs.footprint_evidence else None,
            "ground_footprint_method": obs.footprint_evidence.method if obs.footprint_evidence else None,
            "ground_footprint_reviewer_id": obs.footprint_evidence.reviewer_id if obs.footprint_evidence else None,
            "ground_footprint_reviewed_at": obs.footprint_evidence.reviewed_at if obs.footprint_evidence else None,
            "ground_footprint_evidence_path": obs.footprint_evidence.evidence_path if obs.footprint_evidence else None,
            "ground_footprint_evidence_sha256": obs.footprint_evidence.evidence_sha256 if obs.footprint_evidence else None,
            "ground_footprint_positional_sigma_m": obs.footprint_evidence.positional_sigma_m if obs.footprint_evidence else None,
            "ground_footprint_roof_to_ground_dx_m": obs.footprint_evidence.roof_to_ground_dx_m if obs.footprint_evidence else None,
            "ground_footprint_roof_to_ground_dy_m": obs.footprint_evidence.roof_to_ground_dy_m if obs.footprint_evidence else None,
            "ground_footprint_independent_evidence": obs.footprint_evidence.independent_ground_evidence if obs.footprint_evidence else None,
            "ground_footprint_reviewed": obs.footprint_evidence.reviewed if obs.footprint_evidence else None,
            "confirmed_identity_key": obs.confirmed_identity_key,
            "identity_reviewed": obs.identity_reviewed,
            "identity_evidence_type": obs.identity_evidence_type,
            "identity_evidence_ref": obs.identity_evidence_ref,
            "identity_evidence_path": obs.identity_evidence_path,
            "identity_evidence_sha256": obs.identity_evidence_sha256,
            "identity_reviewer_id": obs.identity_reviewer_id,
            "identity_reviewed_at": obs.identity_reviewed_at,
            "footprint_candidates": [_footprint_candidate_record(candidate)
                                     for candidate in obs.footprint_candidates],
            "image_alignment_status": obs.image_alignment_status,
            "image_alignment_applied": obs.registered_roof is not None,
            "image_alignment_field_id": obs.image_alignment_field_id,
            "image_alignment_cell_id": obs.image_alignment_cell_id,
            "image_alignment_dx_m": obs.image_alignment_dx_m,
            "image_alignment_dy_m": obs.image_alignment_dy_m,
            "image_alignment_positional_uncertainty_m": obs.image_alignment_positional_uncertainty_m,
            "image_alignment_relative_uncertainty_m": obs.image_alignment_relative_uncertainty_m,
            "image_alignment_review_basis": obs.image_alignment_review_basis,
            "physical_id": obs.physical_id, "candidate_identity": obs.candidate_identity,
            "registration_status": obs.registration_status,
            "registration_dx_m": obs.registration_dx_m, "registration_dy_m": obs.registration_dy_m,
            "registration_sigma_m": obs.registration_sigma_m,
        }
    source_hashes_after = {name: sha256(path) for name, path in tracked_sources.items()}
    pixel_hashes_after = {path: sha256(Path(path)) for path in pixel_hashes_before}
    base_stack_after = inspect_base_stack(base_raster)
    changed_sources = [name for name in tracked_sources
                       if source_hashes_before[name] != source_hashes_after[name]]
    changed_pixels = [path for path in pixel_hashes_before
                      if pixel_hashes_before[path] != pixel_hashes_after.get(path)]
    if changed_sources:
        raise RuntimeError(f"Source files changed during bounded reconstruction: {changed_sources}")
    if changed_pixels:
        raise RuntimeError(f"Source image tiles changed during bounded reconstruction: {changed_pixels}")
    if [row["sha256"] for row in base_stack_after["rasters"]] != [row["sha256"] for row in base_stack["rasters"]]:
        raise RuntimeError("User-provided 10m base changed during the bounded run")
    if reviewed_ground_footprint_report is not None:
        reviewed_ground_footprint_report["run_end_integrity_files_sha256"] = recheck_review_sources(
            reviewed_ground_footprint_report)
        reviewed_ground_footprint_report["run_end_hash_recheck"] = "PASS"
    if reviewed_candidate_report is not None:
        reviewed_candidate_report["run_end_integrity_files_sha256"] = (
            recheck_reviewed_candidate_sources(reviewed_candidate_report))
        reviewed_candidate_report["run_end_hash_recheck"] = "PASS"
    if reviewed_absence_report is not None:
        reviewed_absence_report["run_end_integrity_files_sha256"] = recheck_review_sources(
            reviewed_absence_report)
        reviewed_absence_report["run_end_hash_recheck"] = "PASS"
    if reviewed_identity_report is not None:
        reviewed_identity_report["run_end_integrity_files_sha256"] = (
            recheck_reviewed_identity_sources(reviewed_identity_report))
        reviewed_identity_report["run_end_hash_recheck"] = "PASS"
    if reviewed_alignment_report is not None:
        reviewed_alignment_report["run_end_integrity_files_sha256"] = recheck_reviewed_alignment_sources(
            reviewed_alignment_report)
        reviewed_alignment_report["run_end_hash_recheck"] = "PASS"

    input_sha256 = {}
    for name in ("fixed_case_map", "fixed_case_geometry", "actual_date_registry"):
        input_sha256[name] = source_hashes_after[name]
        input_sha256[f"{name}_before"] = source_hashes_before[name]
        input_sha256[f"{name}_after"] = source_hashes_after[name]
    input_sha256["r4_status_before"] = source_hashes_before["r4_status"]
    input_sha256["r4_status_after"] = source_hashes_after["r4_status"]
    for name in spatial_alignment_tracked_paths:
        input_sha256[f"{name}_before"] = source_hashes_before[name]
        input_sha256[f"{name}_after"] = source_hashes_after[name]
    if reviewed_candidate_report is not None:
        input_sha256["reviewed_ground_footprint_candidate_layer_before"] = (
            source_hashes_before["reviewed_ground_footprint_candidate_layer"])
        input_sha256["reviewed_ground_footprint_candidate_layer_after"] = (
            source_hashes_after["reviewed_ground_footprint_candidate_layer"])
    if reviewed_identity_report is not None:
        input_sha256["reviewed_identity_layer_before"] = source_hashes_before[
            "reviewed_identity_layer"]
        input_sha256["reviewed_identity_layer_after"] = source_hashes_after[
            "reviewed_identity_layer"]
    if rbf_sam_candidate_report is not None:
        input_sha256["rbf_sam_prediction_bundle_before"] = source_hashes_before[
            "rbf_sam_prediction_bundle"]
        input_sha256["rbf_sam_prediction_bundle_after"] = source_hashes_after[
            "rbf_sam_prediction_bundle"]
        input_sha256["rbf_sam_mosaic_diagnostic_before"] = source_hashes_before[
            "rbf_sam_mosaic_diagnostic"]
        input_sha256["rbf_sam_mosaic_diagnostic_after"] = source_hashes_after[
            "rbf_sam_mosaic_diagnostic"]
    if rbf_sam_native_bridge_report is not None:
        input_sha256["rbf_sam_native_output_before"] = source_hashes_before[
            "rbf_sam_native_output"]
        input_sha256["rbf_sam_native_output_after"] = source_hashes_after[
            "rbf_sam_native_output"]
        input_sha256["rbf_sam_inference_manifest_before"] = source_hashes_before[
            "rbf_sam_inference_manifest"]
        input_sha256["rbf_sam_inference_manifest_after"] = source_hashes_after[
            "rbf_sam_inference_manifest"]
    pixel_inputs = [{"path": path,
                     "sha256": pixel_hashes_after[path],
                     "sha256_before": pixel_hashes_before[path],
                     "sha256_after": pixel_hashes_after[path]}
                    for path in sorted(pixel_hashes_before)]
    evidence = {
        "schema_version": "xian_r5.grounded_fixed_case_run.v1", "case_id": case_id,
        "registration_backend": registration_backend,
        "registration_candidate_run_status": (
            "NOT_RUN_BY_EXPLICIT_RECONSTRUCTION_FREEZE"
            if freeze_registration_candidates else "RUN_AS_CONFIGURED"),
        "input_sha256": input_sha256,
        "input_paths": {name: str(path) for name, path in tracked_sources.items()},
        "pixel_inputs": pixel_inputs,
        "neighborhood_date_source": str(RAW_DATE_CSV),
        "observations": [record(o) for o in result.observations],
        "external_footprint_candidate_source": candidate_source_report,
        "rbf_sam_candidate_source": rbf_sam_candidate_report,
        "rbf_sam_native_bridge_source": rbf_sam_native_bridge_report,
        "reviewed_ground_footprint_source": reviewed_ground_footprint_report,
        "reviewed_identity_source": reviewed_identity_report,
        "reviewed_ground_footprint_candidate_source": reviewed_candidate_report,
        "reviewed_absence_source": reviewed_absence_report,
        "reviewed_image_alignment_source": reviewed_alignment_report,
        "image_grid_alignment_applications": alignment_applications,
        "image_grid_alignment_applied_count": sum(item["applied"] for item in alignment_applications),
        "registrations": registrations,
        "registration_date_pairing": {
            "actual_dates_in_order": acquisition_dates,
            "actual_date_pairs_considered": _actual_date_pairs(acquisition_dates),
            "nominal_epoch_order_used_for_ties": False,
            "duplicate_sources": duplicate_date_sources,
        },
        "candidate_edges": result.candidate_edges,
        "identity_graph": result.identity_graph, "presence": result.presence,
        "identity_graph_products": {
            "node_layer": "building_lineage.gpkg#identity_graph_nodes",
            "candidate_edge_layer": "building_lineage.gpkg#candidate_identity_edges",
            "presence_state_layer": "building_lineage.gpkg#building_presence",
            "candidate_edges_are_confirmed_physical_ids": False,
        },
        "transitions": result.transitions, "resolver_events": result.resolver_events,
        "repair_candidates": [{"merged_uid": child["merged_uid"], "prior_uid": child["prior_uid"],
                               "child_index": child["child_index"], "status": child["status"],
                               "geometry_layer": "building_lineage.gpkg#instance_repair_candidates"}
                              for child in result.repair_candidates],
        "product_manifest": manifest,
        "image_grid_roof_candidate_preview_rendered": manifest.get(
            "image_alignment_candidate_preview_rendered", False),
        "image_grid_roof_candidate_product_count": len(
            manifest.get("image_grid_roof_candidate_products", [])),
        "candidate_aligned_roofs_are_ground_footprints": False,
        "10m_base_stack": base_stack,
        "ground_footprints_confirmed": sum(o.ground_footprint is not None for o in observations),
        "confirmed_changes": sum(r["state"] != "UNRESOLVED" for r in result.transitions),
        "rbf_sam_status": (
            "PREDICTION_BUNDLE_IMPORTED_UNVERIFIED"
            if rbf_sam_candidate_report is not None
            else "NOT_RUN_NO_OFFICIAL_CHECKPOINT"),
        "interpretation": "ENGINEERING_DIAGNOSTIC_NOT_FORMAL_ACCEPTANCE",
    }
    evidence["10m_base_stack_sha256_recheck"] = "ALL_SIX_MATCH"
    evidence["input_sha256_recheck"] = "ALL_FIXED_AND_DATE_INPUTS_MATCH"
    evidence["pixel_input_sha256_recheck"] = "ALL_READ_TILE_HASHES_MATCH"
    if composed_alignment_source_hashes:
        final_hashes = {path: sha256(Path(path)) for path in composed_alignment_source_hashes}
        if final_hashes != composed_alignment_source_hashes:
            raise RuntimeError("Composed alignment source changed during reconstruction")
        evidence["composed_alignment_field_sources"] = [
            {"path": path, "sha256": digest}
            for path, digest in composed_alignment_source_hashes.items()]
        evidence["composed_alignment_field_count"] = len(composed_alignment_fields)
        evidence["composed_alignment_field_applied"] = any(
            (field.field_id, cell.cell_id) in applied_field_cells
            for field in composed_alignment_fields for cell in field.cells)
    if direct_alignment_source_hashes:
        final_hashes = {path: sha256(Path(path)) for path in direct_alignment_source_hashes}
        if final_hashes != direct_alignment_source_hashes:
            raise RuntimeError("Direct base-tie source changed during reconstruction")
        evidence["direct_base_tie_field_sources"] = [
            {"path": path, "sha256": digest}
            for path, digest in direct_alignment_source_hashes.items()]
        evidence["direct_base_tie_field_count"] = len(direct_alignment_fields)
        evidence["direct_base_tie_field_applied"] = any(
            (field.field_id, cell.cell_id) in applied_field_cells
            for field in direct_alignment_fields for cell in field.cells)
    if multi_anchor_alignment_source_hashes:
        final_hashes = {path: sha256(Path(path)) for path in multi_anchor_alignment_source_hashes}
        if final_hashes != multi_anchor_alignment_source_hashes:
            raise RuntimeError("Multi-anchor base-tie sources changed during reconstruction")
        evidence["multi_anchor_base_tie_source_report"] = multi_anchor_alignment_report
        evidence["multi_anchor_base_tie_field_count"] = len(multi_anchor_alignment_fields)
        evidence["multi_anchor_base_tie_numeric_cell_count"] = sum(
            cell.correction_dx_m_candidate is not None
            for field in multi_anchor_alignment_fields for cell in field.cells)
        evidence["multi_anchor_base_tie_accepted_cell_count"] = sum(
            cell.accepted for field in multi_anchor_alignment_fields for cell in field.cells)
        evidence["multi_anchor_base_tie_field_applied"] = any(
            (field.field_id, cell.cell_id) in applied_field_cells
            for field in multi_anchor_alignment_fields for cell in field.cells)
    if component_tie_alignment_source_hashes:
        final_hashes = {path: sha256(Path(path))
                        for path in component_tie_alignment_source_hashes}
        if final_hashes != component_tie_alignment_source_hashes:
            raise RuntimeError("Component-tie consensus sources changed during reconstruction")
        evidence["component_tie_consensus_source_report"] = component_tie_alignment_report
        evidence["component_tie_consensus_field_count"] = len(component_tie_alignment_fields)
        evidence["component_tie_consensus_cell_count"] = sum(
            len(field.cells) for field in component_tie_alignment_fields)
        evidence["component_tie_consensus_numeric_cell_count"] = sum(
            cell.correction_dx_m_candidate is not None
            for field in component_tie_alignment_fields for cell in field.cells)
        evidence["component_tie_consensus_accepted_cell_count"] = sum(
            cell.accepted for field in component_tie_alignment_fields for cell in field.cells)
        evidence["component_tie_consensus_field_applied"] = any(
            (field.field_id, cell.cell_id) in applied_field_cells
            for field in component_tie_alignment_fields for cell in field.cells)
    if lightglue_alignment_source_hashes:
        final_hashes = {path: sha256(Path(path)) for path in lightglue_alignment_source_hashes}
        if final_hashes != lightglue_alignment_source_hashes:
            raise RuntimeError("LightGlue local-field source changed during reconstruction")
        evidence["lightglue_local_field_sources"] = [
            {"path": path, "sha256": digest}
            for path, digest in lightglue_alignment_source_hashes.items()]
        evidence["lightglue_local_field_count"] = len(lightglue_alignment_fields)
        evidence["lightglue_local_field_cell_count"] = sum(
            len(field.cells) for field in lightglue_alignment_fields)
        evidence["lightglue_local_field_accepted_cell_count"] = sum(
            cell.accepted for field in lightglue_alignment_fields for cell in field.cells)
        evidence["lightglue_local_field_applied"] = any(
            (field.field_id, cell.cell_id) in applied_field_cells
            for field in lightglue_alignment_fields for cell in field.cells)
    if spatial_alignment_tracked_paths:
        evidence["spatial_alignment_composition"] = {
            "manifest_path": str(Path(spatial_alignment_composition_manifest).resolve()),
            "field_count": len(spatial_composed_alignment_fields),
            "cell_count": sum(len(field.cells)
                               for field in spatial_composed_alignment_fields),
            "cells_by_actual_date": {
                field.moving_date: len(field.cells)
                for field in spatial_composed_alignment_fields
            },
            "status_counts": {
                status: sum(cell.status == status
                            for field in spatial_composed_alignment_fields
                            for cell in field.cells)
                for status in sorted({cell.status
                                      for field in spatial_composed_alignment_fields
                                      for cell in field.cells})
            },
            "candidate_fields_in_product_manifest": True,
            "field_applied": any(cell.applied
                                  for field in spatial_composed_alignment_fields
                                  for cell in field.cells),
            "reviewed": any(cell.reviewed
                            for field in spatial_composed_alignment_fields
                            for cell in field.cells),
            "absolute_ground_position_resolved": False,
            "roof_to_ground_parallax_resolved": False,
            "formal_change_use": False,
            "annual_reference_acquisition_date": "UNKNOWN",
            "per_pixel_base_component_provenance": "NOT_RECORDED",
            "input_paths_sha256": {
                name: {"path": str(path), "sha256": source_hashes_after[name]}
                for name, path in spatial_alignment_tracked_paths.items()
            },
        }
    if alignment_path_comparisons:
        evidence["alignment_path_comparisons"] = alignment_path_comparisons
    (output / "EVIDENCE_RECEIPT.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="B03-TEX04")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-raster", type=Path, default=DEFAULT_BASE_RASTER)
    parser.add_argument("--candidate-footprint-layer", type=Path, default=None,
                        help="Optional external geometry layer; overlaps and polygons within 20 m remain unverified candidates")
    parser.add_argument("--candidate-footprint-source-epoch-label", type=str, default=None,
                        help="User-reported source epoch label only (not an exact acquisition date or ground-truth claim)")
    parser.add_argument("--candidate-footprint-source-geometry-semantics", type=str,
                        choices=("ROOF_BOUNDARY_FROM_IMAGE",
                                 "GROUND_FOOTPRINT_FROM_ORTHO_OR_SURVEY", "UNKNOWN"),
                        default=None,
                        help="User-reported source geometry semantics; does not verify ground-footprint status")
    parser.add_argument("--reviewed-ground-footprint-layer", type=Path, default=None,
                        help="Optional strict GeoPackage of independently evidenced, human-reviewed EPSG:32649 footprints")
    parser.add_argument("--reviewed-ground-footprint-candidate-layer", type=Path, default=None,
                        help="Optional candidate-only visual ground-boundary layer; absolute positional accuracy and presence remain unresolved")
    parser.add_argument("--reviewed-absence-csv", type=Path, default=None,
                        help="Optional strict CSV of reviewed full-footprint-coverage absence evidence")
    parser.add_argument("--reviewed-identity-layer", type=Path, default=None,
                        help="Optional non-spatial GeoPackage layer of independently reviewed physical identities; does not require ground footprints")
    parser.add_argument("--reviewed-image-alignment-layer", type=Path, default=None,
                        help="Optional strict GeoPackage layer reviewing exact local image-grid tie cells; reviewed roof shifts never become ground footprints")
    parser.add_argument("--composed-alignment-gpkg", type=Path, default=None,
                        help="Optional composed image-to-reference candidate field GeoPackage")
    parser.add_argument("--composed-alignment-manifest", type=Path, default=None,
                        help="Diagnostic JSON paired with --composed-alignment-gpkg")
    parser.add_argument("--direct-base-tie-manifest", type=Path, default=None,
                        help="Optional actual-date local-tie diagnostic directly against the selected base raster")
    parser.add_argument("--multi-anchor-base-tie-manifest", type=Path, action="append", default=None,
                        help="Repeat for candidate-only local-tie diagnostics against co-gridded annual 10 m anchors; requires the selected Xian_2019 grid")
    parser.add_argument("--component-tie-consensus-manifest", type=Path, default=None,
                        help="Optional exact-source-tile, multi-reference consensus receipt; remains candidate-only")
    parser.add_argument("--lightglue-local-field-diagnostic", type=Path, default=None,
                        help="Optional date-aware ALIKED/LightGlue local-field candidate receipt; never applied automatically")
    parser.add_argument("--spatial-alignment-composition-manifest", type=Path, default=None,
                        help="Optional spatial-overlap-composed actual-date/base-grid candidate receipt; candidate-only")
    parser.add_argument("--registration-worker-python", type=Path, default=None,
                        help="Optional Python executable with a working OpenCV/SIFT runtime; image shifts remain unreviewed candidates")
    parser.add_argument("--freeze-registration-candidates", action="store_true",
                        help="Do not launch new image-registration candidate diagnostics; only consume explicitly supplied reviewed evidence")
    parser.add_argument("--rbf-sam-prediction-bundle", type=Path, default=None,
                        help="Optional fixed-case xian_r5.rbf_sam_prediction_bundle.v2; imports translated roof masks as unverified candidates only")
    parser.add_argument("--rbf-sam-mosaic-diagnostic", type=Path, default=None,
                        help="Exact-date source-tile diagnostic required with either RBF-SAM input mode; derives model-input georeferencing")
    parser.add_argument("--rbf-sam-native-output", type=Path, default=None,
                        help="Optional native MMDetection JSON output; mutually exclusive with --rbf-sam-prediction-bundle")
    parser.add_argument("--rbf-sam-inference-manifest", type=Path, default=None,
                        help="Exact observation/date/tile/prompt/model-hash bindings for --rbf-sam-native-output")
    args = parser.parse_args()
    receipt = run(args.case_id, args.output, args.base_raster, args.candidate_footprint_layer,
                  args.composed_alignment_gpkg, args.composed_alignment_manifest,
                  args.direct_base_tie_manifest, args.lightglue_local_field_diagnostic,
                  args.reviewed_ground_footprint_layer, args.reviewed_absence_csv,
                  args.reviewed_image_alignment_layer, args.multi_anchor_base_tie_manifest,
                  args.registration_worker_python, args.rbf_sam_prediction_bundle,
                  args.rbf_sam_mosaic_diagnostic,
                  candidate_footprint_source_epoch_label=args.candidate_footprint_source_epoch_label,
                  candidate_footprint_source_geometry_semantics=(
                      args.candidate_footprint_source_geometry_semantics),
                  component_tie_consensus_manifest=args.component_tie_consensus_manifest,
                  rbf_sam_native_output=args.rbf_sam_native_output,
                  rbf_sam_inference_manifest=args.rbf_sam_inference_manifest,
                  reviewed_ground_footprint_candidate_layer=(
                      args.reviewed_ground_footprint_candidate_layer),
                  spatial_alignment_composition_manifest=(
                      args.spatial_alignment_composition_manifest),
                  reviewed_identity_layer=args.reviewed_identity_layer,
                  freeze_registration_candidates=args.freeze_registration_candidates)
    print(json.dumps({"output": str(args.output), "observations": len(receipt["observations"]),
                      "registrations": len(receipt["registrations"]),
                      "ground_footprints_confirmed": receipt["ground_footprints_confirmed"],
                      "confirmed_changes": receipt["confirmed_changes"]}, ensure_ascii=False))
