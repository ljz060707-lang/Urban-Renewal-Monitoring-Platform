"""Run frozen B0 geometry on Shenzhen without materialising huge probability TIFFs.

The immutable GEID mosaics are very large.  This runner keeps the exact B0
tile/overlap/threshold contract, fuses one vertical strip at a time, and
persists only strip vector candidates plus the final canonical building
polygons.  It is intentionally separate from the historical Tongzhou runner:
the latter's full probability/mask/boundary GeoTIFF contract would exceed the
available E: volume for four Shenzhen mosaics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NESTED_REPO = PROJECT_ROOT / "outputs" / "urban-renewal-phase1-starter"
if str(NESTED_REPO) not in sys.path:
    sys.path.insert(0, str(NESTED_REPO))
if str(NESTED_REPO / "src") not in sys.path:
    sys.path.insert(0, str(NESTED_REPO / "src"))

# Importing the already accepted runner gives this path the same model factory,
# preprocessing, weighted fusion and checkpoint contract as B0.
from tools import infer_rsbuilding_static_geometry as base  # noqa: E402
from urban_monitor.geometry_stream import intersecting_tile_origins, iter_strip_ranges  # noqa: E402


METRIC_CRS = "EPSG:32650"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def strip_polygon_candidates(
    source: rasterio.DatasetReader,
    result: np.ndarray,
    valid: np.ndarray,
    core_start: int,
    core_end: int,
    transform_to_metric,
) -> list:
    binary = (result >= base.THRESHOLD) & valid
    if not binary.any():
        return []
    window_transform = source.window_transform(Window(0, core_start, source.width, core_end - core_start))
    out = []
    for geom_json, value in shapes(binary.astype(np.uint8), mask=binary, transform=window_transform):
        if int(value) != 1:
            continue
        geom = shapely_transform(transform_to_metric, shape(geom_json))
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        out.append(geom)
    return out


def dissolve_polygons(polygons: list) -> list:
    if not polygons:
        return []
    # A strip can contain many adjacent fragments.  Dissolve in bounded
    # chunks, then once more globally; this prevents a single giant temporary
    # object from exhausting memory while preserving overlap de-duplication.
    chunks = []
    chunk_size = 50000
    for start in range(0, len(polygons), chunk_size):
        chunks.append(unary_union(polygons[start : start + chunk_size]))
    merged = unary_union(chunks)
    if merged.geom_type == "Polygon":
        return [merged]
    if merged.geom_type == "MultiPolygon":
        return list(merged.geoms)
    return [part for part in getattr(merged, "geoms", []) if part.geom_type in {"Polygon", "MultiPolygon"}]


def scene_manifest(root: Path) -> dict:
    for name in ("SHENZHEN_SCENES.json", "TONGZHOU_SCENES.json", "SCENES.json"):
        path = root / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("scene_discovery_status") == "pass":
                return payload
    raise FileNotFoundError(f"no pass scene manifest under {root}")


def process_scene(
    model,
    scene: dict,
    output_root: Path,
    checkpoint: Path,
    config: Path,
    device: str,
    batch_size: int,
    strip_rows: int,
) -> dict:
    scene_date = scene["scene_date"]
    output = output_root / scene_date
    manifest_path = output / "SPARSE_GEOMETRY_MANIFEST.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "success":
            print(f"REUSED completed sparse scene {scene_date}: {output}", flush=True)
            return existing
    output.mkdir(parents=True, exist_ok=True)
    strip_dir = output / "strips"
    strip_dir.mkdir(parents=True, exist_ok=True)
    state_path = output / "_SPARSE_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
        "status": "running",
        "scene_date": scene_date,
        "source_path": scene["selected_source"],
        "source_sha256": scene.get("source_sha256"),
        "checkpoint_sha256": base.EXPECTED_CHECKPOINT_SHA256,
        "strip_rows": strip_rows,
        "batch_size": batch_size,
        "completed_strips": [],
        "created_at": now(),
    }
    if state.get("strip_rows") != strip_rows or state.get("source_sha256") != scene.get("source_sha256"):
        raise RuntimeError(f"unsafe sparse resume metadata mismatch: {state_path}")
    completed = {int(v) for v in state.get("completed_strips", [])}
    started = time.perf_counter()
    strip_stats = []
    with rasterio.open(scene["selected_source"]) as source:
        transformer = Transformer.from_crs(source.crs, METRIC_CRS, always_xy=True)
        to_metric = transformer.transform
        xs = base.tile_origins(source.width)
        ys = base.tile_origins(source.height)
        strips = list(iter_strip_ranges(source.height, strip_rows))
        expected_tiles = len(xs) * len(ys)
        weight = base.blending_weight()
        for strip_index, (core_start, core_end) in enumerate(strips):
            if core_start in completed and (strip_dir / f"strip_{strip_index:05d}.geojson").is_file():
                continue
            strip_ys = intersecting_tile_origins(ys, core_start, core_end, base.TILE_SIZE)
            acc_start = max(0, min(strip_ys))
            acc_end = min(source.height, max(strip_ys) + base.TILE_SIZE)
            accumulator_rows = acc_end - acc_start
            probability_sum = np.zeros((accumulator_rows, source.width), dtype=np.float32)
            weight_sum = np.zeros((accumulator_rows, source.width), dtype=np.float32)
            jobs = [(x, y) for y in strip_ys for x in xs]
            for batch_start in range(0, len(jobs), batch_size):
                batch = jobs[batch_start : batch_start + batch_size]
                images = []
                masks = []
                for x, y in batch:
                    image, valid = base.read_tile(source, x, y)
                    images.append(image)
                    masks.append(valid)
                probabilities = base.predict_batch(model, images, device)
                for (x, y), probability, valid in zip(batch, probabilities, masks):
                    row0 = y - acc_start
                    row1 = min(row0 + base.TILE_SIZE, accumulator_rows)
                    col1 = min(x + base.TILE_SIZE, source.width)
                    tile_h = row1 - row0
                    tile_w = col1 - x
                    local_weight = weight[:tile_h, :tile_w] * valid[:tile_h, :tile_w].astype(np.float32)
                    probability_sum[row0:row1, x:col1] += probability[:tile_h, :tile_w] * local_weight
                    weight_sum[row0:row1, x:col1] += local_weight
            core = slice(core_start - acc_start, core_end - acc_start)
            weights = weight_sum[core, :]
            source_valid = source.dataset_mask(window=Window(0, core_start, source.width, core_end - core_start)) > 0
            valid = (weights > 0) & source_valid
            result = np.full(weights.shape, np.nan, dtype=np.float32)
            # Avoid boolean advanced-index copies here.  A 2048-row strip over
            # the Shenzhen raster contains hundreds of millions of pixels;
            # ``array[valid]`` materializes another ~1.7 GiB temporary and can
            # fail even while the GPU is healthy.  ``where`` keeps division
            # and clipping bounded to the already allocated strip buffers.
            np.divide(probability_sum[core, :], weights, out=result, where=valid)
            np.clip(result, 0.0, 1.0, out=result)
            polygons = strip_polygon_candidates(source, result, valid, core_start, core_end, to_metric)
            # Persist a strip-level vector artifact before advancing state so a
            # crash can resume without recomputing completed strips.
            geoms = dissolve_polygons(polygons)
            rows = []
            for geom in geoms:
                area = float(geom.area)
                if area < 10.0:
                    continue
                rows.append({"strip_index": strip_index, "area_m2": area, "geometry": geom})
            # A valid strip can contain no building candidates (for example,
            # water or an empty margin).  Constructing a GeoDataFrame from an
            # empty list with ``geometry="geometry"`` raises because the
            # column does not yet exist.  Keep the strip artifact explicit and
            # schema-compatible instead of treating an empty strip as a
            # processing failure.
            if rows:
                frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC_CRS)
            else:
                frame = gpd.GeoDataFrame(
                    {
                        "strip_index": np.array([], dtype=np.int64),
                        "area_m2": np.array([], dtype=np.float64),
                    },
                    geometry=gpd.GeoSeries([], dtype="geometry", crs=METRIC_CRS),
                    crs=METRIC_CRS,
                )
            strip_path = strip_dir / f"strip_{strip_index:05d}.geojson"
            frame.to_file(strip_path, driver="GeoJSON")
            receipt = {
                "status": "complete",
                "scene_date": scene_date,
                "strip_index": strip_index,
                "core_rows": [core_start, core_end],
                "tile_count": len(jobs),
                "object_fragment_count": len(rows),
                # Invalid pixels remain NaN, so a comparison naturally excludes
                # them without materialising the large ``result[valid]`` copy.
                "positive_fraction": (
                    float(np.count_nonzero(result >= base.THRESHOLD) / np.count_nonzero(valid))
                    if np.count_nonzero(valid)
                    else 0.0
                ),
                "source_sha256": scene.get("source_sha256"),
                "checkpoint_sha256": base.EXPECTED_CHECKPOINT_SHA256,
                "scientific_status": "candidate_not_ground_truth",
                "created_at": now(),
            }
            write_json(strip_dir / f"STRIP_RECEIPT_{strip_index:05d}.json", receipt)
            completed.add(core_start)
            state.update({"status": "running", "completed_strips": sorted(completed), "updated_at": now()})
            write_json(state_path, state)
            strip_stats.append(receipt)
            print(f"{scene_date} sparse strips {len(completed)}/{len(strips)}", flush=True)
            del probability_sum, weight_sum, result

    strip_files = sorted(strip_dir.glob("strip_*.geojson"))
    # A global unary_union over the multi-gigabyte strip collection can exceed
    # GEOS addressable memory even though each strip was already dissolved and
    # owns a disjoint half-open core row range.  Stream the canonical objects
    # directly from strips instead.  This preserves strip ownership and avoids
    # materialising all geometries in one Python/GEOS object graph.
    gpkg = output / "building_objects.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    object_count = 0
    wrote_layer = False
    for path in strip_files:
        frame = gpd.read_file(path)
        if frame.empty:
            continue
        frame = frame[["geometry"]].copy()
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
        if frame.empty:
            continue
        frame["object_id"] = np.arange(object_count + 1, object_count + 1 + len(frame), dtype=np.int64)
        frame["scene_date"] = scene_date
        frame["state"] = "building_candidate"
        frame["area_m2"] = frame.geometry.area.astype(float)
        frame["perimeter_m"] = frame.geometry.length.astype(float)
        frame["source_model"] = base.SOURCE_MODEL
        frame["scientific_status"] = "candidate_not_ground_truth"
        frame = gpd.GeoDataFrame(frame, geometry="geometry", crs=METRIC_CRS)
        pyogrio.write_dataframe(
            frame,
            gpkg,
            layer="buildings",
            driver="GPKG",
            append=wrote_layer,
        )
        object_count += len(frame)
        wrote_layer = True
    if not wrote_layer:
        empty = gpd.GeoDataFrame(
            {
                "object_id": np.array([], dtype=np.int64),
                "scene_date": [],
                "state": [],
                "area_m2": np.array([], dtype=float),
                "perimeter_m": np.array([], dtype=float),
                "source_model": [],
                "scientific_status": [],
                "geometry": [],
            },
            geometry="geometry",
            crs=METRIC_CRS,
        )
        pyogrio.write_dataframe(empty, gpkg, layer="buildings", driver="GPKG")
    positive_values = [float(item.get("positive_fraction", 0.0)) for item in strip_stats]
    manifest = {
        "status": "success",
        "scene_date": scene_date,
        "source_path": scene["selected_source"],
        "source_sha256": scene.get("source_sha256"),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": base.sha256_file(checkpoint),
        "config_path": str(config),
        "source_model": base.SOURCE_MODEL,
        "threshold": base.THRESHOLD,
        "tile_size": base.TILE_SIZE,
        "overlap": base.OVERLAP,
        "stride": base.STRIDE,
        "batch_size": batch_size,
        "strip_rows": strip_rows,
        "storage_mode": "sparse_vector_only",
        "probability_raster_materialized": False,
        "mask_raster_materialized": False,
        "boundary_raster_materialized": False,
        "positive_fraction_mean_by_strip": float(np.mean(positive_values)) if positive_values else None,
        "object_count": object_count,
        "global_dissolve": "skipped_streaming_strip_owned_cores",
        "cross_strip_duplicate_protection": "half_open_core_row_ownership",
        "crs": METRIC_CRS,
        "runtime_seconds": time.perf_counter() - started,
        "completed_strips": len(strip_files),
        "expected_strips": len(strips),
        "expected_tiles": expected_tiles,
        "outputs": {"building_objects": str(gpkg), "strip_directory": str(strip_dir)},
        "scientific_status": "candidate_not_ground_truth",
        "test_reads": {"WHU": 0, "SECOND": 0},
        "completed_at": now(),
    }
    write_json(manifest_path, manifest)
    state.update({"status": "success", "completed_strips": len(strip_files), "completed_at": now()})
    write_json(state_path, state)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--strip-rows", type=int, default=2048)
    args = parser.parse_args()
    if args.batch_size < 1 or args.strip_rows < 1:
        raise SystemExit("batch-size and strip-rows must be positive")
    scenes = scene_manifest(args.scene_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name in ("SHENZHEN_SCENES.json", "TONGZHOU_SCENES.json"):
        source = args.scene_root / name
        target = args.output_root / name
        if source.is_file() and not target.exists():
            target.write_bytes(source.read_bytes())
    model = base.build_model(args.config.resolve(), args.checkpoint.resolve(), args.device)
    results = []
    for scene in scenes["scenes"]:
        try:
            results.append(process_scene(model, scene, args.output_root, args.checkpoint.resolve(), args.config.resolve(), args.device, args.batch_size, args.strip_rows))
        except Exception as exc:
            # Keep a machine-readable failure receipt beside resumable strip
            # artifacts before propagating the original traceback to stderr.
            failure = {
                "schema_version": "urrs.shenzhen.sparse_geometry_failure.v1",
                "status": "FAILURE_RECEIPT",
                "scene_date": scene.get("scene_date"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "resume_safe": True,
                "raw_tifs_modified": False,
                "test_reads": {"WHU": 0, "SECOND": 0},
                "scientific_status": "candidate_not_ground_truth",
                "created_at": now(),
            }
            write_json(args.output_root / f"SPARSE_GEOMETRY_FAILURE_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", failure)
            raise
    payload = {
        "schema_version": "urrs.shenzhen.sparse_geometry_run.v1",
        "status": "success",
        "scenes": results,
        "storage_mode": "sparse_vector_only",
        "raw_tifs_modified": False,
        "test_reads": {"WHU": 0, "SECOND": 0},
        "scientific_status": "candidate_not_ground_truth",
        "created_at": now(),
    }
    write_json(args.output_root / "SHENZHEN_SPARSE_GEOMETRY_RUN.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
