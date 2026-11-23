"""Build exact, source-grid-aligned Xi'an three-state temporal products.

This post-processing tool consumes immutable sparse Building Foundation
vectors and the three read-only mosaics under ``E:\\xian\\xian``.  It does not
train or alter a model.  Matching is deterministic and deliberately
conservative: mutual nearest-centroid matches are accepted only when their
metric distance and area ratio are within the frozen contract.  All geometry
areas are measured after clipping to the supplied Xi'an boundary in EPSG:32649.

The GeoTIFF state layers retain the exact width/height/CRS/transform of the
source mosaic.  Rasterisation is windowed so the 15.6-billion-pixel source
grid is never materialised in RAM.  Raster outputs are candidate evidence,
not Ground Truth; date reliability and mixed-date exclusions are written to
the receipts and tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

METRIC_CRS = "EPSG:32649"
SOURCE_CRS_FALLBACK = "EPSG:4326"
DATE_RE = re.compile(r"(?<!\d)(\d{4})[-/]?(\d{2})[-/]?(\d{2})(?!\d)")
DEFAULT_GATE_M = 35.0
DEFAULT_MIN_AREA_RATIO = 0.10
DEFAULT_MIN_IOU = 0.05
DEFAULT_BLOCK = 512


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_date(value: Any) -> str | None:
    match = DATE_RE.search(str(value)) if value is not None else None
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError:
        return None


@dataclass(frozen=True)
class Epoch:
    key: str
    nominal_year: int
    source: Path
    geometry: Path
    nominal_date: str | None
    actual_date: str | None
    date_status: str
    date_counts: dict[str, int]
    source_sha256: str


def read_date_audit(root: Path) -> dict[str, dict[str, Any]]:
    """Read actual-date audit rows without using nominal years as evidence."""

    path = root / "date_audit" / "tables" / "epoch_summary.csv"
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = str(row.get("task_name") or "").strip()
            if not key:
                continue
            def as_int(name: str) -> int:
                try:
                    return int(float(row.get(name) or 0))
                except (TypeError, ValueError):
                    return 0
            rows[key] = {
                "nominal_date": parse_date(row.get("target_input_date")),
                "actual_date": parse_date(row.get("dominant_actual_date")),
                "actual_date_count": as_int("actual_date_count"),
                "known_date_coverage_pct": float(row.get("known_date_coverage_pct") or 0),
                "missing_tile_count": as_int("missing_tile_count"),
                "date_span_days": as_int("date_span_days"),
                "mixing_severity": str(row.get("mixing_severity") or "unknown"),
                "dominant_area_share_pct": float(row.get("dominant_area_share_pct") or 0),
            }
    detail = root / "date_audit" / "tables" / "date_summary_by_epoch.csv"
    if detail.is_file():
        with detail.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = str(row.get("task_name") or "").strip()
                actual = parse_date(row.get("actual_capture_date"))
                if not key or actual is None:
                    continue
                try:
                    count = int(float(row.get("tile_count") or 0))
                except (TypeError, ValueError):
                    count = 0
                rows.setdefault(key, {}).setdefault("date_counts", {})[actual] = count
    for value in rows.values():
        value.setdefault("date_counts", {})
    return rows


def load_epoch(args: argparse.Namespace, key: str, year: int, source: Path, geometry: Path) -> Epoch:
    audit = read_date_audit(args.source_root).get(key, {})
    actual = audit.get("actual_date") or parse_date(key)
    span = int(audit.get("date_span_days") or 0)
    mixing = str(audit.get("mixing_severity") or "unknown")
    coverage = float(audit.get("known_date_coverage_pct") or 0)
    missing = int(audit.get("missing_tile_count") or 0)
    if audit.get("actual_date_count", 0) <= 1 and span == 0 and missing == 0:
        status = "VALID_TEMPORAL"
    elif actual and span <= 31 and coverage >= 99.0 and missing == 0:
        status = "OFF_DATE_VALID"
    elif actual and mixing in {"severe", "moderate"}:
        status = "TEMPORAL_INSUFFICIENT"
    else:
        status = "TEMPORAL_INSUFFICIENT"
    return Epoch(
        key=key,
        nominal_year=year,
        source=source.resolve(),
        geometry=geometry.resolve(),
        nominal_date=audit.get("nominal_date"),
        actual_date=actual,
        date_status=status,
        date_counts=dict(audit.get("date_counts") or {}),
        source_sha256=sha256_file(source),
    )


def load_frame(path: Path):
    import geopandas as gpd
    import pyogrio

    try:
        frame = pyogrio.read_dataframe(path, layer="buildings", use_arrow=True)
    except RuntimeError as exc:
        # The cloud runtime may not ship pyarrow.  Keep the same GDAL/OGR
        # reader and fall back to the non-Arrow path rather than changing the
        # geometry contract or silently dropping the epoch.
        if "pyarrow required" not in str(exc).lower():
            raise
        frame = pyogrio.read_dataframe(path, layer="buildings", use_arrow=False)
    if frame.empty:
        raise RuntimeError(f"empty building vector: {path}")
    if frame.crs is None:
        raise RuntimeError(f"building vector has no CRS: {path}")
    frame = frame.to_crs(METRIC_CRS)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    frame.geometry = frame.geometry.make_valid()
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if "object_id" not in frame.columns:
        frame["object_id"] = [f"obj_{index:08d}" for index in range(len(frame))]
    frame["object_id"] = frame["object_id"].astype(str)
    frame["area_m2"] = frame.geometry.area.astype(float)
    frame["centroid_x"] = frame.geometry.centroid.x.astype(float)
    frame["centroid_y"] = frame.geometry.centroid.y.astype(float)
    frame = frame.reset_index(drop=True)
    return frame[["object_id", "geometry", "area_m2", "centroid_x", "centroid_y"]]


def pair_frames(left, right, gate_m: float, min_area_ratio: float, min_iou: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return deterministic mutual nearest matches and diagnostics."""

    from scipy.spatial import cKDTree
    from shapely import intersection

    a_xy = left[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    b_xy = right[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    tree_a = cKDTree(a_xy)
    tree_b = cKDTree(b_xy)
    d_ab, i_ab = tree_b.query(a_xy, k=1, distance_upper_bound=gate_m)
    d_ba, i_ba = tree_a.query(b_xy, k=1, distance_upper_bound=gate_m)
    match_a = np.full(len(left), -1, dtype=np.int64)
    candidate = np.isfinite(d_ab) & (i_ab < len(right))
    if candidate.any():
        idx_a = np.flatnonzero(candidate)
        idx_b = i_ab[candidate].astype(np.int64)
        mutual = i_ba[idx_b] == idx_a
        idx_a, idx_b = idx_a[mutual], idx_b[mutual]
        if len(idx_a):
            area_a = left.iloc[idx_a]["area_m2"].to_numpy(dtype=float)
            area_b = right.iloc[idx_b]["area_m2"].to_numpy(dtype=float)
            ratio = np.minimum(area_a, area_b) / np.maximum(area_a, area_b)
            geom_a = left.geometry.iloc[idx_a].to_numpy(dtype=object)
            geom_b = right.geometry.iloc[idx_b].to_numpy(dtype=object)
            try:
                inter = intersection(geom_a, geom_b)
                inter_area = np.asarray([float(g.area) if g is not None and not g.is_empty else 0.0 for g in inter])
            except Exception:
                inter_area = np.zeros(len(idx_a), dtype=float)
            union = np.maximum(area_a + area_b - inter_area, 1e-9)
            iou = inter_area / union
            keep = (ratio >= min_area_ratio) & ((iou >= min_iou) | ((d_ab[idx_a] <= 5.0) & (ratio >= 0.25)))
            match_a[idx_a[keep]] = idx_b[keep]
    matched = match_a >= 0
    reverse = np.full(len(right), -1, dtype=np.int64)
    reverse[match_a[matched]] = np.flatnonzero(matched)
    diagnostics = {
        "left_count": int(len(left)),
        "right_count": int(len(right)),
        "matched_count": int(matched.sum()),
        "match_rate_left": float(matched.mean()) if len(left) else 0.0,
        "match_rate_right": float((reverse >= 0).mean()) if len(right) else 0.0,
        "gate_m": gate_m,
        "min_area_ratio": min_area_ratio,
        "min_iou": min_iou,
        "distance_p50_m": float(np.nanpercentile(d_ab[candidate], 50)) if candidate.any() else None,
        "distance_p99_m": float(np.nanpercentile(d_ab[candidate], 99)) if candidate.any() else None,
        "candidate_count": int(candidate.sum()),
        "mutual_count": int((reverse >= 0).sum()),
    }
    return match_a, reverse, diagnostics


def clip_geometries(frame, city_union):
    """Clip only once in metric CRS and return a compact geometry frame."""

    import geopandas as gpd

    if frame.empty:
        return frame
    mask = frame.geometry.intersects(city_union)
    result = frame.loc[mask].copy()
    if result.empty:
        return result
    result.geometry = result.geometry.intersection(city_union)
    result = result[result.geometry.notna() & ~result.geometry.is_empty].copy()
    result["area_m2"] = result.geometry.area.astype(float)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=METRIC_CRS).reset_index(drop=True)


def geom_rows(frame, indices: Iterable[int], state: str, epoch: Epoch, extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra = extra or {}
    for index in indices:
        row = frame.iloc[int(index)]
        item = {
            "object_id": str(row.object_id),
            "epoch": epoch.key,
            "nominal_year": epoch.nominal_year,
            "actual_date": epoch.actual_date,
            "date_status": epoch.date_status,
            "state": state,
            "area_m2": float(row.area_m2),
            "geometry": row.geometry,
            "candidate_not_ground_truth": True,
        }
        item.update(extra)
        rows.append(item)
    return rows


def write_vector(path: Path, rows: Sequence[dict[str, Any]], crs: str = METRIC_CRS) -> None:
    import geopandas as gpd
    from shapely.geometry import GeometryCollection

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        gdf = gpd.GeoDataFrame({"state": [], "area_m2": []}, geometry=[], crs=crs)
    else:
        attrs = [{k: v for k, v in row.items() if k != "geometry"} for row in rows]
        gdf = gpd.GeoDataFrame(attrs, geometry=[row["geometry"] for row in rows], crs=crs)
    gdf.to_file(path, layer=path.stem[:50], driver="GPKG")


def write_parquet(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import geopandas as gpd

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        # GeoPackages are the canonical vector artifacts.  Keep the run
        # complete on workers without the optional Arrow dependency and leave
        # an explicit sidecar rather than a misleading .parquet file.
        write_json(path.with_suffix(path.suffix + ".unavailable.json"), {
            "status": "parquet_unavailable",
            "path": str(path),
            "reason": str(exc),
            "canonical_vector_artifact": str(path.with_suffix(".gpkg")),
        })
        return
    if not rows:
        gpd.GeoDataFrame({"state": [], "area_m2": []}, geometry=[], crs=METRIC_CRS).to_parquet(path, index=False)
        return
    attrs = [{k: v for k, v in row.items() if k != "geometry"} for row in rows]
    gpd.GeoDataFrame(attrs, geometry=[row["geometry"] for row in rows], crs=METRIC_CRS).to_parquet(path, index=False)


def source_grid(path: Path) -> dict[str, Any]:
    import rasterio

    with rasterio.open(path) as src:
        profile = src.profile.copy()
        return {
            "width": int(src.width),
            "height": int(src.height),
            "count": int(src.count),
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs or SOURCE_CRS_FALLBACK),
            "transform": list(src.transform)[:6],
            "bounds": [float(v) for v in src.bounds],
            "nodata": src.nodata,
            "overviews": src.overviews(1),
            "profile": profile,
        }


def _metric_window_bounds(src, window, to_metric=None):
    from pyproj import Transformer
    from rasterio.windows import bounds
    from shapely.geometry import box
    if to_metric is None:
        to_metric = Transformer.from_crs(src.crs or SOURCE_CRS_FALLBACK, METRIC_CRS, always_xy=True).transform
    left, bottom, right, top = bounds(window, src.transform)
    return box(*to_metric(left, bottom), *to_metric(right, top))


def rasterize_rows(path: Path, source_path: Path, rows: Sequence[dict[str, Any]], city_union, value: int, block: int = DEFAULT_BLOCK) -> dict[str, Any]:
    """Windowed rasterisation onto the exact source grid."""

    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window, transform as window_transform
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
    from shapely.strtree import STRtree

    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, tiled=True, blockxsize=block, blockysize=block, compress="DEFLATE", predictor=2, BIGTIFF="IF_SAFER", SPARSE_OK="TRUE")
        profile.pop("photometric", None)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Keep only non-empty geometries.  A metric STRtree avoids transforming
        # all polygons into lon/lat up front.
        geoms = [row["geometry"] for row in rows if row.get("geometry") is not None and not row["geometry"].is_empty]
        tree = STRtree(geoms) if geoms else None
        to_metric = Transformer.from_crs(src.crs or SOURCE_CRS_FALLBACK, METRIC_CRS, always_xy=True).transform
        to_source = Transformer.from_crs(METRIC_CRS, src.crs or SOURCE_CRS_FALLBACK, always_xy=True).transform
        tile_count = 0
        filled_pixels = 0
        with rasterio.open(path, "w", **profile) as dst:
            for row0 in range(0, src.height, block):
                for col0 in range(0, src.width, block):
                    h = min(block, src.height - row0)
                    w = min(block, src.width - col0)
                    window = Window(col0, row0, w, h)
                    tile_count += 1
                    if tree is None:
                        continue
                    query_box = _metric_window_bounds(src, window, to_metric)
                    indices = tree.query(query_box, predicate="intersects")
                    if len(indices) == 0:
                        continue
                    shapes = []
                    for index in np.asarray(indices).reshape(-1):
                        geom = geoms[int(index)]
                        if city_union is not None:
                            geom = geom.intersection(city_union)
                        if geom is None or geom.is_empty:
                            continue
                        geom = shp_transform(to_source, geom)
                        if not geom.is_empty:
                            shapes.append((geom, value))
                    if not shapes:
                        continue
                    arr = rasterize(shapes, out_shape=(int(h), int(w)), transform=window_transform(window, src.transform), fill=0, dtype="uint8", all_touched=False)
                    if arr.any():
                        dst.write(arr, 1, window=window)
                        filled_pixels += int((arr > 0).sum())
            # Internal overviews make QGIS quicklook responsive without
            # changing the full-resolution alignment contract.
            try:
                dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.nearest)
                dst.update_tags(ns="rio_overview", resampling="nearest")
            except Exception:
                pass
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "tile_count": tile_count, "filled_pixels": filled_pixels, "source_grid": source_grid(source_path)}


def save_map(source: Path, layers: dict[str, Sequence[dict[str, Any]]], city_union, out_path: Path, title: str, simple: bool = False) -> None:
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.enums import Resampling
    from matplotlib.patches import Patch

    colors = {"STABLE": "#2563eb", "NEW": "#16a34a", "DEMOLITION": "#dc2626", "REDEVELOPMENT_101": "#f59e0b"}
    with rasterio.open(source) as src:
        # Use an overview rather than reading multi-gigabyte source pixels.
        factor = max(1, (src.width // (2200 if not simple else 1400)))
        out_w, out_h = max(1, src.width // factor), max(1, src.height // factor)
        image = src.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=Resampling.average)
        bounds = src.bounds
        crs = src.crs
    display = np.moveaxis(image.astype(np.float32), 0, -1)
    for channel in range(3):
        vals = display[..., channel]
        lo, hi = np.percentile(vals[np.isfinite(vals)], [2, 98]) if np.isfinite(vals).any() else (0, 1)
        display[..., channel] = np.clip((vals - lo) / max(1e-6, hi - lo), 0, 1)
    fig, ax = plt.subplots(figsize=(16, 10) if not simple else (12, 8), dpi=220)
    ax.imshow(display, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], origin="upper")
    for state, rows in layers.items():
        if not rows:
            continue
        gdf = gpd.GeoDataFrame({"state": [state] * len(rows)}, geometry=[r["geometry"] for r in rows], crs=METRIC_CRS).to_crs(crs)
        gdf.boundary.plot(ax=ax, color=colors.get(state, "#111827"), linewidth=0.65, alpha=0.85)
        if not simple and len(gdf) < 4000:
            gdf.plot(ax=ax, facecolor=colors.get(state, "#111827"), edgecolor="none", alpha=0.22)
    if city_union is not None:
        gpd.GeoSeries([city_union], crs=METRIC_CRS).to_crs(crs).boundary.plot(ax=ax, color="white", linewidth=1.1, alpha=0.9)
    handles = [Patch(facecolor=color, edgecolor=color, alpha=0.55, label=label) for label, color in colors.items() if layers.get(label)]
    if handles:
        ax.legend(handles=handles, loc="upper right", title="candidate state")
    ax.set_title(title)
    ax.set_xlabel(str(crs))
    ax.set_ylabel(str(crs))
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def parse_spec(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must be KEY=PATH: {value}")
        key, raw = value.split("=", 1)
        result[key.strip()] = Path(raw.strip()).expanduser().resolve()
    return result


def parse_year_spec(values: Sequence[str], label: str) -> dict[str, int]:
    """Parse KEY=YYYY values without resolving the year as a filesystem path."""

    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must be KEY=YYYY: {value}")
        key, raw = value.split("=", 1)
        try:
            result[key.strip()] = int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"{label} must use an integer year: {value}") from exc
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="KEY=source GeoTIFF")
    parser.add_argument("--geometry", action="append", required=True, help="KEY=BUILDINGS.gpkg")
    parser.add_argument("--nominal-year", action="append", required=True, help="KEY=YYYY")
    parser.add_argument("--city-boundary", type=Path, required=True)
    parser.add_argument("--county-boundary", type=Path, default=None)
    parser.add_argument("--gate-m", type=float, default=DEFAULT_GATE_M)
    parser.add_argument("--min-area-ratio", type=float, default=DEFAULT_MIN_AREA_RATIO)
    parser.add_argument("--min-iou", type=float, default=DEFAULT_MIN_IOU)
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="skip RGB/basemap rendering",
    )
    parser.add_argument(
        "--skip-rasters",
        action="store_true",
        help="skip unfiltered internal state rasters; downstream direct area-filtered products rasterize only >=100 m2 vectors",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse PASS period raster/vector artifacts already present under output-root",
    )
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    sources = parse_spec(args.source, "--source")
    geometries = parse_spec(args.geometry, "--geometry")
    years = parse_year_spec(args.nominal_year, "--nominal-year")
    if set(sources) != set(geometries) or set(sources) != set(years):
        raise SystemExit("source, geometry and nominal-year keys must match")
    required = {"xa180429", "xa190101", "xa201231"}
    if not required.issubset(sources):
        raise SystemExit("the three direct epochs xa180429, xa190101 and xa201231 are required")

    import geopandas as gpd

    output = args.output_root
    for directory in ("00_provenance", "01_vectors", "02_rasters", "03_figures", "04_statistics", "05_qa", "logs", "receipts"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    city = gpd.read_file(args.city_boundary)
    if city.empty or city.crs is None:
        raise SystemExit("city boundary is empty or has no CRS")
    city_union = city.to_crs(METRIC_CRS).geometry.unary_union
    county_rows: list[dict[str, Any]] = []
    if args.county_boundary and args.county_boundary.is_file():
        county = gpd.read_file(args.county_boundary)
        if not county.empty and county.crs is not None:
            county = county.to_crs(METRIC_CRS)
            name_col = next((col for col in ("name", "NAME", "县", "县区", "区县") if col in county.columns), None)
            for idx, row in county.iterrows():
                county_rows.append({"name": str(row[name_col]) if name_col else f"county_{idx+1:02d}", "geometry": row.geometry})

    order = ["xa180429", "xa190101", "xa201231"]
    epochs = [load_epoch(args, key, years[key], sources[key], geometries[key]) for key in order]
    for epoch in epochs:
        if not epoch.source.is_file() or not epoch.geometry.is_file():
            raise SystemExit(f"missing source/geometry: {epoch.key} {epoch.source} {epoch.geometry}")

    provenance = {
        "created_at": utc_now(),
        "source_root": str(args.source_root),
        "source_grid_contract": "state GeoTIFFs retain each source width/height/CRS/transform",
        "metric_crs": METRIC_CRS,
        "city_boundary": str(args.city_boundary.resolve()),
        "city_boundary_sha256": sha256_file(args.city_boundary),
        "county_boundary": str(args.county_boundary.resolve()) if args.county_boundary else None,
        "epochs": [{
            "key": e.key, "nominal_year": e.nominal_year, "source": str(e.source), "source_sha256": e.source_sha256,
            "geometry": str(e.geometry), "nominal_date": e.nominal_date, "actual_date": e.actual_date,
            "date_status": e.date_status, "date_counts": e.date_counts, "grid": source_grid(e.source),
        } for e in epochs],
        "checkpoint_sha256_expected": "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7",
        "candidate_not_ground_truth": True,
        "raw_inputs_modified": False,
        "model_weights_modified": False,
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
    }
    write_json(output / "00_provenance" / "SOURCE_PROVENANCE.json", provenance)
    write_csv(output / "00_provenance" / "EPOCH_DATE_AUDIT.csv", [{
        "epoch": e.key, "nominal_year": e.nominal_year, "nominal_date": e.nominal_date, "actual_date": e.actual_date,
        "date_status": e.date_status, "date_counts": json.dumps(e.date_counts, ensure_ascii=False), "source_sha256": e.source_sha256,
    } for e in epochs])

    frames = []
    for epoch in epochs:
        started = time.time()
        frame = clip_geometries(load_frame(epoch.geometry), city_union)
        frame["epoch"] = epoch.key
        frames.append(frame)
        write_json(output / "receipts" / f"{epoch.key}_VECTOR_LOAD_RECEIPT.json", {"epoch": epoch.key, "source": str(epoch.geometry), "object_count_inside_city": int(len(frame)), "area_m2_inside_city": float(frame.area_m2.sum()), "elapsed_seconds": time.time() - started, "metric_crs": METRIC_CRS, "candidate_not_ground_truth": True})

    a, b, c = frames
    m_ab, rev_ab, diag_ab = pair_frames(a, b, args.gate_m, args.min_area_ratio, args.min_iou)
    m_bc, rev_bc, diag_bc = pair_frames(b, c, args.gate_m, args.min_area_ratio, args.min_iou)
    m_ac, rev_ac, diag_ac = pair_frames(a, c, args.gate_m, args.min_area_ratio, args.min_iou)
    write_json(output / "04_statistics" / "MATCH_DIAGNOSTICS.json", {"2018_to_2019": diag_ab, "2019_to_2020": diag_bc, "2018_to_2020": diag_ac, "contract": {"gate_m": args.gate_m, "min_area_ratio": args.min_area_ratio, "min_iou": args.min_iou}})

    rows: list[dict[str, Any]] = []
    pair_stats: list[dict[str, Any]] = []
    for left, right, match, el, er, label in ((a, b, m_ab, epochs[0], epochs[1], "2018_2019"), (b, c, m_bc, epochs[1], epochs[2], "2019_2020")):
        stable_idx = np.flatnonzero(match >= 0)
        removed_idx = np.flatnonzero(match < 0)
        matched_right = set(match[stable_idx].tolist())
        added_idx = np.array([idx for idx in range(len(right)) if idx not in matched_right], dtype=np.int64)
        stable_rows = geom_rows(right, stable_idx * 0 + match[stable_idx], "STABLE", er, {"left_object_id": [str(left.iloc[int(i)].object_id) for i in stable_idx]}) if False else []
        # Keep one record per polygon; the destination geometry is used for
        # stable evidence and the source geometry for demolition evidence.
        stable_rows = []
        for i in stable_idx:
            j = int(match[int(i)])
            rr = right.iloc[j]
            stable_rows.append({"object_id": str(rr.object_id), "epoch": er.key, "nominal_year": er.nominal_year, "actual_date": er.actual_date, "date_status": er.date_status, "state": "STABLE", "area_m2": float(rr.area_m2), "geometry": rr.geometry, "left_object_id": str(left.iloc[int(i)].object_id), "candidate_not_ground_truth": True})
        new_rows = geom_rows(right, added_idx, "NEW", er)
        demolition_rows = geom_rows(left, removed_idx, "DEMOLITION", el)
        rows.extend(stable_rows + new_rows + demolition_rows)
        pair_stats.extend([
            {"pair": label, "state": "STABLE", "count": len(stable_rows), "area_m2": sum(r["area_m2"] for r in stable_rows), "date_status": f"{el.date_status}|{er.date_status}", "strict_change_eligible": el.date_status in {"VALID_TEMPORAL", "OFF_DATE_VALID"} and er.date_status in {"VALID_TEMPORAL", "OFF_DATE_VALID"}},
            {"pair": label, "state": "NEW", "count": len(new_rows), "area_m2": sum(r["area_m2"] for r in new_rows), "date_status": f"{el.date_status}|{er.date_status}", "strict_change_eligible": el.date_status in {"VALID_TEMPORAL", "OFF_DATE_VALID"} and er.date_status in {"VALID_TEMPORAL", "OFF_DATE_VALID"}},
            {"pair": label, "state": "DEMOLITION", "count": len(demolition_rows), "area_m2": sum(r["area_m2"] for r in demolition_rows), "date_status": f"{el.date_status}|{er.date_status}", "strict_change_eligible": el.date_status in {"VALID_TEMPORAL", "OFF_DATE_VALID"} and er.date_status in {"VALID_TEMPORAL", "OFF_DATE_VALID"}},
        ])
        period_artifacts_pass = args.resume and all(
            (output / "receipts" / f"XIAN_{label}_{state}_RASTER_RECEIPT.json").is_file()
            and (output / "02_rasters" / f"XIAN_{label}_{state}.tif").is_file()
            for state in ("STABLE", "NEW", "DEMOLITION")
        )
        if not period_artifacts_pass:
            write_vector(output / "01_vectors" / f"{label}_STABLE.gpkg", stable_rows)
            write_vector(output / "01_vectors" / f"{label}_NEW.gpkg", new_rows)
            write_vector(output / "01_vectors" / f"{label}_DEMOLITION.gpkg", demolition_rows)
            write_parquet(output / "01_vectors" / f"{label}_STATES.parquet", stable_rows + new_rows + demolition_rows)
        grid = source_grid(el.source)
        for state_name, state_rows in (("STABLE", stable_rows), ("NEW", new_rows), ("DEMOLITION", demolition_rows)):
            # Each state-specific product is a binary mask (1=state, 0=not
            # state).  A separate combined state-code layer can be rebuilt
            # from these masks without making the per-state QGIS layers
            # ambiguous.
            if not period_artifacts_pass and not args.skip_rasters:
                info = rasterize_rows(output / "02_rasters" / f"XIAN_{label}_{state_name}.tif", el.source, state_rows, city_union, 1, args.block)
                write_json(output / "receipts" / f"XIAN_{label}_{state_name}_RASTER_RECEIPT.json", {"pair": label, "state": state_name, "value": 1, "source_grid": grid, "raster": info, "candidate_not_ground_truth": True, "qgis_alignment": {"exact_source_width_height_crs_transform": True}})
        if not args.skip_figures:
            save_map(el.source, {"STABLE": stable_rows, "NEW": new_rows, "DEMOLITION": demolition_rows}, city_union, output / "03_figures" / f"XIAN_{label}_THREE_STATE_HIGHRES.png", f"Xi'an {label.replace('_', ' → ')} | three-state candidate evidence")

    # 1-0-1: old A is absent in B but a mutual A-C match exists and the C
    # object is not already occupied by a B-C match.  This preserves separate
    # physical IDs while assigning one redevelopment episode.
    c_from_b = set(m_bc[m_bc >= 0].tolist())
    redevelop: list[dict[str, Any]] = []
    for i in np.flatnonzero(m_ab < 0):
        j = int(m_ac[int(i)])
        if j < 0 or j in c_from_b:
            continue
        old = a.iloc[int(i)]
        new = c.iloc[j]
        redevelop.append({
            "old_object_id": str(old.object_id), "new_object_id": str(new.object_id), "old_epoch": epochs[0].key, "new_epoch": epochs[2].key,
            "old_track_id": f"{epochs[0].key}_{old.object_id}", "new_track_id": f"{epochs[2].key}_{new.object_id}",
            "episode_id": f"redevelopment_{epochs[0].key}_{old.object_id}_{epochs[2].key}_{new.object_id}",
            "sequence": "1-0-1", "old_area_m2": float(old.area_m2), "new_area_m2": float(new.area_m2),
            "geometry_old": old.geometry, "geometry_new": new.geometry, "actual_dates": ";".join(str(e.actual_date) for e in epochs),
            "date_status": ";".join(e.date_status for e in epochs), "candidate_not_ground_truth": True,
        })
    redevelopment_rows: list[dict[str, Any]] = []
    for item in redevelop:
        for side, geom, area in (("OLD", item["geometry_old"], item["old_area_m2"]), ("NEW", item["geometry_new"], item["new_area_m2"])):
            redevelopment_rows.append({"object_id": item[f"{side.lower()}_object_id"], "epoch": item["old_epoch"] if side == "OLD" else item["new_epoch"], "actual_date": epochs[0].actual_date if side == "OLD" else epochs[2].actual_date, "state": "REDEVELOPMENT_101", "side": side, "area_m2": area, "episode_id": item["episode_id"], "old_track_id": item["old_track_id"], "new_track_id": item["new_track_id"], "sequence": "1-0-1", "geometry": geom, "candidate_not_ground_truth": True})
    write_vector(output / "01_vectors" / "XIAN_1_0_1_REDEVELOPMENT.gpkg", redevelopment_rows)
    write_parquet(output / "01_vectors" / "XIAN_1_0_1_REDEVELOPMENT.parquet", redevelopment_rows)
    if not args.skip_rasters:
        for side, side_rows in (("OLD", [r for r in redevelopment_rows if r["side"] == "OLD"]), ("NEW", [r for r in redevelopment_rows if r["side"] == "NEW"])):
            info = rasterize_rows(output / "02_rasters" / f"XIAN_1_0_1_REDEVELOPMENT_{side}.tif", epochs[0].source, side_rows, city_union, 1, args.block)
            write_json(output / "receipts" / f"XIAN_1_0_1_{side}_RASTER_RECEIPT.json", {"raster": info, "value": 1, "sequence": "1-0-1", "candidate_not_ground_truth": True, "qgis_alignment": {"exact_source_width_height_crs_transform": True}})
    if not args.skip_figures:
        save_map(epochs[0].source, {"REDEVELOPMENT_101": redevelopment_rows}, city_union, output / "03_figures" / "XIAN_1_0_1_REDEVELOPMENT_HIGHRES.png", "Xi'an 1-0-1 redevelopment candidate | actual imagery")

    # Three-epoch states: stable means matched in both adjacent intervals;
    # demolition/new are end-to-end presence changes, excluding 1-0-1 from
    # the simple demolition bucket to avoid double counting.
    stable_3: list[dict[str, Any]] = []
    demolition_3: list[dict[str, Any]] = []
    new_3: list[dict[str, Any]] = []
    redevelop_old = {r["old_object_id"] for r in redevelop}
    redevelop_new = {r["new_object_id"] for r in redevelop}
    for i in range(len(a)):
        j = int(m_ab[i])
        if j >= 0 and int(m_bc[j]) >= 0:
            rr = b.iloc[j]
            stable_3.append({"object_id": str(rr.object_id), "state": "STABLE", "area_m2": float(rr.area_m2), "geometry": rr.geometry, "sequence": "1-1-1", "candidate_not_ground_truth": True})
        elif str(a.iloc[i].object_id) in redevelop_old:
            continue
        elif j < 0:
            demolition_3.append({"object_id": str(a.iloc[i].object_id), "state": "DEMOLITION", "area_m2": float(a.iloc[i].area_m2), "geometry": a.iloc[i].geometry, "sequence": "1-0-0", "candidate_not_ground_truth": True})
    matched_a_to_c = set(m_ac[m_ac >= 0].tolist())
    for j in range(len(c)):
        if str(c.iloc[j].object_id) in redevelop_new:
            continue
        if int(rev_ac[j]) < 0:
            new_3.append({"object_id": str(c.iloc[j].object_id), "state": "NEW", "area_m2": float(c.iloc[j].area_m2), "geometry": c.iloc[j].geometry, "sequence": "0-0-1", "candidate_not_ground_truth": True})
    three_rows = stable_3 + new_3 + demolition_3
    write_vector(output / "01_vectors" / "XIAN_THREE_EPOCH_STATES.gpkg", three_rows)
    write_parquet(output / "01_vectors" / "XIAN_THREE_EPOCH_STATES.parquet", three_rows)
    if not args.skip_rasters:
        for state_name, state_rows in (("STABLE", stable_3), ("NEW", new_3), ("DEMOLITION", demolition_3)):
            info = rasterize_rows(output / "02_rasters" / f"XIAN_THREE_EPOCH_{state_name}.tif", epochs[0].source, state_rows, city_union, 1, args.block)
            write_json(output / "receipts" / f"XIAN_THREE_EPOCH_{state_name}_RASTER_RECEIPT.json", {"raster": info, "state": state_name, "value": 1, "candidate_not_ground_truth": True, "qgis_alignment": {"exact_source_width_height_crs_transform": True}})

    # County statistics are vector-derived, so areas are metric and clipped;
    # a polygon may contribute to a county only once.
    stats_rows = pair_stats + [{"pair": "THREE_EPOCH", "state": name, "count": (len(redevelop) if name == "REDEVELOPMENT_101" else len(items)), "area_m2": sum(r["area_m2"] for r in items), "date_status": ";".join(e.date_status for e in epochs), "strict_change_eligible": all(e.date_status in {"VALID_TEMPORAL", "OFF_DATE_VALID"} for e in epochs)} for name, items in (("STABLE", stable_3), ("NEW", new_3), ("DEMOLITION", demolition_3), ("REDEVELOPMENT_101", redevelopment_rows))]
    write_csv(output / "04_statistics" / "XIAN_THREE_STATE_STATS.csv", stats_rows, ["pair", "state", "count", "area_m2", "date_status", "strict_change_eligible"])
    county_stats: list[dict[str, Any]] = []
    for county in county_rows:
        for pair, state, items in [("THREE_EPOCH", "STABLE", stable_3), ("THREE_EPOCH", "NEW", new_3), ("THREE_EPOCH", "DEMOLITION", demolition_3), ("THREE_EPOCH", "REDEVELOPMENT_101", redevelopment_rows)]:
            area = 0.0
            count = 0
            if state == "REDEVELOPMENT_101":
                # Count one episode per county even though the vector layer
                # stores separate OLD and NEW footprint rows.
                for episode in redevelop:
                    hit = False
                    for geom in (episode.get("geometry_old"), episode.get("geometry_new")):
                        if geom is None or geom.is_empty:
                            continue
                        clipped = geom.intersection(county["geometry"])
                        if not clipped.is_empty:
                            area += float(clipped.area)
                            hit = True
                    count += int(hit)
            else:
                for item in items:
                    geom = item.get("geometry")
                    if geom is None or geom.is_empty:
                        continue
                    clipped = geom.intersection(county["geometry"])
                    if not clipped.is_empty:
                        area += float(clipped.area)
                        count += 1
            county_stats.append({"county": county["name"], "pair": pair, "state": state, "count": count, "area_m2": area, "area_km2": area / 1e6, "candidate_not_ground_truth": True})
    write_csv(output / "04_statistics" / "XIAN_COUNTY_THREE_STATE_STATS.csv", county_stats)
    write_csv(output / "04_statistics" / "XIAN_PAIR_STATS.csv", pair_stats)
    write_json(output / "04_statistics" / "TEMPORAL_RELIABILITY.json", {"epochs": [{"epoch": e.key, "actual_date": e.actual_date, "status": e.date_status, "date_counts": e.date_counts} for e in epochs], "policy": "actual acquisition dates only; duplicates folded; severe mixed-date epoch is temporal-insufficient; candidate changes are not Ground Truth"})
    qa = {
        "finite_geometry_areas": bool(all(math.isfinite(float(r["area_m2"])) and float(r["area_m2"]) > 0 for r in three_rows + redevelopment_rows)),
        "state_counts": {state: (len(redevelop) if state == "REDEVELOPMENT_101" else len(items)) for state, items in (("STABLE", stable_3), ("NEW", new_3), ("DEMOLITION", demolition_3), ("REDEVELOPMENT_101", redevelopment_rows))},
        "b0_checkpoint_sha256": provenance["checkpoint_sha256_expected"],
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        "raw_inputs_modified": False,
        "candidate_not_ground_truth": True,
    }
    write_json(output / "05_qa" / "STATIC_OUTPUT_QA.json", qa)
    receipt = {
        "schema_version": "urrs.xian.exact_three_state.v1",
        "status": "PASS" if qa["finite_geometry_areas"] else "FAIL",
        "run_root": str(output),
        "source_count": len(epochs),
        "geometry_counts_inside_city": {e.key: int(len(frames[i])) for i, e in enumerate(epochs)},
        "three_epoch_state_counts": qa["state_counts"],
        "redevelopment_101_count": len(redevelop),
        "date_reliability": [{"epoch": e.key, "actual_date": e.actual_date, "date_status": e.date_status, "date_counts": e.date_counts} for e in epochs],
        "match_diagnostics": {"2018_to_2019": diag_ab, "2019_to_2020": diag_bc, "2018_to_2020": diag_ac},
        "qgis_alignment_contract": ("unfiltered internal state rasters skipped; downstream >=100 m2 products are rasterized on the exact source grid" if args.skip_rasters else "every state TIFF copies source width/height/CRS/transform; metric areas come from EPSG:32649 vectors"),
        "unfiltered_internal_rasters": "SKIPPED_BY_REQUEST" if args.skip_rasters else "WRITTEN",
        "candidate_not_ground_truth": True,
        "strict_change_policy": "xa201231 is severe mixed-date, so 2019-2020 and three-epoch change fields remain provisional/temporal-insufficient until tile-level date evidence is spatially resolved",
        "test_reads": qa["test_reads"],
        "raw_inputs_modified": False,
        "model_weights_modified": False,
    }
    write_json(output / "receipts" / "FINAL_XIAN_EXACT_THREE_STATE_RECEIPT.json", receipt)
    source_summary = ", ".join(f"{e.key}={e.actual_date or 'UNKNOWN'}/{e.date_status}" for e in epochs)
    report = [
        "# Xi'an exact three-state temporal package",
        "",
        f"- Status: **{receipt['status']}** (candidate evidence, not Ground Truth)",
        f"- Sources: {source_summary}",
        f"- Three-epoch states: {qa['state_counts']}",
        f"- 1-0-1 redevelopment pairs: **{len(redevelop)}** (old/new physical IDs remain distinct)",
        "- All areas are clipped to the Xi'an city boundary and measured in EPSG:32649; county table is included when the supplied county file is valid.",
        ("- Unfiltered internal state TIFFs were skipped; downstream direct >=100 m2 products are source-grid aligned and can be loaded directly in QGIS." if args.skip_rasters else "- State TIFFs are source-grid aligned (same width/height/CRS/transform) and can be loaded directly in QGIS."),
        "- Actual acquisition dates are authoritative. xa190101 is off-date/mixed within 7 days; xa201231 is severely mixed over 348 days and is marked TEMPORAL_INSUFFICIENT for strict change claims.",
        "- No training, Test/Holdout reads, model-weight changes, raw-input changes, or registry changes were performed.",
    ]
    (output / "FINAL_XIAN_EXACT_THREE_STATE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
