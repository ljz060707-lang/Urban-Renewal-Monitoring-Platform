"""Build boundary-scoped Xi'an temporal accounting from immutable geometry.

This is a post-processing utility.  It reads existing sparse-B0 geometry
packages and date-aware source inventory, clips every accounting geometry to
the supplied Xi'an city/county boundaries in EPSG:32649, and writes candidate
epoch/change tables plus reproducible overview figures.  It never trains,
changes a checkpoint, or treats a nominal GEID year as an acquisition date.
The output remains ``candidate_not_ground_truth`` until independent review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

METRIC_CRS = "EPSG:32649"
DATE_RE = re.compile(r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)")
DEFAULT_CITY = Path(r"C:\Users\ljz52\Desktop\西安市_市.geojson")
DEFAULT_COUNTIES = Path(r"C:\Users\ljz52\Desktop\西安市_县.geojson")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def parse_iso_date(value: Any) -> str | None:
    if value is None:
        return None
    match = DATE_RE.search(str(value))
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError:
        return None


def classify_temporal_pair(
    actual_a: str | None,
    actual_b: str | None,
    nominal_year_a: int | None,
    nominal_year_b: int | None,
) -> dict[str, Any]:
    """Classify a pair using actual acquisition dates only.

    A distinct off-date pair is useful evidence but is explicitly labelled
    ``OFF_DATE_VALID``.  Equal dates are folded and cannot produce a change.
    Missing dates are insufficient rather than silently interpreted as zero.
    """

    a = parse_iso_date(actual_a)
    b = parse_iso_date(actual_b)
    if not a or not b:
        return {"status": "TEMPORAL_INSUFFICIENT", "actual_date_a": a, "actual_date_b": b, "genuine_interval": False}
    if a == b:
        return {"status": "DUPLICATE_EPOCH", "actual_date_a": a, "actual_date_b": b, "genuine_interval": False}
    in_nominal_years = bool(
        nominal_year_a is not None
        and nominal_year_b is not None
        and int(a[:4]) == int(nominal_year_a)
        and int(b[:4]) == int(nominal_year_b)
    )
    return {
        "status": "VALID_TEMPORAL" if in_nominal_years else "OFF_DATE_VALID",
        "actual_date_a": a,
        "actual_date_b": b,
        "genuine_interval": True,
        "ordered": a < b,
    }


def intersection_area_m2(geometries: Iterable[Any], boundary: Any) -> float:
    """Return exact metric intersection area for a bounded geometry iterable."""

    total = 0.0
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        intersection = geometry.intersection(boundary)
        if not intersection.is_empty:
            total += float(intersection.area)
    return total


def _load_boundaries(city_path: Path, county_path: Path) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    import geopandas as gpd

    city = gpd.read_file(city_path)
    counties = gpd.read_file(county_path)
    if city.empty:
        raise RuntimeError(f"city boundary has no features: {city_path}")
    if counties.empty:
        raise RuntimeError(f"county boundary has no features: {county_path}")
    if city.crs is None or counties.crs is None:
        raise RuntimeError("city/county boundary CRS is required")
    city_metric = city.to_crs(METRIC_CRS)
    county_metric = counties.to_crs(METRIC_CRS)
    city_union = city_metric.geometry.unary_union
    name_column = next((column for column in ("name", "NAME", "县", "县区") if column in county_metric.columns), None)
    county_rows: list[dict[str, Any]] = []
    for index, row in county_metric.iterrows():
        name = str(row[name_column]) if name_column and row[name_column] not in (None, "") else f"county_{index + 1:02d}"
        county_rows.append({"name": name, "geometry": row.geometry, "index": int(index)})
    audit = {
        "metric_crs": METRIC_CRS,
        "city": {
            "path": str(city_path.resolve()),
            "sha256": sha256_file(city_path),
            "source_crs": city.crs.to_string(),
            "feature_count": int(len(city)),
            "area_m2": float(city_union.area),
            "bounds_m": [float(value) for value in city_union.bounds],
        },
        "counties": {
            "path": str(county_path.resolve()),
            "sha256": sha256_file(county_path),
            "source_crs": counties.crs.to_string(),
            "feature_count": int(len(counties)),
            "names": [row["name"] for row in county_rows],
            "total_area_m2": float(sum(item["geometry"].area for item in county_rows)),
        },
        "clip_policy": "all geometry intersection and area calculations occur after reprojection to EPSG:32649; outside-city slivers excluded",
    }
    return city_union, county_rows, audit


def _inventory_maps(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = read_json(path, {})
    by_source: dict[str, dict[str, Any]] = {}
    by_stem: dict[str, dict[str, Any]] = {}
    for record in payload.get("records", []) if isinstance(payload, dict) else []:
        if not isinstance(record, dict) or not record.get("source_tif"):
            continue
        item = dict(record)
        source_key = str(Path(str(record["source_tif"])).resolve())
        by_source[source_key] = item
        if record.get("stem"):
            by_stem[str(record["stem"])] = item
    return by_source, by_stem


def _manifest_paths(*roots: Path) -> Iterator[Path]:
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            candidate = child / "04_geometry" / "GEOMETRY_MANIFEST.json"
            if candidate.is_file() and str(candidate.resolve()) not in seen:
                seen.add(str(candidate.resolve()))
                yield candidate


def discover_geometry_manifests(epoch_packages: Path, geometry_runs: Path | None) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    roots = [epoch_packages]
    if geometry_runs:
        roots.append(geometry_runs)
    for path in _manifest_paths(*roots):
        payload = read_json(path, {})
        if not isinstance(payload, dict) or payload.get("status") != "pass":
            continue
        scene = payload.get("scene") or {}
        source = scene.get("selected_source")
        if not source:
            continue
        gpkg = path.parent / "BUILDINGS.gpkg"
        if not gpkg.is_file():
            date_value = scene.get("scene_date")
            candidate = path.parent / "scene_outputs" / str(date_value) / "building_objects.gpkg"
            gpkg = candidate if candidate.is_file() else gpkg
        if not gpkg.is_file():
            continue
        manifests.append(
            {
                "manifest": str(path.resolve()),
                "gpkg": str(gpkg.resolve()),
                "source": str(Path(str(source)).resolve()),
                "scene_date": scene.get("scene_date"),
                "source_sha256": scene.get("source_sha256"),
                "checkpoint_sha256": payload.get("checkpoint_sha256"),
            }
        )
    return manifests


def _iter_geometry_chunks(path: Path, chunk_size: int = 50000) -> Iterator[Any]:
    import geopandas as gpd

    try:
        import pyogrio

        info = pyogrio.read_info(path, layer="buildings")
        total = int(info.get("features", 0))
        for offset in range(0, total, chunk_size):
            frame = pyogrio.read_dataframe(
                path,
                layer="buildings",
                skip_features=offset,
                max_features=min(chunk_size, total - offset),
                use_arrow=False,
            )
            if not frame.empty:
                yield frame
        return
    except Exception:
        pass
    # Fiona/GeoPandas fallback for environments without pyogrio.  The slice
    # keeps memory bounded even for the large R2 packages.
    try:
        import fiona

        with fiona.open(path, layer="buildings") as source:
            total = len(source)
        for offset in range(0, total, chunk_size):
            frame = gpd.read_file(path, layer="buildings", rows=slice(offset, offset + chunk_size))
            if not frame.empty:
                yield frame
    except Exception as exc:
        raise RuntimeError(f"cannot read geometry {path}: {type(exc).__name__}: {exc}") from exc


def _aggregate_geometry(
    item: dict[str, Any],
    city_union: Any,
    counties: list[dict[str, Any]],
    inventory: dict[str, dict[str, Any]],
    clip_region: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    import geopandas as gpd

    source_key = str(Path(item["source"]).resolve())
    record = inventory.get(source_key, {})
    if not record:
        # Existing packages may reference a byte-identical combined TIFF via
        # the D: mirror while the corrected inventory indexes the E: mirror.
        # Fall back by immutable task stem so those packages are not silently
        # dropped from the city/county totals.
        source_stem = Path(source_key).parent.name.removesuffix("_combined")
        record = next((value for value in inventory.values() if value.get("stem") == source_stem), {})
    nominal_year = record.get("nominal_year")
    scene_date = item.get("scene_date") or record.get("nominal_date")
    city_count = 0
    city_area = 0.0
    county_counts = Counter()
    county_areas = Counter()
    error: str | None = None
    try:
        city_clip = city_union if clip_region is None else city_union.intersection(clip_region)
        county_clips = {
            county["name"]: county["geometry"] if clip_region is None else county["geometry"].intersection(clip_region)
            for county in counties
        }
        for frame in _iter_geometry_chunks(Path(item["gpkg"])):
            if frame.crs is None:
                raise RuntimeError("geometry CRS missing")
            if frame.crs.to_string() != METRIC_CRS:
                frame = frame.to_crs(METRIC_CRS)
            frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
            if frame.empty:
                continue
            city_intersections = frame.geometry.intersection(city_clip)
            valid_city = city_intersections.notna() & ~city_intersections.is_empty
            city_count += int(valid_city.sum())
            city_area += float(city_intersections[valid_city].area.sum())
            for county in counties:
                intersections = frame.geometry.intersection(county_clips[county["name"]])
                valid = intersections.notna() & ~intersections.is_empty
                county_counts[county["name"]] += int(valid.sum())
                county_areas[county["name"]] += float(intersections[valid].area.sum())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    summary = {
        "source": source_key,
        "stem": record.get("stem") or Path(source_key).parent.name.removesuffix("_combined"),
        "nominal_year": nominal_year,
        "nominal_date": record.get("nominal_date"),
        "scene_date": scene_date,
        "city_object_count": city_count if error is None else None,
        "city_area_m2": city_area if error is None else None,
        "geometry_path": item["gpkg"],
        "geometry_manifest": item["manifest"],
        "checkpoint_sha256": item.get("checkpoint_sha256"),
        "source_sha256": item.get("source_sha256"),
        "status": "pass" if error is None else "blocked_geometry_read",
        "error": error,
        "scientific_status": "candidate_not_ground_truth",
    }
    county_rows = [
        {
            "source": source_key,
            "stem": summary["stem"],
            "nominal_year": nominal_year,
            "scene_date": scene_date,
            "county": county["name"],
            "object_count": int(county_counts[county["name"]]) if error is None else None,
            "area_m2": float(county_areas[county["name"]]) if error is None else None,
            "status": summary["status"],
        }
        for county in counties
    ]
    return summary, county_rows, error


def _bbox_iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    left = max(float(a[0]), float(b[0]))
    bottom = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    top = min(float(a[3]), float(b[3]))
    if right <= left or top <= bottom:
        return 0.0
    inter = (right - left) * (top - bottom)
    area_a = max(0.0, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    area_b = max(0.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
    return inter / max(1e-12, area_a + area_b - inter)


def canonical_exclusive_regions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign overlapping source footprints to deterministic exclusive regions.

    GEID copies can contain both a broad central mosaic and smaller R2/Rx
    children covering the same coordinates.  Summing every package would
    double-count buildings.  We give smaller footprints precedence (then use
    the source path as a stable tie-breaker) and subtract already-covered
    footprint from later sources.  The returned regions are disjoint, so
    epoch sums are additive while preserving all covered area.
    """
    from shapely.geometry import GeometryCollection
    from shapely.ops import unary_union

    ordered = sorted(
        (item for item in items if item.get("footprint") is not None and not item["footprint"].is_empty),
        key=lambda item: (float(item["footprint"].area), str(item.get("source", ""))),
    )
    covered = GeometryCollection()
    result: list[dict[str, Any]] = []
    for order, item in enumerate(ordered):
        footprint = item["footprint"]
        exclusive = footprint if covered.is_empty else footprint.difference(covered)
        result.append(
            {
                **item,
                "order": order,
                "exclusive": exclusive,
                "footprint_area_m2": float(footprint.area),
                "exclusive_area_m2": float(exclusive.area),
            }
        )
        covered = footprint if covered.is_empty else unary_union([covered, footprint])
    return result


def _mode_actual_date_records(records: Iterable[dict[str, Any]]) -> str | None:
    counts: Counter[str] = Counter()
    for record in records:
        for key, value in (record.get("date_counts") or {}).items():
            if parse_iso_date(key):
                try:
                    counts[str(key)] += int(value or 0)
                except (TypeError, ValueError):
                    continue
    return counts.most_common(1)[0][0] if counts else None


def _mode_actual_date(record: dict[str, Any]) -> str | None:
    counts = record.get("date_counts") or {}
    if not isinstance(counts, dict) or not counts:
        return None
    valid = [(str(key), int(value or 0)) for key, value in counts.items() if parse_iso_date(key)]
    return max(valid, key=lambda item: item[1])[0] if valid else None


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    import geopandas as gpd  # noqa: F401  # imported to fail clearly before long work

    run = args.run_dir.resolve()
    stats_dir = run / "04_statistics"
    temporal_dir = run / "05_temporal_outputs"
    visual_dir = run / "06_visuals"
    qa_dir = run / "07_qa"
    for directory in (stats_dir, temporal_dir, visual_dir, qa_dir):
        directory.mkdir(parents=True, exist_ok=True)
    inventory, inventory_by_stem = _inventory_maps(args.inventory.resolve())
    city_union, counties, boundary_audit = _load_boundaries(args.city_boundary.resolve(), args.county_boundary.resolve())
    city_boundary_area_m2 = float(city_union.area)
    county_boundary_area_m2 = {county["name"]: float(county["geometry"].area) for county in counties}
    atomic_json(stats_dir / "BOUNDARY_AUDIT.json", boundary_audit)
    manifests = discover_geometry_manifests(args.epoch_packages.resolve(), args.geometry_runs.resolve() if args.geometry_runs else None)
    source_meta = {key: value.get("metadata", {}) for key, value in inventory.items()}
    # Build deterministic, disjoint source regions before aggregating epochs.
    # This prevents overlapping central/R2/Rx packages from inflating city or
    # county totals while keeping the per-source audit untouched.
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform as shapely_transform

    to_metric = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True).transform
    region_items_by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in manifests:
        source_key = str(Path(item["source"]).resolve())
        record = inventory.get(source_key, {})
        if not record:
            source_stem = Path(source_key).parent.name.removesuffix("_combined")
            record = next((value for value in inventory.values() if value.get("stem") == source_stem), {})
        try:
            nominal_year = int(record.get("nominal_year") or 0)
        except (TypeError, ValueError):
            nominal_year = 0
        bounds = (record.get("metadata") or {}).get("bounds") if isinstance(record, dict) else None
        if nominal_year not in (2018, 2019, 2020) or not bounds or len(bounds) != 4:
            continue
        try:
            footprint = shapely_transform(to_metric, box(*[float(value) for value in bounds]))
        except (TypeError, ValueError):
            continue
        region_items_by_year[nominal_year].append({"source": source_key, "footprint": footprint})
    canonical_regions: dict[str, dict[str, Any]] = {}
    for year, items in region_items_by_year.items():
        for region in canonical_exclusive_regions(items):
            canonical_regions.setdefault(region["source"], {**region, "nominal_year": year})
    write_csv(
        stats_dir / "CANONICAL_SOURCE_ASSIGNMENT.csv",
        [
            {
                "nominal_year": item.get("nominal_year"),
                "source": item.get("source"),
                "priority_order": item.get("order"),
                "footprint_area_m2": item.get("footprint_area_m2"),
                "exclusive_footprint_area_m2": item.get("exclusive_area_m2"),
                "policy": "smallest footprint first; later overlapping footprints clipped to exclusive remainder",
            }
            for item in sorted(canonical_regions.values(), key=lambda value: (int(value.get("nominal_year") or 0), int(value.get("order") or 0), str(value.get("source"))))
        ],
        ["nominal_year", "source", "priority_order", "footprint_area_m2", "exclusive_footprint_area_m2", "policy"],
    )
    source_rows: list[dict[str, Any]] = []
    county_rows: list[dict[str, Any]] = []
    for item in manifests:
        summary, rows, _ = _aggregate_geometry(item, city_union, counties, inventory)
        source_rows.append(summary)
        county_rows.extend(rows)
    source_fields = [
        "source", "stem", "nominal_year", "nominal_date", "scene_date", "city_object_count", "city_area_m2",
        "geometry_path", "geometry_manifest", "checkpoint_sha256", "source_sha256", "status", "error", "scientific_status",
    ]
    county_fields = ["source", "stem", "nominal_year", "scene_date", "county", "object_count", "area_m2", "status"]
    write_csv(stats_dir / "XIAN_SOURCE_BOUNDARY_STATS.csv", source_rows, source_fields)
    write_csv(stats_dir / "XIAN_COUNTY_SOURCE_BOUNDARY_STATS.csv", county_rows, county_fields)
    canonical_source_rows: list[dict[str, Any]] = []
    canonical_county_rows: list[dict[str, Any]] = []
    canonical_seen: set[str] = set()
    for item in manifests:
        source_key = str(Path(item["source"]).resolve())
        region = canonical_regions.get(source_key)
        if not region or source_key in canonical_seen or region.get("exclusive") is None or region["exclusive"].is_empty:
            continue
        canonical_seen.add(source_key)
        summary, rows, _ = _aggregate_geometry(item, city_union, counties, inventory, clip_region=region["exclusive"])
        canonical_source_rows.append(summary)
        canonical_county_rows.extend(rows)
    epoch_rows: list[dict[str, Any]] = []
    county_epoch_rows: list[dict[str, Any]] = []
    for year in (2018, 2019, 2020):
        selected = [row for row in canonical_source_rows if int(row.get("nominal_year") or 0) == year and row.get("status") == "pass"]
        actual_dates = sorted({str(row.get("scene_date")) for row in selected if row.get("scene_date")})
        epoch_rows.append(
            {
                "epoch": year,
                "source_count": len(selected),
                "source_stems": ";".join(sorted(str(row.get("stem")) for row in selected)),
                "building_object_count_inside_city": int(sum(int(row.get("city_object_count") or 0) for row in selected)),
                "building_area_m2_inside_city": float(sum(float(row.get("city_area_m2") or 0.0) for row in selected)),
                "building_area_share_of_city_boundary": float(sum(float(row.get("city_area_m2") or 0.0) for row in selected)) / city_boundary_area_m2 if city_boundary_area_m2 else None,
                "actual_scene_dates": ";".join(actual_dates),
                "nominal_year_is_not_observation": True,
                "area_scope": "Xi'an city boundary intersection, EPSG:32649",
                "scientific_status": "candidate_not_ground_truth",
            }
        )
        for county in counties:
            selected_county = [row for row in canonical_county_rows if int(row.get("nominal_year") or 0) == year and row.get("county") == county["name"] and row.get("status") == "pass"]
            county_epoch_rows.append(
                {
                    "epoch": year,
                    "county": county["name"],
                    "source_count": len({row.get("source") for row in selected_county}),
                    "building_object_count": int(sum(int(row.get("object_count") or 0) for row in selected_county)),
                    "building_area_m2": float(sum(float(row.get("area_m2") or 0.0) for row in selected_county)),
                    "building_area_share_of_city_boundary": float(sum(float(row.get("area_m2") or 0.0) for row in selected_county)) / city_boundary_area_m2 if city_boundary_area_m2 else None,
                    "building_area_share_of_county_boundary": float(sum(float(row.get("area_m2") or 0.0) for row in selected_county)) / county_boundary_area_m2[county["name"]] if county_boundary_area_m2.get(county["name"]) else None,
                    "area_scope": "county boundary intersection, EPSG:32649",
                    "scientific_status": "candidate_not_ground_truth",
                }
            )
    write_csv(
        stats_dir / "XIAN_CITY_EPOCH_STATS.csv",
        epoch_rows,
        ["epoch", "source_count", "source_stems", "building_object_count_inside_city", "building_area_m2_inside_city", "building_area_share_of_city_boundary", "actual_scene_dates", "nominal_year_is_not_observation", "area_scope", "scientific_status"],
    )
    write_csv(
        stats_dir / "XIAN_COUNTY_EPOCH_STATS.csv",
        county_epoch_rows,
        ["epoch", "county", "source_count", "building_object_count", "building_area_m2", "building_area_share_of_city_boundary", "building_area_share_of_county_boundary", "area_scope", "scientific_status"],
    )

    # Candidate change accounting compares the deduplicated epoch footprints,
    # not arbitrary source pairs.  This keeps overlapping central/R2/Rx
    # packages from being counted repeatedly.  It remains a conservative area
    # proxy (no object matcher is implied) and is explicitly candidate evidence.
    source_record_by_source = inventory
    change_rows: list[dict[str, Any]] = []
    county_change_rows: list[dict[str, Any]] = []
    canonical_by_year = {
        year: [row for row in canonical_source_rows if int(row.get("nominal_year") or 0) == year and row.get("status") == "pass"]
        for year in (2018, 2019, 2020)
    }
    canonical_county_by_year = {
        year: [row for row in canonical_county_rows if int(row.get("nominal_year") or 0) == year and row.get("status") == "pass"]
        for year in (2018, 2019, 2020)
    }
    for year_a, year_b in ((2018, 2019), (2019, 2020)):
        left_records = [record for record in inventory.values() if int(record.get("nominal_year") or 0) == year_a]
        right_records = [record for record in inventory.values() if int(record.get("nominal_year") or 0) == year_b]
        temporal = classify_temporal_pair(_mode_actual_date_records(left_records), _mode_actual_date_records(right_records), year_a, year_b)
        left_rows = canonical_by_year.get(year_a, [])
        right_rows = canonical_by_year.get(year_b, [])
        area_a = float(sum(float(row.get("city_area_m2") or 0.0) for row in left_rows))
        area_b = float(sum(float(row.get("city_area_m2") or 0.0) for row in right_rows))
        strict_ok = temporal["status"] in {"VALID_TEMPORAL", "OFF_DATE_VALID"} and temporal.get("genuine_interval")
        stem_a = ";".join(sorted(str(row.get("stem")) for row in left_rows))
        stem_b = ";".join(sorted(str(row.get("stem")) for row in right_rows))
        source_a = ";".join(sorted(str(row.get("source")) for row in left_rows))
        source_b = ";".join(sorted(str(row.get("source")) for row in right_rows))
        change_rows.append(
            {
                "epoch_a": year_a,
                "epoch_b": year_b,
                "source_a": source_a,
                "source_b": source_b,
                "stem_a": stem_a,
                "stem_b": stem_b,
                "bbox_iou": None,
                "actual_date_a": temporal.get("actual_date_a"),
                "actual_date_b": temporal.get("actual_date_b"),
                "temporal_status": temporal["status"],
                "temporal_interval_genuine": temporal.get("genuine_interval"),
                "building_area_a_m2_inside_city": area_a,
                "building_area_b_m2_inside_city": area_b,
                "building_area_a_share_of_city_boundary": area_a / city_boundary_area_m2 if city_boundary_area_m2 else None,
                "building_area_b_share_of_city_boundary": area_b / city_boundary_area_m2 if city_boundary_area_m2 else None,
                "candidate_new_area_m2_inside_city": max(area_b - area_a, 0.0) if strict_ok else None,
                "candidate_demolition_area_m2_inside_city": max(area_a - area_b, 0.0) if strict_ok else None,
                "candidate_new_area_share_of_city_boundary": max(area_b - area_a, 0.0) / city_boundary_area_m2 if strict_ok and city_boundary_area_m2 else None,
                "candidate_demolition_area_share_of_city_boundary": max(area_a - area_b, 0.0) / city_boundary_area_m2 if strict_ok and city_boundary_area_m2 else None,
                "strict_change_fields_excluded": not strict_ok,
                "exclusion_reason": None if strict_ok else temporal["status"],
                "area_scope": "Xi'an city boundary intersection, EPSG:32649",
                "scientific_status": "candidate_not_ground_truth",
            }
        )
        left_county = {row["county"]: row for row in canonical_county_by_year.get(year_a, [])}
        right_county = {row["county"]: row for row in canonical_county_by_year.get(year_b, [])}
        for county in counties:
            name = county["name"]
            row_a = left_county.get(name, {})
            row_b = right_county.get(name, {})
            ca = float(row_a.get("area_m2") or 0.0)
            cb = float(row_b.get("area_m2") or 0.0)
            county_change_rows.append(
                {
                    "epoch_a": year_a,
                    "epoch_b": year_b,
                    "county": name,
                    "stem_a": stem_a,
                    "stem_b": stem_b,
                    "actual_date_a": temporal.get("actual_date_a"),
                    "actual_date_b": temporal.get("actual_date_b"),
                    "temporal_status": temporal["status"],
                    "building_area_a_m2": ca,
                    "building_area_b_m2": cb,
                    "building_area_a_share_of_city_boundary": ca / city_boundary_area_m2 if city_boundary_area_m2 else None,
                    "building_area_b_share_of_city_boundary": cb / city_boundary_area_m2 if city_boundary_area_m2 else None,
                    "building_area_a_share_of_county_boundary": ca / county_boundary_area_m2[name] if county_boundary_area_m2.get(name) else None,
                    "building_area_b_share_of_county_boundary": cb / county_boundary_area_m2[name] if county_boundary_area_m2.get(name) else None,
                    "candidate_new_area_m2": max(cb - ca, 0.0) if strict_ok else None,
                    "candidate_demolition_area_m2": max(ca - cb, 0.0) if strict_ok else None,
                    "candidate_new_area_share_of_city_boundary": max(cb - ca, 0.0) / city_boundary_area_m2 if strict_ok and city_boundary_area_m2 else None,
                    "candidate_demolition_area_share_of_city_boundary": max(ca - cb, 0.0) / city_boundary_area_m2 if strict_ok and city_boundary_area_m2 else None,
                    "candidate_new_area_share_of_county_boundary": max(cb - ca, 0.0) / county_boundary_area_m2[name] if strict_ok and county_boundary_area_m2.get(name) else None,
                    "candidate_demolition_area_share_of_county_boundary": max(ca - cb, 0.0) / county_boundary_area_m2[name] if strict_ok and county_boundary_area_m2.get(name) else None,
                    "strict_change_fields_excluded": not strict_ok,
                    "area_scope": "county boundary intersection, EPSG:32649",
                    "scientific_status": "candidate_not_ground_truth",
                }
            )
    write_csv(temporal_dir / "XIAN_CITY_CHANGE_STATS.csv", change_rows, list(change_rows[0].keys()) if change_rows else ["epoch_a", "epoch_b", "temporal_status"])
    write_csv(temporal_dir / "XIAN_COUNTY_CHANGE_STATS.csv", county_change_rows, list(county_change_rows[0].keys()) if county_change_rows else ["epoch_a", "epoch_b", "county", "temporal_status"])
    # A compact state table is intentionally derived only from genuine
    # before/after intervals.  ``unchanged`` is the conservative overlap-area
    # proxy (the smaller of the two clipped footprints), while rows without a
    # real interval retain NULL event fields and are excluded from strict
    # change denominators.
    city_state_rows: list[dict[str, Any]] = []
    for row in change_rows:
        valid = not bool(row.get("strict_change_fields_excluded"))
        area_a = float(row.get("building_area_a_m2_inside_city") or 0.0)
        area_b = float(row.get("building_area_b_m2_inside_city") or 0.0)
        city_state_rows.append(
            {
                "epoch_a": row.get("epoch_a"),
                "epoch_b": row.get("epoch_b"),
                "stem_a": row.get("stem_a"),
                "stem_b": row.get("stem_b"),
                "temporal_status": row.get("temporal_status"),
                "new_area_m2_inside_city": max(area_b - area_a, 0.0) if valid else None,
                "demolition_area_m2_inside_city": max(area_a - area_b, 0.0) if valid else None,
                "unchanged_area_m2_inside_city": min(area_a, area_b) if valid else None,
                "new_area_share_of_city_boundary": max(area_b - area_a, 0.0) / city_boundary_area_m2 if valid and city_boundary_area_m2 else None,
                "demolition_area_share_of_city_boundary": max(area_a - area_b, 0.0) / city_boundary_area_m2 if valid and city_boundary_area_m2 else None,
                "unchanged_area_share_of_city_boundary": min(area_a, area_b) / city_boundary_area_m2 if valid and city_boundary_area_m2 else None,
                "strict_change_fields_excluded": not valid,
                "area_scope": "Xi'an city boundary intersection, EPSG:32649",
                "scientific_status": "candidate_not_ground_truth",
            }
        )
    county_state_rows: list[dict[str, Any]] = []
    for row in county_change_rows:
        valid = not bool(row.get("strict_change_fields_excluded"))
        area_a = float(row.get("building_area_a_m2") or 0.0)
        area_b = float(row.get("building_area_b_m2") or 0.0)
        county_state_rows.append(
            {
                "epoch_a": row.get("epoch_a"),
                "epoch_b": row.get("epoch_b"),
                "county": row.get("county"),
                "stem_a": row.get("stem_a"),
                "stem_b": row.get("stem_b"),
                "temporal_status": row.get("temporal_status"),
                "new_area_m2": max(area_b - area_a, 0.0) if valid else None,
                "demolition_area_m2": max(area_a - area_b, 0.0) if valid else None,
                "unchanged_area_m2": min(area_a, area_b) if valid else None,
                "new_area_share_of_county_boundary": max(area_b - area_a, 0.0) / county_boundary_area_m2[str(row.get("county"))] if valid and county_boundary_area_m2.get(str(row.get("county"))) else None,
                "demolition_area_share_of_county_boundary": max(area_a - area_b, 0.0) / county_boundary_area_m2[str(row.get("county"))] if valid and county_boundary_area_m2.get(str(row.get("county"))) else None,
                "unchanged_area_share_of_county_boundary": min(area_a, area_b) / county_boundary_area_m2[str(row.get("county"))] if valid and county_boundary_area_m2.get(str(row.get("county"))) else None,
                "strict_change_fields_excluded": not valid,
                "area_scope": "county boundary intersection, EPSG:32649",
                "scientific_status": "candidate_not_ground_truth",
            }
        )
    write_csv(
        temporal_dir / "XIAN_CITY_STATE_STATS.csv",
        city_state_rows,
        ["epoch_a", "epoch_b", "stem_a", "stem_b", "temporal_status", "new_area_m2_inside_city", "demolition_area_m2_inside_city", "unchanged_area_m2_inside_city", "new_area_share_of_city_boundary", "demolition_area_share_of_city_boundary", "unchanged_area_share_of_city_boundary", "strict_change_fields_excluded", "area_scope", "scientific_status"],
    )
    write_csv(
        temporal_dir / "XIAN_COUNTY_STATE_STATS.csv",
        county_state_rows,
        ["epoch_a", "epoch_b", "county", "stem_a", "stem_b", "temporal_status", "new_area_m2", "demolition_area_m2", "unchanged_area_m2", "new_area_share_of_county_boundary", "demolition_area_share_of_county_boundary", "unchanged_area_share_of_county_boundary", "strict_change_fields_excluded", "area_scope", "scientific_status"],
    )
    atomic_json(
        temporal_dir / "TEMPORAL_ACCOUNTING_RECEIPT.json",
        {
            "created_at": now_iso(),
            "status": "pass" if source_rows else "blocked_no_geometry",
            "source_manifest_count": len(manifests),
            "source_success_count": sum(row.get("status") == "pass" for row in source_rows),
            "canonical_source_count": len(canonical_source_rows),
            "overlap_policy": "epoch areas/counts use deterministic exclusive source footprints; per-source audit remains unmodified",
            "city_epoch_count": len(epoch_rows),
            "city_change_pair_count": len(change_rows),
            "county_change_row_count": len(county_change_rows),
            "city_state_row_count": len(city_state_rows),
            "county_state_row_count": len(county_state_rows),
            "strict_policy": "new/demolition are omitted when actual dates are missing or duplicate; nominal year never supplies evidence",
            "boundary_audit": boundary_audit,
            "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
            "raw_inputs_modified": False,
            "scientific_status": "candidate_not_ground_truth",
        },
    )
    render_visuals(args, run, city_union, counties, epoch_rows, county_epoch_rows, change_rows)
    atomic_json(
        qa_dir / "BOUNDARY_TEMPORAL_QA.json",
        {
            "status": "pass" if source_rows else "blocked_no_geometry",
            "boundary_inside_only": True,
            "metric_crs": METRIC_CRS,
            "source_geometry_count": len(source_rows),
            "canonical_source_geometry_count": len(canonical_source_rows),
            "overlap_double_count_prevented": True,
            "geometry_failures": [row for row in source_rows if row.get("status") != "pass"],
            "temporal_status_counts": dict(Counter(str(row.get("temporal_status")) for row in change_rows)),
            "candidate_not_ground_truth": True,
            "raw_inputs_modified": False,
            "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        },
    )
    return {
        "status": "pass" if source_rows else "blocked_no_geometry",
        "run_dir": str(run),
        "source_geometry_count": len(source_rows),
        "canonical_source_geometry_count": len(canonical_source_rows),
        "city_epoch_stats": str((stats_dir / "XIAN_CITY_EPOCH_STATS.csv").resolve()),
        "county_epoch_stats": str((stats_dir / "XIAN_COUNTY_EPOCH_STATS.csv").resolve()),
        "city_change_stats": str((temporal_dir / "XIAN_CITY_CHANGE_STATS.csv").resolve()),
        "county_change_stats": str((temporal_dir / "XIAN_COUNTY_CHANGE_STATS.csv").resolve()),
        "city_state_stats": str((temporal_dir / "XIAN_CITY_STATE_STATS.csv").resolve()),
        "county_state_stats": str((temporal_dir / "XIAN_COUNTY_STATE_STATS.csv").resolve()),
        "boundary_audit": boundary_audit,
        "raw_inputs_modified": False,
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        "scientific_status": "candidate_not_ground_truth",
    }


def _stretch(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 2:
        data = data[None, ...]
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    if data.shape[0] == 2:
        data = np.concatenate([data, data[:1]], axis=0)
    data = data[:3]
    out = np.zeros_like(data)
    for i, channel in enumerate(data):
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, [2, 98])
        if high <= low:
            low, high = float(finite.min()), float(finite.max())
        out[i] = np.clip((channel - low) / max(1e-6, high - low), 0, 1)
    return np.moveaxis(out, 0, -1)


def render_visuals(
    args: argparse.Namespace,
    run: Path,
    city_union: Any,
    counties: list[dict[str, Any]],
    epoch_rows: list[dict[str, Any]],
    county_epoch_rows: list[dict[str, Any]],
    change_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from shapely.geometry import box

    visual_dir = run / "06_visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    overview_root = args.assembly_run.resolve() / "01_epoch_mosaics"
    for year in (2018, 2019, 2020):
        overview = overview_root / f"XIAN_{year}_OVERVIEW.tif"
        if not overview.is_file():
            continue
        with rasterio.open(overview) as src:
            arr = src.read()
            bounds = src.bounds
            crs = src.crs
        # Boundaries are metric; use a lightweight geographic transform for
        # plotting so the raster and outlines share the same coordinate space.
        import geopandas as gpd

        city_gdf = gpd.GeoDataFrame(geometry=[city_union], crs=METRIC_CRS).to_crs(crs)
        county_gdf = gpd.GeoDataFrame(geometry=[item["geometry"] for item in counties], crs=METRIC_CRS).to_crs(crs)
        fig, ax = plt.subplots(figsize=(12, 8), dpi=180)
        ax.imshow(_stretch(arr), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], origin="upper")
        city_gdf.boundary.plot(ax=ax, color="#dc2626", linewidth=1.1, label="Xi'an city boundary")
        county_gdf.boundary.plot(ax=ax, color="#fbbf24", linewidth=0.45, alpha=0.75, label="county boundary")
        ax.set_title(f"Xi'an {year} observed building coverage (city boundary scoped)")
        ax.set_xlabel(str(crs))
        ax.set_ylabel(str(crs))
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(visual_dir / f"XIAN_{year}_BUILDING_BOUNDARY_OVERVIEW.png", dpi=300, facecolor="white")
        fig.savefig(visual_dir / f"XIAN_{year}_BUILDING_BOUNDARY_OVERVIEW.pdf", facecolor="white")
        plt.close(fig)

    # Temporal reliability is a separate evidence layer.  The source raster is
    # the date-aware, low-resolution product from assembly; colours are codes,
    # not a claim that a nominal year is observed.
    reliability = args.assembly_run.resolve() / "02_temporal_reliability" / "TEMPORAL_RELIABILITY_OVERVIEW.tif"
    if reliability.is_file():
        with rasterio.open(reliability) as src:
            rel = src.read(1)
            bounds = src.bounds
            crs = src.crs
        import geopandas as gpd

        city_gdf = gpd.GeoDataFrame(geometry=[city_union], crs=METRIC_CRS).to_crs(crs)
        county_gdf = gpd.GeoDataFrame(geometry=[item["geometry"] for item in counties], crs=METRIC_CRS).to_crs(crs)
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(["#f8fafc", "#16a34a", "#f59e0b", "#7c3aed", "#dc2626"])
        fig, ax = plt.subplots(figsize=(12, 8), dpi=180)
        ax.imshow(rel, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], origin="upper", cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        city_gdf.boundary.plot(ax=ax, color="#111827", linewidth=1.0)
        county_gdf.boundary.plot(ax=ax, color="#374151", linewidth=0.35, alpha=0.65)
        ax.set_title("Xi'an temporal reliability | actual-date evidence only")
        ax.set_xlabel(str(crs))
        ax.set_ylabel(str(crs))
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(color="#16a34a", label="VALID_TEMPORAL"),
                Patch(color="#f59e0b", label="OFF_DATE_VALID"),
                Patch(color="#7c3aed", label="DUPLICATE_EPOCH"),
                Patch(color="#dc2626", label="TEMPORAL_INSUFFICIENT"),
            ],
            loc="upper right",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(visual_dir / "FIG_TEMPORAL_RELIABILITY_MAP.png", dpi=300, facecolor="white")
        fig.savefig(visual_dir / "FIG_TEMPORAL_RELIABILITY_MAP.pdf", facecolor="white")
        plt.close(fig)

    # Summary chart: areas/counts are clipped to the city boundary and carry
    # the same candidate status as the tables.
    years = [int(row["epoch"]) for row in epoch_rows]
    areas = [float(row.get("building_area_m2_inside_city") or 0.0) / 1e6 for row in epoch_rows]
    counts = [int(row.get("building_object_count_inside_city") or 0) for row in epoch_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    axes[0].bar([str(year) for year in years], areas, color=["#2563eb", "#14b8a6", "#f97316"])
    axes[0].set_ylabel("Building area inside city boundary (km²)")
    axes[0].set_title("Observed footprint area")
    axes[1].bar([str(year) for year in years], counts, color=["#2563eb", "#14b8a6", "#f97316"])
    axes[1].set_ylabel("Clipped object count")
    axes[1].set_title("Observed building candidates")
    fig.suptitle("Xi'an boundary-scoped epoch statistics | candidate evidence")
    fig.tight_layout()
    fig.savefig(visual_dir / "FIG_XIAN_CITY_AREA_COUNT.png", dpi=300, facecolor="white")
    fig.savefig(visual_dir / "FIG_XIAN_CITY_AREA_COUNT.pdf", facecolor="white")
    plt.close(fig)

    # County panel uses fixed county ordering from the supplied boundary file.
    fig, ax = plt.subplots(figsize=(13, 6), dpi=180)
    for year, color in zip(years, ("#2563eb", "#14b8a6", "#f97316")):
        values = [
            float(row.get("building_area_m2") or 0.0) / 1e6
            for row in county_epoch_rows
            if int(row["epoch"]) == year
        ]
        names = [row["county"] for row in county_epoch_rows if int(row["epoch"]) == year]
        if values:
            positions = np.arange(len(names)) + (year - years[0] - 1) * 0.25
            ax.bar(positions, values, width=0.24, label=str(year), color=color)
    ax.set_xticks(np.arange(len(counties)))
    ax.set_xticklabels([item["name"] for item in counties], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Building area (km²), boundary clipped")
    ax.set_title("Xi'an county-level observed footprint area")
    ax.legend()
    fig.tight_layout()
    fig.savefig(visual_dir / "FIG_XIAN_COUNTY_AREA_BY_EPOCH.png", dpi=300, facecolor="white")
    fig.savefig(visual_dir / "FIG_XIAN_COUNTY_AREA_BY_EPOCH.pdf", facecolor="white")
    plt.close(fig)

    temporal_counts = Counter(str(row.get("temporal_status")) for row in change_rows)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    labels = list(temporal_counts) or ["TEMPORAL_INSUFFICIENT"]
    values = [temporal_counts[label] for label in labels] if temporal_counts else [0]
    ax.bar(labels, values, color="#64748b")
    ax.set_ylabel("Spatial source pairs")
    ax.set_title("Actual-date reliability of candidate change pairs")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(visual_dir / "FIG_TEMPORAL_RELIABILITY.png", dpi=300, facecolor="white")
    fig.savefig(visual_dir / "FIG_TEMPORAL_RELIABILITY.pdf", facecolor="white")
    plt.close(fig)
    atomic_json(
        visual_dir / "VISUAL_RECEIPT.json",
        {
            "status": "pass",
            "figure_count": len(list(visual_dir.glob("*.png"))),
            "background_policy": "date-specific low-resolution overview GeoTIFFs from the actual-date-aware assembly; no synthetic basemap",
            "boundary_overlay": "city and county GeoJSON reprojected to metric CRS",
            "scientific_status": "candidate_not_ground_truth",
            "raw_inputs_modified": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-run", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--epoch-packages", type=Path, default=Path(r"E:\URRS Output\xiannew\epoch_packages"))
    parser.add_argument("--geometry-runs", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--city-boundary", type=Path, default=DEFAULT_CITY)
    parser.add_argument("--county-boundary", type=Path, default=DEFAULT_COUNTIES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.assembly_run = args.assembly_run.resolve()
    args.inventory = (args.inventory or (args.assembly_run / "00_audit" / "SOURCE_INVENTORY.json")).resolve()
    args.epoch_packages = args.epoch_packages.resolve()
    args.geometry_runs = args.geometry_runs.resolve() if args.geometry_runs else args.assembly_run / "03_geometry_runs"
    args.run_dir = args.run_dir.resolve()
    result = build_outputs(args)
    atomic_json(args.run_dir / "receipts" / "BOUNDARY_TEMPORAL_OUTPUT_RECEIPT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
