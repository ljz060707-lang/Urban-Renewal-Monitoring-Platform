from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union


SCHEMA_VERSION = "0.4.0"
CLASS_ALIASES = ("class_name", "class", "label", "category", "类别", "类型")
DATE_ALIASES = ("source_date", "date", "time", "日期", "影像日期")
ID_ALIASES = ("object_id", "id", "fid", "ID", "编号")

STANDARD_COLUMNS = [
    "object_id", "class_name", "geometry_role", "review_status", "source_date",
    "schema_version", "coordinate_encoding", "notes", "tracking_geometry_wkt",
    "validation_action", "geometry",
]


def read_building_labels(path: str | Path, *, building_layer: str = "buildings", rows=None) -> gpd.GeoDataFrame:
    """Read a deterministic building layer; never guess among multiple GPKG layers."""
    path = Path(path)
    kwargs = {"rows": rows} if rows is not None else {}
    if path.suffix.casefold() != ".gpkg":
        return gpd.read_file(path, **kwargs)
    try:
        layers = gpd.list_layers(path)["name"].astype(str).tolist()
    except Exception:
        try:
            import fiona
            layers = list(fiona.listlayers(path))
        except Exception as exc:
            raise ValueError(f"Cannot list GeoPackage layers: {path}: {exc}") from exc
    if building_layer in layers:
        return gpd.read_file(path, layer=building_layer, **kwargs)
    if len(layers) == 1:
        return gpd.read_file(path, layer=layers[0], **kwargs)
    raise ValueError(
        f"GeoPackage {path} has multiple layers but no '{building_layer}' layer: {', '.join(layers)}. "
        f"Rename the building layer to '{building_layer}' or set data.building_layer in project.yaml."
    )


@dataclass
class GroundTruthResult:
    accepted: gpd.GeoDataFrame
    rejected: gpd.GeoDataFrame
    filtered_small: gpd.GeoDataFrame
    field_mapping: dict[str, str]
    stats: dict[str, int]


def _find_field(columns, aliases: tuple[str, ...]) -> str | None:
    exact = set(columns)
    lower = {str(column).casefold(): str(column) for column in columns}
    for alias in aliases:
        if alias in exact:
            return alias
        if alias.casefold() in lower:
            return lower[alias.casefold()]
    return None


def _polygonal(geometry):
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if geometry.geom_type == "GeometryCollection":
        parts = [part for part in geometry.geoms if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty]
        if not parts:
            return None
        merged = unary_union(parts)
        return merged if isinstance(merged, (Polygon, MultiPolygon)) else None
    return None


def _stable_id(source_date: str, geometry) -> str:
    normalized = geometry.normalize()
    return "gt_" + sha1(source_date.encode("utf-8") + normalized.wkb).hexdigest()[:16]


def _frame(rows: list[dict[str, Any]], crs) -> gpd.GeoDataFrame:
    if not rows:
        return gpd.GeoDataFrame({column: [] for column in STANDARD_COLUMNS}, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def normalize_label_frame(
    frame: gpd.GeoDataFrame,
    *,
    source_date: str,
    target_crs,
    image_bounds: tuple[float, float, float, float],
    min_building_area_m2: float = 10.0,
    classes: list[str] | tuple[str, ...] = ("building",),
) -> GroundTruthResult:
    """Normalize permissive QGIS polygons into the v0.4 training contract."""
    if frame.crs is None:
        raise ValueError("Ground Truth CRS is missing")
    work = frame.to_crs(target_crs).copy()
    class_field = _find_field(work.columns, CLASS_ALIASES)
    date_field = _find_field(work.columns, DATE_ALIASES)
    id_field = _find_field(work.columns, ID_ALIASES)
    mapping = {
        "class_name": class_field or "<default:building>",
        "source_date": date_field or "<filename>",
        "object_id": id_field or "<generated:date+geometry>",
    }
    allowed_classes = {str(value).casefold() for value in classes}
    image_extent = box(*image_bounds)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    stats = {"input": len(work), "accepted": 0, "rejected": 0, "filtered_small": 0, "repaired": 0, "clipped": 0}

    for _, row in work.iterrows():
        original = row.geometry
        class_name = str(row.get(class_field, "building") if class_field else "building").strip() or "building"
        row_date = str(row.get(date_field, source_date) if date_field else source_date).strip()[:10] or source_date
        if row_date != source_date:
            raise ValueError(f"Ground Truth source_date {row_date} does not match filename date {source_date}")
        base = {
            "object_id": "",
            "class_name": class_name,
            "geometry_role": str(row.get("geometry_role", "visible_roof") or "visible_roof"),
            "review_status": str(row.get("review_status", "reviewed") or "reviewed"),
            "source_date": source_date,
            "schema_version": SCHEMA_VERSION,
            "coordinate_encoding": "projected" if work.crs.is_projected else "wgs84",
            "notes": str(row.get("notes", "") or ""),
            "tracking_geometry_wkt": str(row.get("tracking_geometry_wkt", "") or ""),
        }
        if original is None or original.is_empty:
            rejected.append({**base, "validation_action": "rejected_empty_geometry", "geometry": original})
            continue
        if original.geom_type not in {"Polygon", "MultiPolygon"}:
            rejected.append({**base, "validation_action": f"rejected_geometry_type:{original.geom_type}", "geometry": original})
            continue
        geometry = original
        action = "accepted"
        if not geometry.is_valid:
            geometry = _polygonal(make_valid(geometry))
            if geometry is None or not geometry.is_valid:
                rejected.append({**base, "validation_action": "rejected_invalid_after_make_valid", "geometry": original})
                continue
            action = "repaired_make_valid"
            stats["repaired"] += 1
        if class_name.casefold() not in allowed_classes:
            rejected.append({**base, "validation_action": "rejected_class_not_selected", "geometry": geometry})
            continue
        if not geometry.intersects(image_extent):
            rejected.append({**base, "validation_action": "rejected_outside_image", "geometry": geometry})
            continue
        if not image_extent.covers(geometry):
            geometry = _polygonal(make_valid(geometry.intersection(image_extent)))
            if geometry is None or geometry.is_empty:
                rejected.append({**base, "validation_action": "rejected_outside_image", "geometry": original})
                continue
            action = action + "+clipped_to_image"
            stats["clipped"] += 1
        supplied_id = str(row.get(id_field, "") if id_field else "").strip()
        base["object_id"] = supplied_id or _stable_id(source_date, geometry)
        output_row = {**base, "validation_action": action, "geometry": geometry}
        if geometry.area < float(min_building_area_m2):
            output_row["validation_action"] = "filtered_small_object"
            filtered.append(output_row)
        else:
            accepted.append(output_row)

    stats["accepted"] = len(accepted)
    stats["rejected"] = len(rejected)
    stats["filtered_small"] = len(filtered)
    return GroundTruthResult(
        accepted=_frame(accepted, work.crs),
        rejected=_frame(rejected, work.crs),
        filtered_small=_frame(filtered, work.crs),
        field_mapping=mapping,
        stats=stats,
    )


def normalize_label_file(
    label_path: str | Path,
    *,
    source_date: str,
    target_crs,
    image_bounds: tuple[float, float, float, float],
    min_building_area_m2: float = 10.0,
    classes: list[str] | tuple[str, ...] = ("building",),
    building_layer: str = "buildings",
) -> GroundTruthResult:
    return normalize_label_frame(
        read_building_labels(label_path, building_layer=building_layer),
        source_date=source_date,
        target_crs=target_crs,
        image_bounds=image_bounds,
        min_building_area_m2=min_building_area_m2,
        classes=classes,
    )


def write_ground_truth_report(results: dict[str, GroundTruthResult], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accepted = pd.concat([result.accepted for result in results.values()], ignore_index=True)
    rejected = pd.concat([result.rejected for result in results.values()], ignore_index=True)
    filtered = pd.concat([result.filtered_small for result in results.values()], ignore_index=True)
    crs = next(result.accepted.crs for result in results.values())
    frames = {
        "accepted_labels.gpkg": gpd.GeoDataFrame(accepted, geometry="geometry", crs=crs),
        "rejected_labels.gpkg": gpd.GeoDataFrame(rejected, geometry="geometry", crs=crs),
        "filtered_small_objects.gpkg": gpd.GeoDataFrame(filtered, geometry="geometry", crs=crs),
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output / name
        frame.to_file(path, driver="GPKG")
        paths[name] = path
    mappings = {date: result.field_mapping for date, result in results.items()}
    mapping_path = output / "field_mapping.json"
    mapping_path.write_text(json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["field_mapping.json"] = mapping_path
    lines = ["# Ground Truth validation report", ""]
    for date, result in sorted(results.items()):
        stats = result.stats
        lines.append(
            f"- {date}: input={stats['input']}, accepted={stats['accepted']}, rejected={stats['rejected']}, "
            f"filtered_small={stats['filtered_small']}, repaired={stats['repaired']}, clipped={stats['clipped']}"
        )
    report_path = output / "label_validation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["label_validation_report.md"] = report_path
    return paths
