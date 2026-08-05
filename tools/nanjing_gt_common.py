"""Shared, read-only helpers for the future Nanjing manual-GT workflow."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VALID_DATES = {
    "2015-01-04",
    "2016-08-18",
    "2017-12-11",
    "2018-02-08",
    "2019-03-16",
    "2022-12-02",
}
EXCLUDED_DATES = {"2024-10-20"}
GT_DEFAULT = Path(r"D:\Data Set\Satellite\nanjing_jiangbei\GT\nanjing_manual_gt_v1.gpkg")
LAYER_SPECS: dict[str, dict[str, Any]] = {
    "gt_building": {
        "required": {"gt_id", "date", "split", "label_status", "source", "reviewer_id", "reviewed_at", "geometry_quality"},
        "allowed_status": {"human_verified", "human_corrected", "uncertain", "ignore"},
    },
    "gt_construction": {
        "required": {"gt_id", "date", "split", "label_status", "construction_class", "source", "reviewer_id", "reviewed_at", "geometry_quality"},
        "allowed_status": {"human_verified", "human_corrected", "uncertain", "ignore"},
        "allowed_class": {"background", "exposed_ground", "dustproof_net", "building", "other_construction"},
    },
    "gt_evaluation_extent": {
        "required": {"extent_id", "date", "split", "complete_flag", "source", "reviewer_id", "reviewed_at", "notes"},
        "allowed_status": set(),
    },
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _crs_is_epsg32650(crs: Any, crs_wkt: Any) -> bool:
    try:
        from pyproj import CRS  # type: ignore

        candidate = CRS.from_user_input(crs or crs_wkt)
        return candidate.to_epsg() == 32650
    except Exception:
        value = str(crs or "") + " " + str(crs_wkt or "")
        return "32650" in value


def _geometry_status(feature: dict[str, Any]) -> tuple[str, str]:
    geometry = feature.get("geometry") or {}
    kind = geometry.get("type", "") if isinstance(geometry, dict) else ""
    if kind not in {"Polygon", "MultiPolygon"}:
        return kind or "missing", "invalid_type"
    try:
        from shapely.geometry import shape  # type: ignore

        geom = shape(geometry)
        if geom.is_empty:
            return kind, "empty"
        if not geom.is_valid:
            return kind, "invalid"
    except ImportError:
        return kind, "not_checked_shapely_unavailable"
    except Exception:
        return kind, "invalid"
    return kind, "valid"


def _read_layers(path: Path) -> tuple[list[str], list[str]]:
    try:
        import fiona  # type: ignore

        return list(fiona.listlayers(str(path))), []
    except ImportError:
        return [], ["fiona is required to inspect an existing GeoPackage"]
    except Exception as exc:
        return [], [f"cannot list GeoPackage layers: {exc!r}"]


def iter_layer_features(path: Path, layer: str) -> Iterable[dict[str, Any]]:
    import fiona  # type: ignore

    with fiona.open(str(path), layer=layer) as source:
        for feature in source:
            yield {
                "properties": dict(feature.get("properties") or {}),
                "geometry": feature.get("geometry"),
            }


def inspect_gt(path: Path) -> dict[str, Any]:
    """Inspect a user-owned GeoPackage without changing it."""

    base: dict[str, Any] = {
        "created_at": now_utc(),
        "gt_path": str(path.resolve()),
        "gt_modified": False,
        "training_started": False,
        "inference_started": False,
        "locked_test_pixel_reads": 0,
        "human_ground_truth_available": False,
        "metrics_allowed": False,
    }
    if not path.exists():
        return {
            **base,
            "status": "awaiting_manual_annotation",
            "reason": "manual GT GeoPackage is not present; no empty replacement was created",
            "layers": [],
            "errors": [],
        }
    layers, errors = _read_layers(path)
    if errors:
        return {**base, "status": "GT_VALIDATION_ERROR", "layers": layers, "errors": errors}
    result: dict[str, Any] = {**base, "status": "GT_VALIDATION_ERROR", "layers": layers, "layer_audits": {}, "errors": []}
    for layer_name, spec in LAYER_SPECS.items():
        if layer_name not in layers:
            result["errors"].append(f"missing required layer: {layer_name}")
            continue
        try:
            import fiona  # type: ignore

            with fiona.open(str(path), layer=layer_name) as source:
                fields = set((source.schema or {}).get("properties", {}).keys())
                crs_ok = _crs_is_epsg32650(source.crs, getattr(source, "crs_wkt", ""))
                geometry_types: dict[str, int] = {}
                feature_count = 0
                invalid_geometry_count = 0
                empty_geometry_count = 0
                ids: list[str] = []
                accepted_count = 0
                excluded_count = 0
                complete_count = 0
                for raw in source:
                    feature_count += 1
                    feature = {"properties": dict(raw.get("properties") or {}), "geometry": raw.get("geometry")}
                    props = feature["properties"]
                    geom_type, geom_status = _geometry_status(feature)
                    geometry_types[geom_type] = geometry_types.get(geom_type, 0) + 1
                    invalid_geometry_count += geom_status in {"invalid", "invalid_type"}
                    empty_geometry_count += geom_status == "empty"
                    if layer_name == "gt_evaluation_extent":
                        identifier = str(props.get("extent_id", ""))
                        complete_count += str(props.get("complete_flag", "0")).lower() in {"1", "true", "yes"}
                    else:
                        identifier = str(props.get("gt_id", ""))
                        status = str(props.get("label_status", ""))
                        if status in {"human_verified", "human_corrected"}:
                            accepted_count += 1
                        elif status in {"uncertain", "ignore"}:
                            excluded_count += 1
                        elif status:
                            result["errors"].append(f"{layer_name}: unsupported label_status={status}")
                        if layer_name == "gt_construction" and props.get("construction_class") not in spec["allowed_class"]:
                            result["errors"].append(f"{layer_name}: unsupported construction_class={props.get('construction_class')}")
                    ids.append(identifier)
                    date = str(props.get("date", ""))
                    if date not in VALID_DATES:
                        result["errors"].append(f"{layer_name}: invalid or excluded date={date}")
                    if str(props.get("split", "")) not in {"train", "validation", "holdout"}:
                        result["errors"].append(f"{layer_name}: invalid split={props.get('split')}")
                missing = sorted(spec["required"] - fields)
                if missing:
                    result["errors"].append(f"{layer_name}: missing fields={missing}")
                if len(ids) != len(set(ids)):
                    result["errors"].append(f"{layer_name}: duplicate identifiers")
                if invalid_geometry_count or empty_geometry_count:
                    result["errors"].append(f"{layer_name}: invalid_or_empty_geometry")
                result["layer_audits"][layer_name] = {
                    "feature_count": feature_count,
                    "fields": sorted(fields),
                    "crs_ok_epsg32650": crs_ok,
                    "geometry_types": geometry_types,
                    "invalid_geometry_count": invalid_geometry_count,
                    "empty_geometry_count": empty_geometry_count,
                    "accepted_count": accepted_count,
                    "excluded_count": excluded_count,
                    "complete_extent_count": complete_count,
                }
                if not crs_ok:
                    result["errors"].append(f"{layer_name}: CRS is not EPSG:32650")
        except Exception as exc:
            result["errors"].append(f"{layer_name}: read error={exc!r}")
    extent_audit = result.get("layer_audits", {}).get("gt_evaluation_extent", {})
    result["metrics_allowed"] = not result["errors"] and extent_audit.get("complete_extent_count", 0) > 0
    result["human_ground_truth_available"] = not bool(result["errors"])
    result["status"] = "GT_VALIDATION_PASS" if not result["errors"] else "GT_VALIDATION_ERROR"
    return result


def accepted_features(path: Path, layer: str) -> list[dict[str, Any]]:
    result = []
    for feature in iter_layer_features(path, layer):
        props = feature["properties"]
        if props.get("label_status") in {"human_verified", "human_corrected"}:
            result.append(props)
    return result

