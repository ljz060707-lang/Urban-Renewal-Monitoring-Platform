"""Additive, resume-safe post-inference checks for the Xi'an canonical core.

The canonical B0 -> geometry -> tracker -> temporal chain remains the source
of truth.  This module only consumes its outputs and writes a separate
``16_additive_post_inference`` tree.  Missing tracker or reference evidence is
reported as unavailable; it is never silently converted into a state, label,
or error-corrected area estimate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FORMAL_STATES = ("STABLE", "DEMOLITION", "ADDITION")
STATE_ALIASES = {
    "NEW": "ADDITION",
    "ADDED": "ADDITION",
    "DEMOLISHED": "DEMOLITION",
    "UNCHANGED": "STABLE",
}
EVIDENCE_FIELDS = (
    "trajectory_id", "building_id", "primary_state", "actual_dates", "temporal_pattern",
    "temporal_persistence_score", "temporal_precision_days", "geometry_confidence_pre",
    "geometry_confidence_post", "identity_quality", "boundary_jitter_score",
    "DSIFN_change_score", "S2Looking_direction_score", "SECOND_semantic_score",
    "Full20_support_score", "cloud_overlap", "blur_overlap", "seam_overlap",
    "valid_observation_fraction", "area_pre_m2", "area_post_m2", "centroid_shift_m",
    "polygon_iou", "hausdorff_distance_m", "district", "grid_id", "land_context",
    "date_block_quality", "candidate_not_ground_truth",
)
METRIC_CRS = "EPSG:32649"
DATE_RE = re.compile(r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_date(value: Any) -> str | None:
    match = DATE_RE.search(str(value)) if value is not None else None
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError:
        return None


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def zip_directory(source_dir: Path, archive_path: Path) -> dict[str, Any]:
    """Create a deterministic ZIP of an additive package directory.

    The archive is written outside ``source_dir`` and is never included in
    itself. Files are added in lexical order with stable archive metadata so
    repeated resume-safe runs produce equivalent member lists for unchanged
    inputs.
    """

    source_dir = source_dir.resolve()
    archive_path = archive_path.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(archive_path.name + ".part")
    if temporary.exists():
        temporary.unlink()
    files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            relative = path.relative_to(source_dir).as_posix()
            info = zipfile.ZipInfo(relative)
            info.date_time = (2020, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            archive.writestr(info, path.read_bytes())
    temporary.replace(archive_path)
    return {
        "path": str(archive_path),
        "sha256": sha256_file(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "member_count": len(files),
        "status": "available",
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for field in row:
                if field not in seen:
                    fields.append(field)
                    seen.add(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> dict[str, Any]:
    """Write a tabular artifact with a deterministic CSV fallback.

    Parquet is preferred for the evidence table, but a missing optional
    parquet engine must not make the post-inference layer fabricate a result
    or block the canonical package.  The returned descriptor is recorded in
    the package manifest so consumers can follow the fallback explicitly.
    """

    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        if fields:
            for field in fields:
                if field not in frame.columns:
                    frame[field] = None
            frame = frame[list(fields)]
        frame.to_parquet(path, index=False)
        return {"path": str(path), "format": "parquet", "row_count": len(rows), "status": "available"}
    except Exception as error:
        fallback = path.with_suffix(".csv")
        write_csv(fallback, rows, fields)
        return {"path": str(fallback), "requested_path": str(path), "format": "csv_fallback", "row_count": len(rows), "status": "fallback", "reason": repr(error)}


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _score(value: Any, default: float = 0.0) -> float:
    parsed = _float(value, default)
    return float(parsed if parsed is not None else default)


def _ratio(value: Any) -> float | None:
    parsed = _float(value)
    if parsed is None:
        return None
    # Preserve the common 0..1 representation.  Values in 0..100 are only
    # interpreted as percentages when they are clearly outside that range;
    # the original value is never overwritten in the source row.
    if 1.0 < parsed <= 100.0:
        parsed /= 100.0
    return max(0.0, min(1.0, parsed))


def observation_quality_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise available image-quality ratios without inventing evidence."""

    names = ("cloud_ratio", "shadow_ratio", "haze_ratio", "blur_ratio", "seamline_ratio", "nodata_ratio")
    metrics = {name: _ratio(row.get(name)) for name in names}
    supplied_valid = _ratio(row.get("valid_observation_ratio"))
    if supplied_valid is not None:
        metrics["valid_observation_ratio"] = supplied_valid
        valid_source = "supplied"
    elif metrics["nodata_ratio"] is not None:
        metrics["valid_observation_ratio"] = 1.0 - metrics["nodata_ratio"]
        valid_source = "derived_from_nodata_only"
    else:
        metrics["valid_observation_ratio"] = None
        valid_source = "unavailable"
    known = sum(metrics[name] is not None for name in names) + int(metrics["valid_observation_ratio"] is not None)
    metrics["quality_completeness"] = known / (len(names) + 1)
    metrics["quality_status"] = "KNOWN" if known == len(names) + 1 else ("PARTIAL" if known else "UNKNOWN")
    metrics["valid_ratio_source"] = valid_source
    return metrics


def decode_presence_1_0_u(value: Any, *, valid_observation: Any = True, known_search_region: Any = True) -> str:
    """Decode presence conservatively; an unobserved object remains ``U``."""

    if not bool(valid_observation) or not bool(known_search_region):
        return "U"
    if value is None or str(value).strip().upper() in {"", "U", "UNKNOWN", "NAN", "NONE"}:
        return "U"
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"1", "TRUE", "PRESENT", "YES", "BUILDING"}:
            return "1"
        if normalized in {"0", "FALSE", "ABSENT", "NO", "MISSING"}:
            return "0"
    try:
        return "1" if float(value) > 0 else "0"
    except (TypeError, ValueError):
        return "U"


def temporal_pattern_for_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Build identity patterns from explicit presence/state fields only."""

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        identity = str(row.get("trajectory_id") or row.get("track_id") or row.get("building_id") or row.get("id") or "")
        if not identity:
            continue
        actual = parse_date(row.get("actual_date") or row.get("scene_date") or row.get("date"))
        if not actual:
            continue
        explicit = row.get("presence")
        if explicit is None:
            state = str(row.get("state") or row.get("primary_state") or "").upper()
            explicit = "1" if state in {"PRESENT", "STABLE", "NEW", "ADDITION", "BUILDING"} else ("0" if state in {"ABSENT", "DEMOLITION", "DEMOLISHED"} else "U")
        grouped[identity].append((actual, decode_presence_1_0_u(explicit, valid_observation=row.get("valid_observation", True), known_search_region=row.get("known_search_region", True))))
    patterns = {}
    for identity, observations in grouped.items():
        by_date = {actual: value for actual, value in observations}
        patterns[identity] = "".join(by_date[actual] for actual in sorted(by_date))
    return patterns


def _sample_key(row: Mapping[str, Any]) -> str:
    value = row.get("sample_key") or row.get("trajectory_id") or row.get("track_id") or row.get("building_id") or row.get("id")
    if value is not None:
        return str(value)
    return hashlib.sha256(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _observation_date_status(row: Mapping[str, Any], actual_date: str | None) -> str:
    """Classify an observation without treating a nominal year as evidence."""

    if not actual_date:
        return "DATE_UNKNOWN"
    actual_year = int(actual_date[:4])
    nominal = row.get("nominal_year") or row.get("nominal_epoch") or row.get("task_year")
    try:
        nominal_year = int(str(nominal)[:4]) if nominal is not None else None
    except (TypeError, ValueError):
        nominal_year = None
    # With no nominal year to compare, the actual acquisition is still a core
    # date observation.  A disagreement is retained as an off-date observation
    # rather than being silently relabelled to the nominal epoch.
    return "CORE_DATE" if nominal_year is None or nominal_year == actual_year else "OFF_DATE_VALID"


def fold_actual_observations(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Collapse identical real acquisitions while preserving provenance.

    A single actual date is not an interval.  ``DUPLICATE_EPOCH`` and
    ``TEMPORAL_INSUFFICIENT`` therefore never imply STABLE or no change.
    """

    normalized = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        row["actual_date"] = parse_date(row.get("actual_date") or row.get("scene_date") or row.get("date"))
        row["date_status"] = _observation_date_status(row, row["actual_date"])
        row["_input_order"] = index
        normalized.append(row)
    valid_dates = sorted({row["actual_date"] for row in normalized if row.get("actual_date")})
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        if row.get("actual_date"):
            by_date[str(row["actual_date"])].append(row)
    effective: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for actual_date in valid_dates:
        candidates = sorted(
            by_date[actual_date],
            key=lambda row: (-_score(row.get("quality_score") or row.get("valid_observation_ratio"), 0.0), str(row.get("source") or row.get("stem") or ""), int(row["_input_order"])),
        )
        effective_row = {key: value for key, value in candidates[0].items() if key != "_input_order"}
        effective_row["date_status"] = "DUPLICATE_EPOCH" if len(candidates) > 1 else effective_row.get("date_status", "CORE_DATE")
        effective_row["duplicate_count"] = max(0, len(candidates) - 1)
        effective.append(effective_row)
        for duplicate in candidates[1:]:
            duplicate_row = {key: value for key, value in duplicate.items() if key != "_input_order"}
            duplicate_row["date_status"] = "DUPLICATE_EPOCH"
            duplicate_row["duplicate_of_actual_date"] = actual_date
            duplicates.append(duplicate_row)
    if not valid_dates or any(row.get("actual_date") is None for row in normalized):
        status = "TEMPORAL_INSUFFICIENT"
    elif len(valid_dates) == 1:
        status = "DUPLICATE_EPOCH" if len(normalized) > 1 else "TEMPORAL_INSUFFICIENT"
    else:
        status = "OFF_DATE_VALID"
    return {
        "status": status,
        "actual_dates": valid_dates,
        "effective_epoch_count": len(valid_dates),
        "effective_observations": effective,
        "duplicate_observations": duplicates,
        "unknown_date_count": sum(row.get("actual_date") is None for row in normalized),
        "genuine_interval": len(valid_dates) >= 2,
        "date_status_counts": dict(Counter(row.get("date_status") for row in normalized)),
        "nominal_year_is_never_evidence": True,
    }


def _to_geometry(value: Any):
    from shapely.geometry import shape
    from shapely import wkt
    from shapely import wkb

    if value is None:
        return None
    if hasattr(value, "geom_type"):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return wkb.loads(bytes(value))
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return wkt.loads(value)
        except Exception:
            try:
                try:
                    return wkb.loads(bytes.fromhex(value))
                except Exception:
                    return shape(json.loads(value))
            except Exception:
                return None
    if isinstance(value, Mapping):
        try:
            return shape(value)
        except Exception:
            return None
    return None


def compute_boundary_jitter(previous: Any, current: Any) -> dict[str, Any]:
    """Return deterministic polygon boundary jitter metrics in source units."""

    old = _to_geometry(previous)
    new = _to_geometry(current)
    if old is None or new is None or old.is_empty or new.is_empty or old.area <= 0 or new.area <= 0:
        return {"status": "UNAVAILABLE_INVALID_GEOMETRY", "boundary_jitter_score": None}
    try:
        intersection = old.intersection(new)
        union = old.union(new)
        iou = float(intersection.area / union.area) if union.area else 0.0
        symmetric_difference_ratio = float(old.symmetric_difference(new).area / union.area) if union.area else 1.0
        scale = max(math.sqrt(max(old.area, new.area) / math.pi), 1e-9)
        hausdorff = float(old.hausdorff_distance(new))
        centroid_shift = float(old.centroid.distance(new.centroid))
        area_change = abs(float(new.area - old.area)) / max(float(old.area), 1e-9)
        perimeter_change = abs(float(new.length - old.length)) / max(float(old.length), 1e-9)
        min_perimeter = max(min(float(old.length), float(new.length)), 1e-9)
        boundary_overlap = float(old.boundary.intersection(new.boundary).length / min_perimeter)
        jitter = (
            0.40 * (1.0 - max(0.0, min(1.0, iou)))
            + 0.20 * max(0.0, min(1.0, symmetric_difference_ratio))
            + 0.15 * max(0.0, min(1.0, hausdorff / scale))
            + 0.15 * max(0.0, min(1.0, centroid_shift / scale))
            + 0.10 * max(0.0, min(1.0, area_change))
        )
        return {
            "status": "OK",
            "polygon_iou": iou,
            "symmetric_difference_ratio": symmetric_difference_ratio,
            "hausdorff_distance_m": hausdorff,
            "centroid_shift_m": centroid_shift,
            "area_change_ratio": area_change,
            "perimeter_change_ratio": perimeter_change,
            "boundary_overlap_ratio": max(0.0, min(1.0, boundary_overlap)),
            "boundary_jitter_score": max(0.0, min(1.0, float(jitter))),
        }
    except Exception as error:  # pragma: no cover - defensive geometry boundary
        return {"status": "UNAVAILABLE_GEOMETRY_ERROR", "boundary_jitter_score": None, "error": str(error)}


def confidence_tier_from_evidence(evidence: Mapping[str, Any]) -> str:
    """Map evidence strength to a tier; formal states remain unchanged."""

    persistence = _score(evidence.get("temporal_persistence_score"), 0.0)
    identity = _score(evidence.get("identity_quality"), 0.0)
    jitter = _score(evidence.get("boundary_jitter_score"), 1.0)
    valid = _score(evidence.get("valid_observation_fraction"), 0.0)
    model_agreement = _score(evidence.get("model_agreement"), 0.0)
    if persistence >= 0.90 and identity >= 0.90 and jitter <= 0.10 and valid >= 0.90:
        return "CONSERVATIVE"
    if persistence >= 0.65 and identity >= 0.65 and valid >= 0.65 and (model_agreement >= 0.5 or jitter <= 0.25):
        return "MAIN"
    return "EXTENDED"


def deterministic_validation_sample(rows: Iterable[Mapping[str, Any]], calibration_keys: Iterable[str] | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """Select a reproducible independent sample, excluding calibration keys."""

    excluded = {str(value) for value in (calibration_keys or ())}
    candidates = [dict(row) for row in rows if _sample_key(row) not in excluded]
    candidates.sort(key=lambda row: _sample_key(row))
    return candidates[: max(0, int(limit))]


def build_area_error_matrix(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Estimate area corrections only when reference labels and weights exist."""

    records = [dict(row) for row in rows]
    labeled = [row for row in records if (row.get("reference_state") or row.get("ground_truth_state")) and row.get("predicted_state")]
    if not labeled:
        return {
            "status": "NOT_ESTIMABLE_NO_REFERENCE_LABELS",
            "matrix": {},
            "mapped_area_m2": {},
            "error_adjusted_area_m2": None,
            "ci95": None,
            "formal_states": list(FORMAL_STATES),
        }
    missing_weight = [row for row in labeled if _float(row.get("sample_weight") or row.get("design_weight")) is None]
    matrix = {reference: {predicted: 0 for predicted in FORMAL_STATES} for reference in FORMAL_STATES}
    mapped = Counter()
    for row in labeled:
        reference = str(row.get("reference_state") or row.get("ground_truth_state"))
        predicted = str(row.get("predicted_state"))
        if reference in matrix and predicted in matrix[reference]:
            matrix[reference][predicted] += 1
        if predicted in FORMAL_STATES:
            mapped[predicted] += _score(row.get("area_m2"), 0.0)
    if missing_weight:
        return {
            "status": "REFERENCE_LABELS_WITHOUT_DESIGN_WEIGHTS",
            "matrix": matrix,
            "mapped_area_m2": dict(mapped),
            "error_adjusted_area_m2": None,
            "ci95": None,
            "formal_states": list(FORMAL_STATES),
        }
    adjusted = Counter()
    variance = Counter()
    for row in labeled:
        reference = str(row.get("reference_state") or row.get("ground_truth_state"))
        if reference not in FORMAL_STATES:
            continue
        contribution = _score(row.get("area_m2"), 0.0) * _score(row.get("sample_weight") or row.get("design_weight"), 0.0)
        adjusted[reference] += contribution
        variance[reference] += contribution * contribution
    ci = {state: [max(0.0, adjusted[state] - 1.96 * math.sqrt(variance[state])), adjusted[state] + 1.96 * math.sqrt(variance[state])] for state in FORMAL_STATES}
    return {
        "status": "ESTIMABLE_WITH_REFERENCE_AND_DESIGN_WEIGHTS",
        "matrix": matrix,
        "mapped_area_m2": dict(mapped),
        "error_adjusted_area_m2": {state: float(adjusted[state]) for state in FORMAL_STATES},
        "ci95": ci,
        "formal_states": list(FORMAL_STATES),
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []


def _source_rows(canonical_run: Path) -> list[dict[str, Any]]:
    selection = canonical_run / "00_provenance" / "GEOMETRY_SOURCE_SELECTION.csv"
    rows = _read_csv(selection) if selection.is_file() else []
    inventory_path = canonical_run / "00_provenance" / "SOURCE_PROVENANCE.json"
    provenance = read_json(inventory_path, {}) or {}
    inventory_candidates = (
        canonical_run.parent / "00_audit" / "SOURCE_INVENTORY.json",
        canonical_run / "00_audit" / "SOURCE_INVENTORY.json",
        canonical_run.parent / "assembly" / "00_audit" / "SOURCE_INVENTORY.json",
    )
    inventory = next((path for path in inventory_candidates if path.is_file()), inventory_candidates[0])
    inv_by_stem = {}
    payload = read_json(inventory, {}) or {}
    for record in payload.get("records", []) if isinstance(payload, dict) else []:
        if record.get("stem"):
            inv_by_stem[str(record["stem"])] = record
    merged = []
    for row in rows:
        stem = str(row.get("stem") or "")
        source = inv_by_stem.get(stem, {})
        actual = parse_date(row.get("actual_date") or row.get("scene_date") or source.get("actual_date") or source.get("scene_date"))
        merged.append({**source, **row, "stem": stem, "actual_date": actual})
    if not merged and isinstance(provenance.get("actual_date_counts"), dict):
        for actual, count in sorted(provenance["actual_date_counts"].items()):
            merged.append({"stem": f"actual_{actual}", "actual_date": parse_date(actual), "source_count": count})
    return merged


def _find_tracker_table(canonical_run: Path, exclude_roots: Iterable[Path] = ()):
    try:
        import pandas as pd
    except Exception:
        return None, "PANDAS_UNAVAILABLE"
    candidates = []
    excluded = [path.resolve() for path in exclude_roots]
    for path in sorted(canonical_run.rglob("*")):
        resolved = path.resolve()
        if any(resolved == root or root in resolved.parents for root in excluded):
            continue
        if not path.is_file() or any(token in path.name.lower() for token in ("test", "holdout")):
            continue
        if path.suffix.lower() not in {".parquet", ".csv"}:
            continue
        lowered = path.name.lower()
        if not any(token in lowered for token in ("track", "trajectory", "evidence", "state")):
            continue
        try:
            frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        except Exception:
            continue
        columns = {str(column).lower() for column in frame.columns}
        if columns.intersection({"track_id", "trajectory_id", "building_id"}) and columns.intersection({"state", "primary_state", "presence", "actual_date", "date"}):
            candidates.append((path, frame))
    if not candidates:
        return None, "NOT_AVAILABLE_NO_TRACKER_ARTIFACT"
    path, frame = candidates[0]
    return frame, str(path)


def _frame_records(frame: Any) -> list[dict[str, Any]]:
    if frame is None:
        return []
    try:
        return frame.to_dict(orient="records")
    except Exception:
        return [dict(row) for row in frame]


def _tracker_jitter(frame: Any) -> tuple[list[dict[str, Any]], str]:
    records = _frame_records(frame)
    if not records:
        return [], "NOT_AVAILABLE_NO_TRACKER_ROWS"
    id_key = next((key for key in ("trajectory_id", "track_id", "building_id", "id") if key in records[0]), None)
    geometry_key = next((key for key in ("geometry", "geom", "wkt") if key in records[0]), None)
    date_key = next((key for key in ("actual_date", "date", "scene_date") if key in records[0]), None)
    if not id_key or not geometry_key:
        return [], "NOT_AVAILABLE_TRACKER_SCHEMA"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get(id_key))].append(row)
    output = []
    for identity, items in sorted(grouped.items()):
        items.sort(key=lambda row: parse_date(row.get(date_key)) if date_key and parse_date(row.get(date_key)) else "")
        for previous, current in zip(items, items[1:]):
            metrics = compute_boundary_jitter(previous.get(geometry_key), current.get(geometry_key))
            output.append({"identity_id": identity, "date_previous": parse_date(previous.get(date_key)) if date_key else None, "date_current": parse_date(current.get(date_key)) if date_key else None, **metrics})
    return output, "OK"


def _evidence_records(frame: Any, jitter_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = _frame_records(frame)
    if not records:
        return []
    patterns = temporal_pattern_for_rows(records)
    jitter_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jitter_rows:
        jitter_by_id[str(row["identity_id"])].append(row)
    output = []
    for row in records:
        item = dict(row)
        identity = str(item.get("trajectory_id") or item.get("track_id") or item.get("building_id") or item.get("id") or "")
        item["trajectory_id"] = identity
        raw_state = str(item.get("primary_state") or item.get("state") or "").upper()
        item["primary_state"] = STATE_ALIASES.get(raw_state, raw_state if raw_state in FORMAL_STATES else "UNKNOWN")
        item["temporal_pattern"] = item.get("temporal_pattern") or patterns.get(identity)
        actual = parse_date(item.get("actual_date") or item.get("scene_date") or item.get("date"))
        item["actual_dates"] = item.get("actual_dates") or (actual if actual else None)
        matches = jitter_by_id.get(identity, [])
        if matches and item.get("boundary_jitter_score") is None:
            item["boundary_jitter_score"] = sum(_score(match.get("boundary_jitter_score"), 1.0) for match in matches) / len(matches)
        for field in EVIDENCE_FIELDS:
            item.setdefault(field, None)
        item["sample_key"] = item.get("sample_key") or f"{identity}|{actual or 'unknown'}"
        item["confidence_tier"] = confidence_tier_from_evidence(item)
        item["candidate_not_ground_truth"] = True
        output.append(item)
    return output


def _fixed_grid(row: Mapping[str, Any], grid_size: float = 0.05) -> str:
    bounds = row.get("bounds") or row.get("source_bounds")
    if isinstance(bounds, str):
        try:
            bounds = json.loads(bounds)
        except Exception:
            bounds = None
    try:
        left, bottom, right, top = [float(value) for value in bounds]
        lon = (left + right) / 2
        lat = (bottom + top) / 2
        return f"g_{math.floor(lon / grid_size):05d}_{math.floor(lat / grid_size):05d}"
    except (TypeError, ValueError):
        return "grid_unknown"


def build_date_block_shift(rows: Iterable[Mapping[str, Any]], grid_size: float = 0.05) -> list[dict[str, Any]]:
    """Compute a source/object density audit; it never deletes an object."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        actual = parse_date(row.get("actual_date"))
        if actual:
            grouped[(actual, _fixed_grid(row, grid_size))].append(row)
    by_date_values: dict[str, list[float]] = defaultdict(list)
    output = []
    for (actual, grid), members in sorted(grouped.items()):
        count = sum(_score(row.get("object_count") or row.get("building_count"), 1.0) for row in members)
        area = sum(_score(row.get("building_area_m2") or row.get("area_m2") or row.get("exclusive_area_m2"), 0.0) for row in members)
        footprint = sum(_score(row.get("grid_area_m2") or row.get("scene_area_m2") or row.get("valid_area_m2"), 0.0) for row in members)
        quality = [_float(row.get("valid_observation_ratio") or row.get("quality_ratio")) for row in members]
        quality = [value for value in quality if value is not None]
        mean_quality = sum(quality) / len(quality) if quality else None
        by_date_values[actual].append(count)
        low_conf = [_float(row.get("low_confidence_ratio")) for row in members]
        low_conf = [value for value in low_conf if value is not None]
        addition = [_float(row.get("candidate_addition_rate")) for row in members]
        addition = [value for value in addition if value is not None]
        demolition = [_float(row.get("candidate_demolition_rate")) for row in members]
        demolition = [value for value in demolition if value is not None]
        output.append({"actual_date": actual, "grid_id": grid, "building_count_density": count, "building_footprint_ratio": (area / footprint if footprint > 0 else None), "building_area_m2_source_proxy": area, "mean_geometry_confidence": _float(members[0].get("geometry_confidence")), "low_confidence_ratio": (sum(low_conf) / len(low_conf) if low_conf else None), "candidate_addition_rate": (sum(addition) / len(addition) if addition else None), "candidate_demolition_rate": (sum(demolition) / len(demolition) if demolition else None), "quality_ratio": mean_quality, "date_block_model_shift": False, "decision": "AUDIT_ONLY_NO_DELETION", "scope": "source_proxy"})
    if len(by_date_values) >= 2:
        date_means = {actual: sum(values) / len(values) for actual, values in by_date_values.items()}
        baseline = sorted(date_means.values())[len(date_means) // 2]
        for row in output:
            value = date_means.get(row["actual_date"], baseline)
            row["date_block_model_shift"] = bool(baseline and abs(value - baseline) / baseline >= 0.50)
    return output


class AdditivePostInferenceOrchestrator:
    """Run S00-S25 as append-only, stage-receipted post-processing."""

    STAGES = (
        "S00_INFERENCE_GATE", "S01_PROVENANCE", "S02_ACTUAL_DATE", "S03_OBSERVATION_QUALITY", "S04_DATE_BLOCK_SHIFT",
        "S05_GEOMETRY_CANONICALIZATION", "S06_BOUNDARY_JITTER", "S07_TRACKER", "S08_PRESENCE_1_0_U", "S09_V8R8",
        "S10_MULTIMODEL_EVIDENCE", "S11_STATE_DECODER", "S12_EVENT_CLUSTERING", "S13_AREA_ACCOUNTING", "S14_CONFIDENCE_TIER",
        "S15_CALIBRATION_SAMPLE", "S16_INDEPENDENT_VALIDATION", "S17_ERROR_MATRIX", "S18_ERROR_ADJUSTED_AREA_CI",
        "S19_CITY_DISTRICT_STATS", "S20_MAPS", "S21_HEATMAPS", "S22_CLASSIC_CASES", "S23_VISUAL_QA", "S24_ROOT_CAUSE_FIX_LOOP", "S25_FINAL_PACKAGE",
    )

    def __init__(self, canonical_run: Path, output_run: Path | None = None, calibration: Path | None = None) -> None:
        self.canonical_run = canonical_run.resolve()
        self.output_run = (output_run or self.canonical_run / "16_additive_post_inference").resolve()
        self.calibration = calibration.resolve() if calibration else None
        self.receipts = self.output_run / "receipts"
        self.source_rows = _source_rows(self.canonical_run)
        self.tracker_frame = None
        self.tracker_source = "not_loaded"
        self.jitter_rows: list[dict[str, Any]] = []
        self.evidence_rows: list[dict[str, Any]] = []

    def _restore_completed_state(self) -> None:
        """Reload in-memory state needed when resuming after a restart."""

        if read_json(self._receipt_path("S07_TRACKER"), {}).get("status") == "complete":
            self.tracker_frame, self.tracker_source = _find_tracker_table(self.canonical_run, (self.output_run,))
        jitter_path = self._stage_dir("S06_BOUNDARY_JITTER") / "BOUNDARY_JITTER.csv"
        if read_json(self._receipt_path("S06_BOUNDARY_JITTER"), {}).get("status") == "complete":
            self.jitter_rows = _read_csv(jitter_path)
        evidence_path = self._stage_dir("S14_CONFIDENCE_TIER") / "BUILDING_EVIDENCE_TABLE.csv"
        if read_json(self._receipt_path("S14_CONFIDENCE_TIER"), {}).get("status") == "complete":
            self.evidence_rows = _read_csv(evidence_path)
        elif read_json(self._receipt_path("S11_STATE_DECODER"), {}).get("status") == "complete":
            self.evidence_rows = _read_csv(self._stage_dir("S11_STATE_DECODER") / "STATE_DECODER_INPUT.csv")

    def _write_progress(self, *, current_stage: str | None, completed: Sequence[str], status: str = "running", error: str | None = None) -> None:
        completed = list(completed)
        next_stage = next((stage for stage in self.STAGES if stage not in completed), None)
        write_json(self.output_run / "CURRENT_PROGRESS.json", {"schema_version": "urrs.xian.additive.progress.v1", "status": status, "canonical_run": str(self.canonical_run), "output_run": str(self.output_run), "current_stage": current_stage, "completed_stages": completed, "next_stage": next_stage, "updated_at": now_iso(), "error": error, "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}})

    def _stage_dir(self, stage: str) -> Path:
        return self.output_run / stage.lower().replace("_", "-")

    def _receipt_path(self, stage: str) -> Path:
        return self.receipts / f"{stage}_COMPLETE.json"

    def _complete(self, stage: str, payload: Mapping[str, Any]) -> None:
        stage_dir = self._stage_dir(stage)
        output_files: list[str] = []
        output_sha256: dict[str, str] = {}
        if stage_dir.is_dir():
            for path in sorted(path for path in stage_dir.rglob("*") if path.is_file()):
                relative = str(path.relative_to(self.output_run)).replace("\\", "/")
                output_files.append(relative)
                digest = sha256_file(path)
                if digest:
                    output_sha256[relative] = digest
        stage_index = self.STAGES.index(stage) if stage in self.STAGES else -1
        input_receipts: list[str] = []
        canonical_receipt = self.canonical_run / "receipts" / "FINAL_TEMPORAL_PACKAGE_RECEIPT.json"
        if canonical_receipt.is_file():
            input_receipts.append(str(canonical_receipt))
        if stage_index > 0:
            previous = self._receipt_path(self.STAGES[stage_index - 1])
            if previous.is_file():
                input_receipts.append(str(previous))
        receipt = {
            "schema_version": "urrs.xian.additive.stage.v1",
            "stage": stage,
            "status": "complete",
            "created_at": now_iso(),
            "input_receipts": input_receipts,
            "output_files": output_files,
            "output_sha256": output_sha256,
            "validation": {"status": "recorded", "output_file_count": len(output_files)},
            **dict(payload),
        }
        # Compute the non-empty check from the already collected paths before
        # serialising the receipt; no files outside this additive run are read
        # for validation.
        stage_paths = [self.output_run / Path(item) for item in output_files]
        receipt["validation"] = {
            "status": "recorded",
            "output_file_count": len(output_files),
            "all_output_files_nonempty": all(path.stat().st_size > 0 for path in stage_paths) if stage_paths else True,
        }
        write_json(self._receipt_path(stage), receipt)

    def _fail(self, stage: str, error: Exception) -> None:
        payload = {"schema_version": "urrs.xian.additive.failure.v1", "stage": stage, "status": "blocked", "created_at": now_iso(), "error": repr(error), "canonical_run": str(self.canonical_run), "candidate_not_ground_truth": True}
        unique = self.receipts / f"FAILURE_RECEIPT_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        write_json(unique, payload)
        stable = self.receipts / "FAILURE_RECEIPT.json"
        if not stable.exists():
            write_json(stable, payload | {"first_failure_receipt": str(unique)})

    def run(self) -> dict[str, Any]:
        self.output_run.mkdir(parents=True, exist_ok=True)
        self._restore_completed_state()
        stage_methods = {stage: getattr(self, f"_{stage.lower()}") for stage in self.STAGES}
        completed_stages = [stage for stage in self.STAGES if read_json(self._receipt_path(stage), {}).get("status") == "complete"]
        self._write_progress(current_stage=None, completed=completed_stages, status="resuming" if completed_stages else "starting")
        for stage in self.STAGES:
            receipt = read_json(self._receipt_path(stage), {}) or {}
            if receipt.get("status") == "complete":
                continue
            self._write_progress(current_stage=stage, completed=completed_stages)
            try:
                stage_methods[stage]()
            except Exception as error:  # fail closed and retain evidence
                self._fail(stage, error)
                self._write_progress(current_stage=stage, completed=completed_stages, status="blocked", error=repr(error))
                return {"status": "blocked", "failed_stage": stage, "error": repr(error), "output_run": str(self.output_run)}
            completed_stages.append(stage)
            self._write_progress(current_stage=None, completed=completed_stages, status="stage_complete")
        s25_receipt = read_json(self._receipt_path("S25_FINAL_PACKAGE"), {}) or {}
        package_descriptor = s25_receipt.get("package") if isinstance(s25_receipt.get("package"), Mapping) else {}
        final = {"status": "complete", "created_at": now_iso(), "canonical_run": str(self.canonical_run), "output_run": str(self.output_run), "formal_states": list(FORMAL_STATES), "candidate_not_ground_truth": True, "scientific_status": "candidate_not_ground_truth", "fusion_started": False, "package": package_descriptor.get("path"), "package_archive": package_descriptor.get("archive"), "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}}
        write_json(self.output_run / "ADDITIVE_POST_INFERENCE_RECEIPT.json", final)
        self._write_progress(current_stage=None, completed=completed_stages, status="complete")
        return final

    def _s00_inference_gate(self) -> None:
        receipt_path = self.canonical_run / "receipts" / "FINAL_TEMPORAL_PACKAGE_RECEIPT.json"
        receipt = read_json(receipt_path, {}) or {}
        if receipt.get("status") != "pass":
            raise RuntimeError(f"canonical final package is not complete: {receipt_path}")
        if receipt.get("coverage_status") not in {None, "complete"}:
            raise RuntimeError(f"canonical coverage is not complete: {receipt.get('coverage_status')}")
        if receipt.get("missing_geometry_stems"):
            raise RuntimeError(f"canonical geometry is incomplete: {receipt.get('missing_geometry_stems')}")
        test_reads = receipt.get("test_reads") or {}
        if any(_score(test_reads.get(key), 0.0) != 0.0 for key in ("WHU", "SECOND", "Holdout")):
            raise RuntimeError(f"protected split was read: {test_reads}")
        if receipt.get("raw_inputs_modified") or receipt.get("model_weights_modified"):
            raise RuntimeError("canonical receipt reports protected asset modification")
        self._complete("S00_INFERENCE_GATE", {"canonical_receipt": str(receipt_path), "canonical_receipt_sha256": sha256_file(receipt_path), "all_inference_complete": True, "coverage_status": receipt.get("coverage_status", "complete"), "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}})

    def _s01_provenance(self) -> None:
        folded = fold_actual_observations(self.source_rows)
        duplicate_stems = {str(row.get("stem")) for row in folded.get("duplicate_observations", [])}
        duplicate_dates = {str(row.get("actual_date")) for row in folded.get("effective_observations", []) if _score(row.get("duplicate_count"), 0.0) > 0}
        effective_stems = {str(row.get("stem")) for row in folded.get("effective_observations", [])}
        provenance_rows = []
        for row in self.source_rows:
            item = dict(row)
            stem = str(item.get("stem"))
            item["date_status"] = "DUPLICATE_EPOCH" if stem in duplicate_stems or str(item.get("actual_date")) in duplicate_dates else ("CORE_DATE" if stem in effective_stems else _observation_date_status(item, item.get("actual_date")))
            item["effective_observation"] = stem in effective_stems
            source_value = item.get("source_tif") or item.get("source_image") or item.get("source_path")
            if source_value:
                source_path = Path(str(source_value))
                item["source_sha256"] = sha256_file(source_path) if source_path.is_file() else None
            provenance_rows.append(item)
        write_csv(self._stage_dir("S01_PROVENANCE") / "SOURCE_ROWS.csv", provenance_rows)
        write_json(self._stage_dir("S01_PROVENANCE") / "ACTUAL_DATE_EPOCHS.json", folded)
        self._complete("S01_PROVENANCE", {"source_row_count": len(self.source_rows), "actual_date_status": folded["status"], "effective_epoch_count": folded["effective_epoch_count"], "nominal_year_is_never_evidence": True})

    def _s02_actual_date(self) -> None:
        folded = fold_actual_observations(self.source_rows)
        write_json(self._stage_dir("S02_ACTUAL_DATE") / "DATE_FOLD.json", folded)
        date_rows = folded.get("effective_observations", []) + folded.get("duplicate_observations", [])
        write_csv(self._stage_dir("S02_ACTUAL_DATE") / "DATE_STATUS.csv", date_rows, fields=(list(date_rows[0]) if date_rows else ["stem", "actual_date", "date_status"]))
        self._complete("S02_ACTUAL_DATE", {"status": folded["status"], "actual_dates": folded["actual_dates"], "duplicate_count": len(folded["duplicate_observations"]), "temporal_insufficient": not folded["genuine_interval"], "date_status_counts": folded.get("date_status_counts", {})})

    def _s03_observation_quality(self) -> None:
        rows = []
        for row in self.source_rows:
            quality = observation_quality_metrics(row)
            rows.append({"stem": row.get("stem"), "actual_date": row.get("actual_date"), **quality, "quality_status": quality["quality_status"]})
        write_csv(self._stage_dir("S03_OBSERVATION_QUALITY") / "OBSERVATION_QUALITY.csv", rows)
        self._complete("S03_OBSERVATION_QUALITY", {"row_count": len(rows), "unknown_quality_fields_preserved": True, "derived_valid_ratio_only_from_nodata": True})

    def _s04_date_block_shift(self) -> None:
        rows = build_date_block_shift(self.source_rows)
        write_csv(self._stage_dir("S04_DATE_BLOCK_SHIFT") / "DATE_BLOCK_SHIFT.csv", rows)
        self._complete("S04_DATE_BLOCK_SHIFT", {"row_count": len(rows), "model_shift_count": sum(bool(row.get("date_block_model_shift")) for row in rows), "policy": "lower confidence only; no object deletion"})

    def _s05_geometry_canonicalization(self) -> None:
        manifest_count = len(list(self.canonical_run.rglob("GEOMETRY_MANIFEST.json")))
        exact_tables = [str(path) for path in self.canonical_run.glob("05_temporal_outputs/*THREE_STATE_EXACT.csv") if path.is_file()]
        exact_rows = []
        for table in exact_tables:
            exact_rows.extend(_read_csv(Path(table)))
        invalid_area_count = sum(1 for row in exact_rows if row.get("area_m2") is not None and _float(row.get("area_m2")) is None)
        summary = {"metric_crs": METRIC_CRS, "geometry_manifest_count": manifest_count, "exact_state_tables": exact_tables, "exact_area_row_count": len(exact_rows), "invalid_area_count": invalid_area_count, "canonical_geometry_authority": "B0/RSBuilding accepted path", "raster_xor_is_not_area": True}
        write_json(self._stage_dir("S05_GEOMETRY_CANONICALIZATION") / "CANONICAL_GEOMETRY_AUDIT.json", summary)
        self._complete("S05_GEOMETRY_CANONICALIZATION", summary)

    def _s06_boundary_jitter(self) -> None:
        self.tracker_frame, self.tracker_source = _find_tracker_table(self.canonical_run, (self.output_run,))
        self.jitter_rows, status = _tracker_jitter(self.tracker_frame)
        write_csv(self._stage_dir("S06_BOUNDARY_JITTER") / "BOUNDARY_JITTER.csv", self.jitter_rows)
        write_table(self._stage_dir("S06_BOUNDARY_JITTER") / "BOUNDARY_JITTER.parquet", self.jitter_rows)
        self._complete("S06_BOUNDARY_JITTER", {"status": status, "tracker_source": self.tracker_source, "row_count": len(self.jitter_rows), "metric_crs": METRIC_CRS})

    def _s07_tracker(self) -> None:
        if self.tracker_frame is None:
            self.tracker_frame, self.tracker_source = _find_tracker_table(self.canonical_run, (self.output_run,))
        trajectory_rows = _frame_records(self.tracker_frame)
        trajectory_table = write_table(self._stage_dir("S07_TRACKER") / "BUILDING_TRAJECTORIES.parquet", trajectory_rows)
        self._complete("S07_TRACKER", {"status": "available" if self.tracker_frame is not None else "deferred_no_tracker_artifact", "tracker_source": self.tracker_source, "identity_authority": "canonical Building Tracker only", "trajectory_table": trajectory_table})

    def _s08_presence_1_0_u(self) -> None:
        records = _frame_records(self.tracker_frame)
        decoded = []
        for row in records:
            item = dict(row)
            item["presence_1_0_u"] = decode_presence_1_0_u(item.get("presence") if item.get("presence") is not None else item.get("state"), valid_observation=item.get("valid_observation", True), known_search_region=item.get("known_search_region", True))
            decoded.append(item)
        write_csv(self._stage_dir("S08_PRESENCE_1_0_U") / "PRESENCE_1_0_U.csv", decoded)
        self._complete("S08_PRESENCE_1_0_U", {"status": "available" if records else "deferred_no_tracker_artifact", "zero_requires_valid_observation": True, "unknown_preserved": True, "decoded_row_count": len(decoded)})

    def _s09_v8r8(self) -> None:
        patterns = temporal_pattern_for_rows(_frame_records(self.tracker_frame))
        write_csv(self._stage_dir("S09_V8R8") / "TEMPORAL_PATTERNS.csv", [{"identity_id": identity, "temporal_pattern": pattern, "is_gap_return_diagnostic": "101" in pattern, "is_flicker_diagnostic": "010" in pattern} for identity, pattern in sorted(patterns.items())])
        self._complete("S09_V8R8", {"status": "diagnostic_only", "canonical_temporal_reasoner": "V8R8 when present", "gap_return_not_demolition": True, "pattern_count": len(patterns)})

    def _s10_multimodel_evidence(self) -> None:
        self._complete("S10_MULTIMODEL_EVIDENCE", {"status": "supporting_evidence_only", "models": ["DSIFN", "S2Looking", "SECOND", "Full20"], "identity_or_polygon_mutation": False})

    def _s11_state_decoder(self) -> None:
        self.evidence_rows = _evidence_records(self.tracker_frame, self.jitter_rows)
        write_csv(self._stage_dir("S11_STATE_DECODER") / "STATE_DECODER_INPUT.csv", self.evidence_rows)
        self._complete("S11_STATE_DECODER", {"status": "available" if self.evidence_rows else "deferred_no_tracker_artifact", "formal_states": list(FORMAL_STATES), "internal_diagnostic_states": ["GAP_RETURN", "TEMPORARY", "REBUILD", "YEAR_UNCERTAIN", "TEMPORAL_INSUFFICIENT", "UNOBSERVED"]})

    def _s12_event_clustering(self) -> None:
        self._complete("S12_EVENT_CLUSTERING", {"status": "deferred_no_object_events" if not self.evidence_rows else "diagnostic_object_events", "identity_unit": "building object", "pixel_neighborhood_is_not_identity": True})

    def _s13_area_accounting(self) -> None:
        exact = list(self.canonical_run.glob("05_temporal_outputs/*THREE_STATE_EXACT.csv"))
        rows = []
        for path in exact:
            rows.extend(_read_csv(path))
        write_csv(self._stage_dir("S13_AREA_ACCOUNTING") / "CANONICAL_EXACT_AREA_ROWS.csv", rows)
        canonical_receipt = read_json(self.canonical_run / "receipts" / "FINAL_TEMPORAL_PACKAGE_RECEIPT.json", {}) or {}
        self._complete("S13_AREA_ACCOUNTING", {"status": "canonical_rows_reused", "row_count": len(rows), "area_crs": METRIC_CRS, "area_source": "canonical polygon boundary intersections", "outside_city_excluded": True, "boundary_clipping_evidence": canonical_receipt.get("boundary_audit") or "canonical_exact_state_table_contract"})

    def _s14_confidence_tier(self) -> None:
        rows = self.evidence_rows or []
        counts = Counter(row.get("confidence_tier") for row in rows)
        write_csv(self._stage_dir("S14_CONFIDENCE_TIER") / "BUILDING_EVIDENCE_TABLE.csv", rows)
        parquet = write_table(self._stage_dir("S14_CONFIDENCE_TIER") / "BUILDING_EVIDENCE_TABLE.parquet", rows)
        self._complete("S14_CONFIDENCE_TIER", {"status": "available" if rows else "deferred_no_tracker_artifact", "tier_counts": dict(counts), "formal_states_unchanged": True, "table": parquet})

    def _s15_calibration_sample(self) -> None:
        calibration_rows = _read_csv(self.calibration) if self.calibration and self.calibration.is_file() else []
        keys = [_sample_key(row) for row in calibration_rows]
        write_csv(self._stage_dir("S15_CALIBRATION_SAMPLE") / "CALIBRATION_SAMPLE.csv", calibration_rows)
        self._complete("S15_CALIBRATION_SAMPLE", {"status": "available" if calibration_rows else "not_provided", "row_count": len(calibration_rows), "calibration_keys_sha256": hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest() if keys else None, "no_threshold_tuning": True})

    def _s16_independent_validation(self) -> None:
        calibration_rows = _read_csv(self.calibration) if self.calibration and self.calibration.is_file() else []
        keys = [_sample_key(row) for row in calibration_rows]
        rows = deterministic_validation_sample(self.evidence_rows, keys, 200)
        for row in rows:
            state = str(row.get("primary_state") or row.get("state") or "").upper()
            row["validation_stratum"] = "NO_CHANGE_BUILDING" if state in {"STABLE", "UNCHANGED"} else ("CHANGE_CANDIDATE" if state in {"DEMOLITION", "ADDITION", "NEW", "DEMOLISHED"} else "UNCLASSIFIED")
        write_csv(self._stage_dir("S16_INDEPENDENT_VALIDATION") / "INDEPENDENT_VALIDATION_SAMPLE.csv", rows)
        strata = Counter(row.get("validation_stratum") for row in rows)
        self._complete("S16_INDEPENDENT_VALIDATION", {"status": "unlabeled_proxy_only" if rows else "deferred_no_tracker_artifact", "row_count": len(rows), "disjoint_from_calibration": not set(keys).intersection({_sample_key(row) for row in rows}), "no_change_regions_required": True, "no_change_building_count": strata.get("NO_CHANGE_BUILDING", 0), "no_change_nonbuilding_status": "NOT_AVAILABLE_NO_REFERENCE_LABELS"})

    def _s17_error_matrix(self) -> None:
        rows = _read_csv(self._stage_dir("S16_INDEPENDENT_VALIDATION") / "INDEPENDENT_VALIDATION_SAMPLE.csv")
        result = build_area_error_matrix(rows)
        write_json(self._stage_dir("S17_ERROR_MATRIX") / "AREA_ERROR_MATRIX.json", result)
        matrix_rows = []
        for reference in FORMAL_STATES:
            for predicted in FORMAL_STATES:
                matrix_rows.append({"reference_state": reference, "predicted_state": predicted, "count": (result.get("matrix") or {}).get(reference, {}).get(predicted, 0)})
        write_csv(self._stage_dir("S17_ERROR_MATRIX") / "AREA_ERROR_MATRIX.csv", matrix_rows)
        self._complete("S17_ERROR_MATRIX", result)

    def _s18_error_adjusted_area_ci(self) -> None:
        result = read_json(self._stage_dir("S17_ERROR_MATRIX") / "AREA_ERROR_MATRIX.json", {}) or {}
        write_json(self._stage_dir("S18_ERROR_ADJUSTED_AREA_CI") / "AREA_ADJUSTED_ESTIMATES.json", {"status": result.get("status", "not_estimable"), "error_adjusted_area_m2": result.get("error_adjusted_area_m2"), "ci95": result.get("ci95"), "raw_canonical_area_preserved": True})
        self._complete("S18_ERROR_ADJUSTED_AREA_CI", {"status": result.get("status", "not_estimable"), "raw_canonical_area_preserved": True})

    def _s19_city_district_stats(self) -> None:
        source = self.canonical_run / "05_temporal_outputs"
        rows = _read_csv(source / "XIAN_CITY_THREE_STATE_EXACT.csv") + _read_csv(source / "XIAN_COUNTY_THREE_STATE_EXACT.csv")
        write_csv(self._stage_dir("S19_CITY_DISTRICT_STATS") / "CITY_DISTRICT_STATS.csv", rows)
        self._complete("S19_CITY_DISTRICT_STATS", {"status": "canonical_boundary_tables_reused", "row_count": len(rows), "metric_crs": METRIC_CRS, "outside_city_excluded": True})

    def _s20_maps(self) -> None:
        existing = sorted(str(path) for path in self.canonical_run.glob("06_visuals/*"))
        required = ("DATE_COVERAGE", "DATE_COMPARISON", "T0_BUILDING_OUTLINES", "ALL_STATES", "TEMPORAL_RELIABILITY_MAP", "BOUNDARY_JITTER_MAP", "MODEL_DISAGREEMENT_MAP", "DATE_BLOCK_SHIFT_MAP", "OBSERVATION_QUALITY_MAP")
        index = {name: {"status": "available" if any(name.lower() in Path(item).name.lower() for item in existing) else "deferred_no_canonical_artifact", "matches": [item for item in existing if name.lower() in Path(item).name.lower()]} for name in required}
        write_json(self._stage_dir("S20_MAPS") / "MAP_INDEX.json", {"status": "reused_canonical_maps", "maps": existing, "required_products": index, "basemap_policy": "actual source TIF/overview only"})
        self._complete("S20_MAPS", {"status": "reused_canonical_maps", "map_count": len(existing), "required_product_status": {key: value["status"] for key, value in index.items()}})

    def _s21_heatmaps(self) -> None:
        existing = sorted(str(path) for path in self.canonical_run.glob("06_visuals/*HEAT*"))
        required = ("demolition_count_heatmap", "demolition_area_heatmap", "addition_count_heatmap", "addition_area_heatmap", "total_change_heatmap")
        index = {name: {"status": "available" if any(name.lower() in Path(item).name.lower() for item in existing) else "deferred_no_canonical_artifact", "matches": [item for item in existing if name.lower() in Path(item).name.lower()]} for name in required}
        write_json(self._stage_dir("S21_HEATMAPS") / "HEATMAP_INDEX.json", {"status": "reused_canonical_heatmaps" if existing else "deferred_no_heatmap_artifact", "heatmaps": existing, "required_products": index, "grid_policy": "canonical fixed metric grid; no re-tuning from results"})
        self._complete("S21_HEATMAPS", {"status": "reused_canonical_heatmaps" if existing else "deferred_no_heatmap_artifact", "heatmaps": existing, "required_product_status": {key: value["status"] for key, value in index.items()}})

    def _s22_classic_cases(self) -> None:
        case_files = sorted(str(path) for path in (self.canonical_run / "08_cases").glob("*"))
        write_json(self._stage_dir("S22_CLASSIC_CASES") / "CASE_INDEX.json", {"status": "reused_canonical_cases" if case_files else "deferred_no_case_artifact", "files": case_files, "candidate_not_ground_truth": True})
        self._complete("S22_CLASSIC_CASES", {"status": "reused_canonical_cases" if case_files else "deferred_no_case_artifact", "case_file_count": len(case_files), "candidate_not_ground_truth": True})

    def _s23_visual_qa(self) -> None:
        figures = [path for path in self.canonical_run.glob("06_visuals/*") if path.is_file()]
        checks = []
        for path in figures:
            item = {"path": str(path), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else 0, "readable": False}
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"} and item["size_bytes"] > 0:
                try:
                    from PIL import Image

                    with Image.open(path) as image:
                        image.verify()
                    item["readable"] = True
                except Exception as error:
                    item["read_error"] = repr(error)
            else:
                # Vector/PDF existence and non-zero size are the only safe
                # checks without rendering or changing the canonical figures.
                item["readable"] = item["size_bytes"] > 0
            checks.append(item)
        qa_status = "pass" if checks and all(item["size_bytes"] > 0 and item["readable"] for item in checks) else ("deferred" if not checks else "failed_readability")
        write_json(self._stage_dir("S23_VISUAL_QA") / "VISUAL_QA.json", {"status": qa_status, "checks": checks, "candidate_not_ground_truth": True})
        self._complete("S23_VISUAL_QA", {"status": qa_status if checks else "deferred_no_figures", "figure_count": len(checks)})

    def _s24_root_cause_fix_loop(self) -> None:
        self._complete("S24_ROOT_CAUSE_FIX_LOOP", {"status": "no_automatic_model_fix", "policy": "diagnose -> owning module -> bounded repair; canonical weights and thresholds unchanged"})

    def _write_final_package(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        """Materialise a small, self-describing additive package.

        Large canonical rasters and vector layers are intentionally not copied.
        The package stores explicit pointers and only writes derived tables and
        receipts, so rerunning this layer cannot mutate or fork the canonical
        B0/tracker outputs.
        """

        package = self.output_run / "FINAL_XIAN_PACKAGE"
        for name in ("00_provenance", "01_actual_dates", "02_quality", "03_geometry", "04_tracking", "05_temporal", "06_states", "07_evidence", "08_area", "09_validation", "10_statistics", "11_maps", "12_heatmaps", "13_classic_cases", "14_visual_qa", "15_receipt"):
            (package / name).mkdir(parents=True, exist_ok=True)

        folded = fold_actual_observations(self.source_rows)
        write_json(package / "01_actual_dates" / "ACTUAL_DATE_EPOCHS.json", folded)
        write_csv(package / "01_actual_dates" / "DATE_OBSERVATIONS.csv", folded.get("effective_observations", []) + folded.get("duplicate_observations", []))
        reliability_rows = [{"actual_date": row.get("actual_date"), "stem": row.get("stem"), "date_status": row.get("date_status"), "effective": True, "temporal_reliability": "VALID_TEMPORAL" if folded.get("genuine_interval") else "TEMPORAL_INSUFFICIENT"} for row in folded.get("effective_observations", [])]
        reliability_rows.extend({"actual_date": row.get("actual_date"), "stem": row.get("stem"), "date_status": "DUPLICATE_EPOCH", "effective": False, "temporal_reliability": "DUPLICATE_EPOCH"} for row in folded.get("duplicate_observations", []))
        write_csv(package / "01_actual_dates" / "temporal_reliability_summary.csv", reliability_rows)

        quality_rows = _read_csv(self._stage_dir("S03_OBSERVATION_QUALITY") / "OBSERVATION_QUALITY.csv")
        write_csv(package / "02_quality" / "observation_quality_summary.csv", quality_rows)

        exact_rows = _read_csv(self._stage_dir("S13_AREA_ACCOUNTING") / "CANONICAL_EXACT_AREA_ROWS.csv")
        write_table(package / "08_area" / "canonical_exact_area.parquet", exact_rows)
        write_csv(package / "08_area" / "canonical_exact_area.csv", exact_rows)

        evidence_descriptor = write_table(package / "07_evidence" / "building_evidence_table.parquet", self.evidence_rows)
        write_csv(package / "07_evidence" / "building_evidence_table.csv", self.evidence_rows)
        trajectory_rows = _frame_records(self.tracker_frame)
        trajectory_descriptor = write_table(package / "04_tracking" / "building_trajectories.parquet", trajectory_rows)

        validation_rows = _read_csv(self._stage_dir("S16_INDEPENDENT_VALIDATION") / "INDEPENDENT_VALIDATION_SAMPLE.csv")
        write_table(package / "09_validation" / "independent_validation_sample.parquet", validation_rows)
        write_csv(package / "09_validation" / "independent_validation_sample.csv", validation_rows)

        matrix = read_json(self._stage_dir("S17_ERROR_MATRIX") / "AREA_ERROR_MATRIX.json", {}) or {}
        matrix_rows = [
            {"reference_state": reference, "predicted_state": predicted, "count": (matrix.get("matrix") or {}).get(reference, {}).get(predicted, 0)}
            for reference in FORMAL_STATES
            for predicted in FORMAL_STATES
        ]
        write_csv(package / "08_area" / "area_error_matrix.csv", matrix_rows)
        adjusted = matrix.get("error_adjusted_area_m2")
        ci = matrix.get("ci95") or {}
        adjusted_rows = [{"state": state, "raw_or_mapped_area_m2": (matrix.get("mapped_area_m2") or {}).get(state), "error_adjusted_area_m2": (adjusted or {}).get(state) if isinstance(adjusted, Mapping) else None, "ci95_low": ci.get(state, [None, None])[0] if isinstance(ci.get(state), list) else None, "ci95_high": ci.get(state, [None, None])[1] if isinstance(ci.get(state), list) else None, "status": matrix.get("status")} for state in FORMAL_STATES]
        write_csv(package / "08_area" / "area_adjusted_estimates.csv", adjusted_rows)

        stats_rows = _read_csv(self._stage_dir("S19_CITY_DISTRICT_STATS") / "CITY_DISTRICT_STATS.csv")
        city_rows = []
        district_rows = []
        for row in stats_rows:
            marker = " ".join(str(row.get(key, "")) for key in ("scope", "level", "region_type", "admin_level", "name", "region"))
            if any(token in marker.lower() for token in ("county", "district", "county_summary", "区", "县")):
                district_rows.append(row)
            elif any(token in marker.lower() for token in ("city", "市", "xian")):
                city_rows.append(row)
        if stats_rows and not city_rows and not district_rows:
            city_rows = stats_rows
        write_csv(package / "10_statistics" / "city_summary.csv", city_rows)
        write_csv(package / "10_statistics" / "district_summary.csv", district_rows)

        canonical_receipt = self.canonical_run / "receipts" / "FINAL_TEMPORAL_PACKAGE_RECEIPT.json"
        canonical_pointer = {"path": str(canonical_receipt), "sha256": sha256_file(canonical_receipt), "immutable": True}
        unavailable = {
            "stable.gpkg": "not_materialized_by_additive_layer; use canonical state outputs",
            "demolition.gpkg": "not_materialized_by_additive_layer; use canonical state outputs",
            "addition.gpkg": "not_materialized_by_additive_layer; use canonical state outputs",
            "gap_return_1_0_1.gpkg": "diagnostic-only internal state; no formal fourth category",
            "renewal_events.gpkg": "not_materialized_by_additive_layer; canonical event package remains authoritative",
        }
        expected_core = (
            "all_buildings.gpkg", "stable.gpkg", "demolition.gpkg", "addition.gpkg", "gap_return_1_0_1.gpkg",
            "building_trajectories.parquet", "building_evidence_table.parquet", "building_events.csv", "renewal_events.gpkg",
            "city_summary.csv", "district_summary.csv", "observation_quality_summary.csv", "temporal_reliability_summary.csv",
            "area_error_matrix.csv", "area_adjusted_estimates.csv",
        )
        canonical_index = {}
        for name in expected_core:
            matches = [path for path in self.canonical_run.rglob(name) if path.is_file()]
            canonical_index[name] = {"status": "available_pointer" if matches else "not_available_in_canonical_run", "paths": [str(path) for path in matches[:20]]}
        artifact_manifest = {
            "building_evidence_table": evidence_descriptor,
            "building_trajectories": trajectory_descriptor,
            "canonical_exact_area": {"path": str(package / "08_area" / "canonical_exact_area.parquet"), "row_count": len(exact_rows), "status": "available" if exact_rows else "empty_canonical_table"},
            "city_summary": {"path": str(package / "10_statistics" / "city_summary.csv"), "row_count": len(city_rows), "status": "available" if city_rows else "unavailable_no_city_rows"},
            "district_summary": {"path": str(package / "10_statistics" / "district_summary.csv"), "row_count": len(district_rows), "status": "available" if district_rows else "unavailable_no_district_rows"},
            "canonical_core_artifacts": canonical_index,
            "unavailable_or_pointer_only": unavailable,
        }
        package_manifest = {
            "schema_version": "urrs.xian.final.package.v1",
            "canonical_core": {"run": str(self.canonical_run), "final_receipt": canonical_pointer, "geometry_authority": "B0/RSBuilding accepted path", "identity_authority": "canonical Building Tracker", "formal_states": list(FORMAL_STATES)},
            "additive_run": str(self.output_run),
            "artifacts": artifact_manifest,
            "actual_date_policy": "nominal year is never evidence; identical actual dates fold into one effective epoch",
            "area_policy": "canonical polygon boundary intersections in EPSG:32649; no raster XOR/bbox area",
            "candidate_not_ground_truth": True,
            "fusion_started": False,
            "stage_receipts": {stage: {"path": str(self._receipt_path(stage)), "sha256": sha256_file(self._receipt_path(stage))} for stage in self.STAGES if self._receipt_path(stage).is_file()},
            "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        }
        write_json(package / "00_provenance" / "CANONICAL_CORE_POINTERS.json", package_manifest["canonical_core"])
        write_json(package / "PACKAGE_MANIFEST.json", package_manifest)
        final_receipt = {**dict(summary), "schema_version": "urrs.xian.final.package.v1", "status": "complete", "package_manifest": str(package / "PACKAGE_MANIFEST.json"), "canonical_receipt_sha256": canonical_pointer["sha256"], "candidate_not_ground_truth": True, "fusion_started": False, "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}}
        write_json(package / "FINAL_XIAN_RECEIPT.json", final_receipt)
        write_json(package / "15_receipt" / "FINAL_XIAN_RECEIPT.json", final_receipt)
        report = [
            "# Xi'an additive final package",
            "",
            "The canonical B0 -> geometry -> tracker -> V8R8 -> 1/0/U chain remains authoritative.",
            "This package adds date reliability, quality, jitter, evidence, validation and area-audit artifacts without changing canonical polygons, model weights or formal state classes.",
            "",
            f"- Canonical run: `{self.canonical_run}`",
            f"- Actual-date status: **{folded['status']}**; effective epochs: **{folded['effective_epoch_count']}**",
            f"- Tracker source: `{self.tracker_source}`; evidence rows: **{len(self.evidence_rows)}**",
            f"- Area error status: **{matrix.get('status', 'not_estimable')}**",
            "- Formal user-facing states: **STABLE / DEMOLITION / ADDITION**.",
            "- `DUPLICATE_EPOCH`, `TEMPORAL_INSUFFICIENT`, `GAP_RETURN` and `U` remain diagnostic/reliability states, not added formal classes.",
            "- No Test/Holdout was read; no model weights, raw imagery or canonical outputs were modified.",
            "- `candidate_not_ground_truth=true`; adjusted area and confidence claims require reference labels and design weights.",
        ]
        (package / "FINAL_XIAN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        return {"path": str(package), "manifest": str(package / "PACKAGE_MANIFEST.json"), "receipt": str(package / "FINAL_XIAN_RECEIPT.json"), "artifact_manifest": artifact_manifest}

    def _s25_final_package(self) -> None:
        summary = {"canonical_run": str(self.canonical_run), "additive_run": str(self.output_run), "source_row_count": len(self.source_rows), "tracker_source": self.tracker_source, "evidence_row_count": len(self.evidence_rows), "formal_states": list(FORMAL_STATES), "candidate_not_ground_truth": True, "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}}
        package = self._write_final_package(summary)
        summary["package"] = package
        write_json(self._stage_dir("S25_FINAL_PACKAGE") / "ADDITIVE_SUMMARY.json", summary)
        (self._stage_dir("S25_FINAL_PACKAGE") / "ADDITIVE_REPORT.md").write_text("# Xi'an additive post-inference layer\n\nCanonical geometry, tracker identity, and exact boundary-clipped areas remain authoritative. Missing tracker/reference evidence is reported as unavailable; no state or adjusted area is fabricated.\n", encoding="utf-8")
        archive = zip_directory(Path(package["path"]), self.output_run / "FINAL_XIAN_PACKAGE.zip")
        package["archive"] = archive
        summary["package"] = package
        write_json(self._stage_dir("S25_FINAL_PACKAGE") / "ADDITIVE_SUMMARY.json", summary)
        write_json(self.output_run / "FINAL_XIAN_PACKAGE_ARCHIVE.json", {"schema_version": "urrs.xian.final.package.archive.v1", "package": package["path"], "archive": archive, "candidate_not_ground_truth": True, "fusion_started": False, "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}})
        self._complete("S25_FINAL_PACKAGE", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = AdditivePostInferenceOrchestrator(args.canonical_run, args.output_run, args.calibration).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
