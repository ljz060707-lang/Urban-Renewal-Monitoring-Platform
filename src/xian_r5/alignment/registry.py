"""Actual-date/source registry for the isolated Xi'an R5 candidate.

The registry is intentionally conservative.  A nominal scene label is kept
for traceability, while the date-audit table supplies actual dates and mixing
severity.  Mixed scenes remain visible but are marked as temporally mixed;
they are not silently collapsed into a clean observation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import sha256_file, utc_now, write_csv, write_json, write_parquet


MAIN_SCENES = {
    "2018": {"scene_id": "xa180429", "nominal_date": "2018-04-29"},
    "2019": {"scene_id": "xa190101", "nominal_date": "2019-01-01"},
    "2020": {"scene_id": "xa201231", "nominal_date": "2020-12-31"},
}

PIXEL_EXTENSIONS = {".tif", ".tiff", ".jp2", ".jpg", ".jpeg", ".png"}
CACHE_EXTENSIONS = {".raw", ".datever", ".ver", ".cryped", ".vrt"}
DATE_EXTENSIONS = {".geid", ".log", ".status", ".dat", ".txt", ".html", ".map", ".tfw", ".bpw"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _scene_paths(main_root: Path, scene_id: str) -> dict[str, Path | None]:
    scene_dir = main_root / scene_id
    combined = main_root / f"{scene_id}_combined" / f"{scene_id}_zoom_20.tif"
    tile_root = scene_dir / "20"
    return {
        "scene_dir": scene_dir,
        "tile_root": tile_root if tile_root.is_dir() else None,
        "pixel_path": combined if combined.is_file() else None,
        "geid_path": main_root / f"{scene_id}.geid",
        "log_path": main_root / f"{scene_id}_log.txt",
    }


def _date_audit_rows(date_audit_root: Path) -> dict[str, list[dict[str, str]]]:
    rows = _read_csv(date_audit_root / "tables" / "date_summary_by_epoch.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("task_name", ""), []).append(row)
    return grouped


def _main_scene_rows(main_root: Path, date_audit_root: Path) -> list[dict[str, Any]]:
    summaries = _date_audit_rows(date_audit_root)
    rows: list[dict[str, Any]] = []
    for epoch, meta in MAIN_SCENES.items():
        scene_id = meta["scene_id"]
        paths = _scene_paths(main_root, scene_id)
        date_rows = summaries.get(scene_id, [])
        known = [r for r in date_rows if r.get("actual_capture_date") not in {"", "DATE_UNKNOWN", None}]
        actual_dates = sorted({r["actual_capture_date"] for r in known})
        dominant = ""
        if known:
            dominant = max(known, key=lambda r: float(r.get("area_share_pct") or 0.0)).get("actual_capture_date", "")
        audit_status = "UNVERIFIED_NO_DATE_AUDIT"
        mixing = "UNKNOWN"
        if date_rows:
            audit_status = "VERIFIED_ACTUAL_DATE_AUDIT"
            if len(actual_dates) <= 1:
                mixing = "none_or_negligible"
            else:
                mixing = "severe"
        unknown_rows = [r for r in date_rows if r.get("actual_capture_date") == "DATE_UNKNOWN"]
        tile_root = paths["tile_root"]
        pixel_path = paths["pixel_path"]
        row = {
            "epoch": epoch,
            "scene_id": scene_id,
            "nominal_date": meta["nominal_date"],
            "actual_date": dominant or None,
            "actual_dates": ";".join(actual_dates),
            "actual_date_status": audit_status,
            "source": "F_XIAN_MAIN",
            "pixel_path": str(pixel_path) if pixel_path else None,
            "tile_root": str(tile_root) if tile_root else None,
            "coverage": "PIXEL_RASTER_AND_TILE_TREE" if pixel_path and tile_root else "PARTIAL_OR_MISSING",
            "quality": "VERIFIED_NO_MIXING" if mixing == "none_or_negligible" else "TEMPORALLY_MIXED" if mixing == "severe" else "UNKNOWN",
            "mixing_severity": mixing,
            "date_unknown_tile_count": sum(int(r.get("tile_count") or 0) for r in unknown_rows),
            "duplicate_group": f"actual-date:{';'.join(actual_dates)}" if actual_dates else None,
            "canonical_observation": True if pixel_path else False,
            "alignment_eligible": bool(pixel_path and tile_root and mixing == "none_or_negligible"),
            "canonicalization_rule": "scene retained; actual-date mixture never imputed",
            "date_evidence_path": str(date_audit_root / "tables" / "date_summary_by_epoch.csv"),
            "source_read_policy": "READ_ONLY",
        }
        rows.append(row)
    return rows


def classify_asset(path: Path, root: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in PIXEL_EXTENSIONS:
        return "REAL_PIXEL_RASTER"
    if suffix in CACHE_EXTENSIONS:
        return "CACHE_METADATA"
    if suffix in DATE_EXTENSIONS or name.endswith(".geid"):
        return "DATE_METADATA"
    return "UNKNOWN"


def classify_supplement_assets(root: Path, *, hash_files: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify supplemental assets without reading raster pixels."""

    root = root.resolve()
    if not root.is_dir():
        return [], {"status": "MISSING_ROOT", "root": str(root)}
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                stat = path.stat()
                asset_class = classify_asset(path, root)
                record: dict[str, Any] = {
                    "relative_path": path.relative_to(root).as_posix(),
                    "asset_class": asset_class,
                    "bytes": stat.st_size,
                    "mtime_utc": stat.st_mtime,
                    "pixel_read": False,
                    "sha256": sha256_file(path) if hash_files else None,
                }
            except OSError as exc:
                record = {
                    "relative_path": str(path),
                    "asset_class": "UNKNOWN",
                    "bytes": None,
                    "mtime_utc": None,
                    "pixel_read": False,
                    "sha256": None,
                    "error": str(exc),
                }
                asset_class = "UNKNOWN"
            rows.append(record)
            counts[asset_class] += 1
    return rows, {
        "status": "PASS_ASSET_CLASSIFIED",
        "root": str(root),
        "file_count": len(rows),
        "class_counts": dict(sorted(counts.items())),
        "hash_files": hash_files,
        "pixel_read": False,
    }


def reuse_supplement_audit(registry_path: Path, output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reuse a completed supplement audit without walking the source tree again.

    The source audit already contains folder inventories, extension classes,
    datever evidence, and duplicate-acquisition evidence.  This function
    writes a compact folder-level pointer for the R5 observation registry and
    preserves the original audit path instead of pretending that a second
    file-level scan was performed.
    """

    if not registry_path.is_file():
        return [], {"status": "MISSING_PRECOMPUTED_SUPPLEMENT_AUDIT", "path": str(registry_path)}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for folder in payload.get("folders", []):
        extension_counts = folder.get("extension_counts") or {}
        classes: list[str] = []
        if int(folder.get("pixel_file_count") or 0) > 0:
            classes.append("REAL_PIXEL_RASTER")
        if any(ext in extension_counts for ext in (".raw", ".ver", ".cryped", ".vrt")):
            classes.append("CACHE_METADATA")
        if any(ext in extension_counts for ext in (".datever", ".dat", ".geid", ".log", ".txt", ".map", ".tfw", ".bpw")):
            classes.append("DATE_METADATA")
        if not classes:
            classes.append("UNKNOWN")
        rows.append({
            "relative_path": folder.get("folder"),
            "asset_class": ";".join(classes),
            "payload_status": folder.get("payload_status"),
            "file_count": folder.get("file_count"),
            "pixel_file_count": folder.get("pixel_file_count"),
            "tile_key_count": folder.get("tile_key_count"),
            "datever_record_count": folder.get("datever_record_count"),
            "actual_dates": ";".join(folder.get("actual_dates") or []),
            "inventory_sha256": folder.get("inventory_sha256"),
            "pixel_read": False,
            "source_audit": str(registry_path),
        })
    receipt = {
        "status": "PASS_REUSED_PRECOMPUTED_SUPPLEMENT_AUDIT",
        "root": payload.get("root"),
        "source_registry": str(registry_path),
        "observation_count": payload.get("observation_count"),
        "tile_date_evidence_count": payload.get("tile_date_evidence_count"),
        "same_acquisition_duplicate_count": payload.get("same_acquisition_duplicate_count"),
        "file_level_scan_in_this_run": False,
        "metadata_is_not_pixel": True,
        "folder_rows": len(rows),
    }
    write_json(output_root / "SUPPLEMENT_AUDIT_REFERENCE.json", receipt)
    return rows, receipt


def build_observation_registry(
    main_root: Path,
    supplement_root: Path,
    date_audit_root: Path,
    output_root: Path,
    *,
    hash_supplement_files: bool = False,
    supplement_registry_path: Path | None = None,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    main_rows = _main_scene_rows(main_root.resolve(), date_audit_root.resolve())
    if supplement_registry_path is not None and supplement_registry_path.is_file():
        supplement_rows, supplement_receipt = reuse_supplement_audit(supplement_registry_path.resolve(), output_root)
    else:
        supplement_rows, supplement_receipt = classify_supplement_assets(supplement_root.resolve(), hash_files=hash_supplement_files)
    out_rows = output_root / "IMAGE_OBSERVATION_REGISTRY.csv"
    out_parquet = output_root / "IMAGE_OBSERVATION_REGISTRY.parquet"
    parquet_result = write_parquet(out_parquet, main_rows)
    write_csv(out_rows, main_rows)
    write_csv(output_root / "SUPPLEMENT_ASSET_CLASSIFICATION.csv", supplement_rows)
    receipt = {
        "schema_version": "xian_r5.image_observation_registry.v1",
        "generated_at_utc": utc_now(),
        "main_root": str(main_root.resolve()),
        "supplement_root": str(supplement_root.resolve()),
        "date_audit_root": str(date_audit_root.resolve()),
        "main_rows": len(main_rows),
        "main_alignment_eligible_rows": sum(bool(row["alignment_eligible"]) for row in main_rows),
        "supplement": supplement_receipt,
        "parquet": parquet_result,
        "policy": {
            "raw_inputs_read_only": True,
            "metadata_is_not_pixel": True,
            "actual_dates_not_imputed": True,
            "mixed_epochs_retained_with_warning": True,
        },
    }
    write_json(output_root / "IMAGE_OBSERVATION_REGISTRY_RECEIPT.json", receipt)
    return {"rows": main_rows, "supplement_rows": supplement_rows, "receipt": receipt}
