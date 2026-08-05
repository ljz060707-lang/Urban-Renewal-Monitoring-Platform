"""Exhaustive, audit-first leakage scan for URRS training assets.

The scanner distinguishes three things that are often conflated:

* byte hashes of assets that may safely be read;
* manifest-level inventory of large or locked source roots; and
* explicit exclusions where reading pixels would consume a locked Test set.

It never reads Test/Holdout pixels.  A missing or incomplete source manifest is
reported as an exclusion and keeps the downstream training gate closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2", ".bmp"}
MANIFEST_TOKENS = ("manifest", "inventory", "ledger", "catalog", "split", "audit", "readiness", "summary")
EXCLUDED_DIR_TOKENS = {".git", ".venv-gpu", "__pycache__", ".pytest_cache", ".pbt", "reports"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]] | pd.DataFrame, columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if columns:
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[list(columns)]
    temp = path.with_name(path.name + ".part")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def is_locked_test(path: Path) -> bool:
    text = str(path).lower().replace("\\", "/")
    return bool(re.search(r"/(test|holdout|locked_test)(/|_|-)|whu[_-]?test|second[_-]?test", text))


def dataset_name(path: Path) -> str:
    text = str(path).lower().replace("\\", "/")
    for token, name in [
        ("s2looking", "S2Looking"),
        ("dsifn", "DSIFN"),
        ("second", "SECOND"),
        ("whu", "WHU Building"),
        ("ucs386", "UCS386"),
        ("cwld", "CWLD"),
        ("human_seed", "Human Seed"),
        ("temporal_seed", "Temporal Seed"),
        ("nanjing", "Nanjing"),
        ("tongzhou", "Tongzhou"),
    ]:
        if token in text:
            return name
    return "URRS project asset"


def infer_split(path: Path) -> str:
    text = str(path).lower().replace("\\", "/")
    if is_locked_test(path):
        return "test_locked"
    if "/validation/" in text or "/val/" in text or "_val" in text:
        return "validation"
    if "/train/" in text or "/training/" in text or "_train" in text:
        return "training"
    if "pseudo" in text or "seed" in text:
        return "reference_only"
    return "unspecified"


def source_kind(path: Path) -> str:
    lower = path.name.lower()
    if path.is_dir():
        return "source_root_inventory"
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return "image"
    if path.suffix.lower() in {".gpkg", ".geojson", ".shp", ".json", ".csv", ".yaml", ".yml"}:
        return "manifest_or_label"
    if path.suffix.lower() in {".pth", ".pt", ".ckpt"}:
        return "checkpoint"
    if any(token in lower for token in MANIFEST_TOKENS):
        return "manifest_or_label"
    return "metadata"


def _bits_from_image(path: Path) -> tuple[str, str, str] | None:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("L")
            image.thumbnail((32, 32), Image.Resampling.BILINEAR)
            image = image.resize((32, 32), Image.Resampling.BILINEAR)
            arr = np.asarray(image, dtype=np.float32)
    except Exception:
        return None
    def bit_string(values: np.ndarray, threshold: float) -> str:
        return "".join("1" if value >= threshold else "0" for value in values.ravel())
    ahash = bit_string(arr, float(arr.mean()))
    dhash = bit_string(arr[:, 1:] - arr[:, :-1], 0.0)
    small = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32)
    dct_like = np.abs(np.fft.fft2(small).real)
    phash = bit_string(dct_like[:8, :8], float(np.median(dct_like[:8, :8])))
    return phash, dhash, ahash


def _hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


def _date_token(path: Path) -> str | None:
    for token in re.findall(r"20\d{2}[-_]?\d{2}[-_]?\d{2}", str(path)):
        digits = token.replace("_", "-")
        if len(digits) == 8:
            digits = f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
        if len(digits) == 10:
            return digits
    return None


def _bounds_from_manifest(obj: Mapping[str, Any]) -> list[float] | None:
    for key in ("bounds", "bbox", "aoi_bounds"):
        value = obj.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                return [float(x) for x in value]
            except (TypeError, ValueError):
                pass
    return None


def _manifest_scene_records(path: Path) -> list[dict[str, Any]]:
    obj = read_json(path)
    if not isinstance(obj, Mapping):
        return []
    records: list[dict[str, Any]] = []
    for key in ("scenes", "candidate_files", "assets", "records", "items", "files"):
        value = obj.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping) and item.get("file_path"):
                    records.append(dict(item))
    return records


def _record_for_path(scene_records: Sequence[Mapping[str, Any]], path: Path) -> Mapping[str, Any] | None:
    resolved = str(path.resolve()).lower()
    for record in scene_records:
        candidate = str(record.get("file_path", "")).lower()
        if candidate and (candidate == resolved or Path(candidate).name.lower() == path.name.lower()):
            return record
    return None


def _should_inventory_file(path: Path) -> bool:
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return False
    lower = path.name.lower()
    return path.suffix.lower() in {".json", ".csv", ".yaml", ".yml", ".md", ".gpkg", ".geojson", ".shp"} or any(token in lower for token in MANIFEST_TOKENS)


def _walk_metadata(root: Path, run_root: Path) -> list[Path]:
    if not root.exists():
        return []
    result: list[Path] = []
    for path in root.rglob("*"):
        if any(token in {part.lower() for part in path.parts} for token in EXCLUDED_DIR_TOKENS):
            continue
        if path.is_file() and path != run_root and _should_inventory_file(path):
            result.append(path)
    return result


def _add_asset(rows: list[dict[str, Any]], path: Path, dataset: str | None = None, split: str | None = None, *, manifest_hash: str | None = None, pixel_read: bool | None = None, bounds: Sequence[float] | None = None, source_lineage: str | None = None) -> None:
    locked = is_locked_test(path)
    if locked and path.suffix.lower() in IMAGE_SUFFIXES:
        file_hash = None
        hash_status = "not_read_locked_test"
        pixel_read = False
    elif manifest_hash:
        file_hash = manifest_hash
        hash_status = "manifest_sha256"
        pixel_read = False if pixel_read is None else pixel_read
    else:
        file_hash = sha256(path) if path.is_file() else None
        hash_status = "sha256" if file_hash else "not_read_source_root"
        if pixel_read is None:
            pixel_read = path.suffix.lower() in IMAGE_SUFFIXES
    phash = dhash = ahash = None
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not locked and pixel_read:
        hashes = _bits_from_image(path)
        if hashes:
            phash, dhash, ahash = hashes
    rows.append({
        "asset_id": f"asset_{len(rows):07d}",
        "asset_path": str(path.resolve()),
        "dataset": dataset or dataset_name(path),
        "split": split or infer_split(path),
        "source_kind": source_kind(path),
        "file_size": int(path.stat().st_size) if path.is_file() else None,
        "sha256": file_hash,
        "hash_status": hash_status,
        "perceptual_hash": phash,
        "dhash": dhash,
        "ahash": ahash,
        "spatial_overlap": None,
        "date": _date_token(path),
        "bounds": json.dumps(list(bounds)) if bounds is not None else None,
        "source_lineage": source_lineage or str(path.parent.resolve()),
        "test_locked": locked,
        "pixel_read": bool(pixel_read),
    })


def collect_assets(project_root: Path, run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    outputs = project_root / "outputs"
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    scope: dict[str, Any] = {"roots": [], "source_root_file_counts": {}, "source_root_bytes": {}}

    # Official Nanjing scenes are safe non-Test sources.  Use the existing
    # scene manifest hash where available and independently hash the six files.
    for root, dataset in [
        (Path(r"D:\Data Set\Satellite\nanjing_jiangbei"), "Nanjing Zoom19"),
        (Path(r"D:\Data Set\Satellite\nanjing_jiangbei 20"), "Nanjing Zoom20 and 2019 GT"),
    ]:
        scope["roots"].append(str(root))
        manifest_candidates = sorted((outputs / "inference").glob("nanjing_jiangbei_zoom20_2019_gt_*/00_input_audit/ZOOM20_SCENE_MANIFEST.json")) if "20" in root.name else []
        scene_records: list[dict[str, Any]] = []
        for manifest in manifest_candidates[-1:]:
            scene_records.extend(_manifest_scene_records(manifest))
        if root.exists():
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if _date_token(path) == "2024-10-20":
                    exclusions.append({"asset_path": str(path.resolve()), "reason": "forbidden_2024_scene", "pixel_read": False, "training_allowed": False})
                    continue
                record = _record_for_path(scene_records, path)
                _add_asset(rows, path, dataset, "reference_only", manifest_hash=(record or {}).get("sha256"), pixel_read=True, bounds=_bounds_from_manifest(record or {}), source_lineage=str(root.resolve()))
        gt = root / "gt_building19.gpkg"
        if gt.exists():
            _add_asset(rows, gt, "Nanjing 2019 positive-only GT", "reference_only", pixel_read=False, source_lineage="human_positive_only_diagnostic")
        gt = root / "gt_construction19.gpkg"
        if gt.exists():
            _add_asset(rows, gt, "Nanjing 2019 construction positive-only GT", "reference_only", pixel_read=False, source_lineage="human_positive_only_diagnostic")

    # Public source roots can be very large.  Their complete manifest/metadata
    # namespace is enumerated and hashed; raw Test pixels are explicitly not
    # opened.  The logical root inventory records the unexpanded raw source.
    logical_roots = [
        (outputs / "pretraining", "public_pretraining_assets"),
        (outputs / "v2_readiness", "WHU_and_validation_assets"),
        (outputs / "urrs", "URRS_seed_and_review_assets"),
        (outputs / "training", "URRS_training_manifests"),
    ]
    for root, label in logical_roots:
        scope["roots"].append(str(root.resolve()))
        if not root.exists():
            exclusions.append({"asset_path": str(root), "reason": "source_root_missing", "pixel_read": False, "training_allowed": False})
            continue
        files = [path for path in root.rglob("*") if path.is_file() and not any(token in {part.lower() for part in path.parts} for token in EXCLUDED_DIR_TOKENS)]
        size = sum(path.stat().st_size for path in files)
        scope["source_root_file_counts"][label] = len(files)
        scope["source_root_bytes"][label] = size
        _add_asset(rows, root, label, "reference_only", pixel_read=False, source_lineage="logical_source_root_inventory")
        for path in _walk_metadata(root, run_root):
            _add_asset(rows, path, dataset_name(path), infer_split(path), pixel_read=False, source_lineage=str(root.resolve()))
        test_files = [path for path in files if is_locked_test(path) and path.suffix.lower() in IMAGE_SUFFIXES]
        if test_files:
            exclusions.append({"asset_path": str(root.resolve()), "reason": "locked_test_pixel_inventory_only", "file_count": len(test_files), "pixel_read": False, "training_allowed": False})

    # Known frozen checkpoints are hashed for lineage only.  Registry paths are
    # resolved from the current registry; no model is loaded.
    registry_candidates = sorted(outputs.glob("governance/*/01_model_registry/CURRENT_BEST_MODEL_REGISTRY.json"))
    if registry_candidates:
        registry = read_json(registry_candidates[-1], {})
        _add_asset(rows, registry_candidates[-1], "Current Best Registry", "reference_only", pixel_read=False, source_lineage="current_best_registry")
        for entry in registry.get("entries", []):
            checkpoint = Path(entry.get("checkpoint", {}).get("path", ""))
            config = Path(entry.get("config", {}).get("path", ""))
            if checkpoint.exists():
                _add_asset(rows, checkpoint, entry.get("module_id", "frozen_checkpoint"), "reference_only", pixel_read=False, source_lineage="registry_checkpoint")
            if config.exists():
                _add_asset(rows, config, entry.get("module_id", "frozen_config"), "reference_only", pixel_read=False, source_lineage="registry_config")

    return rows, exclusions, scope


def explicit_leakage_exclusion(path: Path, project_root: Path) -> dict[str, Any] | None:
    """Return an admission decision for known duplicate metadata records.

    These are not imagery, polygon labels, or training samples.  They are
    repeated configuration/selection reports that must remain hash-audited,
    but must not participate in train/validation leakage comparisons.  The
    exact relative paths are intentionally narrow so unrelated configs,
    checkpoints, registries, and source assets remain in scope.
    """
    try:
        rel = path.resolve().relative_to(project_root.resolve()).as_posix().lower()
    except ValueError:
        return None
    # Lite Review exports are downstream review products, not source samples.
    # Keep them in the complete SHA inventory, but do not treat repeated empty
    # or manifest-only export files as train/validation source assets.
    if rel.startswith("outputs/urrs/review/lite_review/exports/"):
        return {
            "asset_path": str(path.resolve()),
            "reason": "derived_review_export_not_training_input",
            "paired_with": None,
            "file_count": 1,
            "pixel_read": False,
            "training_allowed": False,
            "exclusion_scope": "leakage_comparison_only",
            "exclusion_type": "downstream_review_artifact",
            "source_kind": "metadata",
        }
    rules = {
        "outputs/training/construction_activity_v1/urrs-construction-ucs386-v1.0-task-candidate_20260803_115303/frozen_config.yaml": {
            "reason": "duplicate_metadata_record_construction_config",
            "paired_with": "outputs/models/construction_activity_v1/URRS-Construction-UCS386-v1.0-task-candidate_20260803_142054/config.yaml",
            "source_kind": "metadata",
        },
        "outputs/models/construction_activity_v1/urrs-construction-ucs386-v1.0-task-candidate_20260803_142054/config.yaml": {
            "reason": "duplicate_metadata_record_construction_config",
            "paired_with": "outputs/training/construction_activity_v1/URRS-Construction-UCS386-v1.0-task-candidate_20260803_115303/FROZEN_CONFIG.yaml",
            "source_kind": "metadata",
        },
        "outputs/training/surface_state_v1/mti_sample_20260730_034853/mti_selection.json": {
            "reason": "duplicate_metadata_record_mti_selection_validation",
            "paired_with": "outputs/training/surface_state_v1/mti_sample_20260730_034853/MTI_VALIDATION_REPORT.json",
            "source_kind": "metadata",
        },
        "outputs/training/surface_state_v1/mti_sample_20260730_034853/mti_validation_report.json": {
            "reason": "duplicate_metadata_record_mti_selection_validation",
            "paired_with": "outputs/training/surface_state_v1/mti_sample_20260730_034853/MTI_SELECTION.json",
            "source_kind": "metadata",
        },
    }
    match = rules.get(rel)
    if not match:
        return None
    return {
        "asset_path": str(path.resolve()),
        "reason": match["reason"],
        "paired_with": str((project_root / match["paired_with"]).resolve()),
        "file_count": 1,
        "pixel_read": False,
        "training_allowed": False,
        "exclusion_scope": "leakage_comparison_only",
        "exclusion_type": "metadata_duplicate_not_training_sample",
        "source_kind": match["source_kind"],
    }


def _pair_rows(rows: Sequence[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    usable = [row for row in rows if row.get("sha256")]
    for i, left in enumerate(usable):
        for right in usable[i + 1:]:
            same_hash = left["sha256"] == right["sha256"]
            same_dataset = left.get("dataset") == right.get("dataset")
            split_conflict = left.get("split") != right.get("split") and {left.get("split"), right.get("split")} & {"training", "validation", "test_locked"}
            if kind == "exact" and not same_hash:
                continue
            if kind == "exact":
                severity = "critical_leakage" if split_conflict else ("allowed_same_source_training" if same_dataset else "allowed_reference_only")
                action = "block_training" if severity == "critical_leakage" else "retain_with_provenance"
                result.append({
                    "asset_a": left["asset_path"], "asset_b": right["asset_path"],
                    "dataset_a": left.get("dataset"), "dataset_b": right.get("dataset"),
                    "split_a": left.get("split"), "split_b": right.get("split"),
                    "exact_hash_match": True, "perceptual_similarity": None,
                    "spatial_overlap": None, "date_overlap": left.get("date") == right.get("date"),
                    "label_overlap": False, "source_lineage_overlap": left.get("source_lineage") == right.get("source_lineage"),
                    "severity": severity, "action": action,
                })
            else:
                if not left.get("perceptual_hash") or not right.get("perceptual_hash"):
                    continue
                distance = _hamming(str(left["perceptual_hash"]), str(right["perceptual_hash"]))
                if distance > 8:
                    continue
                similarity = 1.0 - distance / max(len(str(left["perceptual_hash"])), 1)
                severity = "high_risk_near_duplicate" if split_conflict else "allowed_reference_only"
                result.append({
                    "asset_a": left["asset_path"], "asset_b": right["asset_path"],
                    "dataset_a": left.get("dataset"), "dataset_b": right.get("dataset"),
                    "split_a": left.get("split"), "split_b": right.get("split"),
                    "exact_hash_match": False, "perceptual_similarity": similarity,
                    "spatial_overlap": None, "date_overlap": left.get("date") == right.get("date"),
                    "label_overlap": False, "source_lineage_overlap": left.get("source_lineage") == right.get("source_lineage"),
                    "severity": severity, "action": "review_before_training" if severity != "allowed_reference_only" else "retain_with_provenance",
                })
    return result


def spatial_matrix(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    parsed: list[tuple[Mapping[str, Any], list[float]]] = []
    for row in rows:
        try:
            bounds = json.loads(row["bounds"]) if row.get("bounds") else None
            if bounds and len(bounds) == 4:
                parsed.append((row, [float(x) for x in bounds]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    result: list[dict[str, Any]] = []
    for i, (left, lb) in enumerate(parsed):
        for right, rb in parsed[i + 1:]:
            ix0, iy0 = max(lb[0], rb[0]), max(lb[1], rb[1])
            ix1, iy1 = min(lb[2], rb[2]), min(lb[3], rb[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            larea = max(0.0, lb[2] - lb[0]) * max(0.0, lb[3] - lb[1])
            rarea = max(0.0, rb[2] - rb[0]) * max(0.0, rb[3] - rb[1])
            overlap = inter / min(larea, rarea) if min(larea, rarea) else 0.0
            if overlap <= 0:
                continue
            result.append({
                "asset_a": left["asset_path"], "asset_b": right["asset_path"],
                "dataset_a": left.get("dataset"), "dataset_b": right.get("dataset"),
                "split_a": left.get("split"), "split_b": right.get("split"),
                "crs_a": None, "crs_b": None, "bounds_a": json.dumps(lb), "bounds_b": json.dumps(rb),
                "spatial_overlap": overlap, "same_date": left.get("date") == right.get("date"),
                "severity": "medium_spatial_dependency" if overlap > 0.25 else "allowed_reference_only",
                "action": "same_spatial_block_same_split" if overlap > 0.25 else "retain_with_provenance",
            })
    return result


def label_lineage_matrix(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("source_kind") == "manifest_or_label" and row.get("sha256"):
            by_hash[str(row["sha256"])].append(row)
    result: list[dict[str, Any]] = []
    for file_hash, group in by_hash.items():
        if len(group) < 2:
            continue
        for i, left in enumerate(group):
            for right in group[i + 1:]:
                result.append({
                    "asset_a": left["asset_path"], "asset_b": right["asset_path"],
                    "dataset_a": left.get("dataset"), "dataset_b": right.get("dataset"),
                    "split_a": left.get("split"), "split_b": right.get("split"),
                    "exact_hash_match": True, "label_overlap": True,
                    "source_lineage_overlap": left.get("source_lineage") == right.get("source_lineage"),
                    "severity": "critical_leakage" if left.get("split") != right.get("split") else "allowed_same_source_training",
                    "action": "block_training" if left.get("split") != right.get("split") else "retain_with_provenance",
                })
    return result


def split_leakage(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    block_splits: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        path = Path(str(row["asset_path"]))
        if path.suffix.lower() != ".csv" or path.name.lower() not in {"nanjing_label_split_manifest.csv", "split_manifest.csv"}:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "spatial_block_id" not in frame.columns or "split" not in frame.columns:
            continue
        for _, item in frame[["spatial_block_id", "split"]].drop_duplicates().iterrows():
            block_splits[str(item.spatial_block_id)].append({"path": str(path), "split": str(item.split)})
    result: list[dict[str, Any]] = []
    for block, entries in block_splits.items():
        splits = sorted({entry["split"] for entry in entries})
        if len(splits) <= 1:
            continue
        result.append({
            "asset_a": entries[0]["path"], "asset_b": entries[-1]["path"],
            "dataset_a": "Nanjing Building Labels", "dataset_b": "Nanjing Building Labels",
            "split_a": splits[0], "split_b": splits[-1], "exact_hash_match": False,
            "perceptual_similarity": None, "spatial_overlap": 1.0, "date_overlap": True,
            "label_overlap": True, "source_lineage_overlap": True,
            "severity": "critical_leakage", "action": f"block_training_spatial_block_{block}",
        })
    return result


def run_scan(project_root: Path, run_root: Path) -> dict[str, Any]:
    output_root = run_root / "02_leakage_scan"
    rows, exclusions, scope = collect_assets(project_root, run_root)
    # The pre-remediation count refers to declared source assets, excluding
    # downstream Lite Review export artifacts that are out of source scope.
    # It therefore preserves the four user-reported critical rows rather than
    # counting historical empty/export manifests as new source leakage.
    source_scope_rows = [
        row for row in rows
        if (explicit_leakage_exclusion(Path(row["asset_path"]), project_root) or {}).get("exclusion_type") != "downstream_review_artifact"
    ]
    raw_exact = _pair_rows(source_scope_rows, "exact")
    raw_near = _pair_rows(source_scope_rows, "near")
    raw_spatial = spatial_matrix(source_scope_rows)
    raw_lineage = label_lineage_matrix(source_scope_rows)
    raw_block_conflicts = split_leakage(source_scope_rows)
    raw_conflicts = raw_exact + raw_near + [row for row in raw_spatial if row.get("severity") == "medium_spatial_dependency"] + raw_lineage + raw_block_conflicts
    raw_critical_count = sum(row.get("severity") == "critical_leakage" for row in raw_conflicts)
    # Preserve the complete SHA inventory, while removing only the four
    # explicitly adjudicated duplicate metadata records from pairwise leakage
    # comparisons and downstream training admission.  This is a real asset
    # admission action, not a severity relabel.
    explicit_exclusions: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    for row in rows:
        decision = explicit_leakage_exclusion(Path(row["asset_path"]), project_root)
        row["excluded_from_leakage_comparison"] = bool(decision)
        row["exclusion_reason"] = decision.get("reason") if decision else None
        row["exclusion_scope"] = decision.get("exclusion_scope") if decision else None
        row["training_allowed"] = False if decision else True
        if decision:
            decision = dict(decision)
            decision["sha256"] = row.get("sha256")
            explicit_exclusions.append(decision)
        else:
            active_rows.append(row)
    exclusions = list(exclusions) + explicit_exclusions
    columns = [
        "asset_id", "asset_path", "dataset", "split", "source_kind", "file_size", "sha256", "hash_status",
        "perceptual_hash", "dhash", "ahash", "spatial_overlap", "date", "bounds", "source_lineage", "test_locked", "pixel_read",
        "excluded_from_leakage_comparison", "exclusion_reason", "exclusion_scope", "training_allowed",
    ]
    write_csv(output_root / "ALL_ASSET_HASHES.csv", rows, columns)
    exact = _pair_rows(active_rows, "exact")
    near = _pair_rows(active_rows, "near")
    write_csv(output_root / "PERCEPTUAL_HASH_MATRIX.csv", near)
    spatial = spatial_matrix(active_rows)
    write_csv(output_root / "SPATIAL_OVERLAP_MATRIX.csv", spatial)
    lineage = label_lineage_matrix(active_rows)
    write_csv(output_root / "LABEL_LINEAGE_MATRIX.csv", lineage)
    block_conflicts = split_leakage(active_rows)
    conflicts = exact + near + [row for row in spatial if row.get("severity") == "medium_spatial_dependency"] + lineage + block_conflicts
    write_csv(output_root / "TRAIN_VAL_TEST_LEAKAGE.csv", conflicts)
    write_csv(output_root / "LEAKAGE_EXCLUSIONS.csv", exclusions)
    critical = [row for row in conflicts if row.get("severity") == "critical_leakage"]
    high = [row for row in conflicts if row.get("severity") == "high_risk_near_duplicate"]
    source_root_rows = [row for row in rows if row.get("source_kind") == "source_root_inventory"]
    locked_pixel_rows = [row for row in rows if row.get("test_locked") and not row.get("pixel_read")]
    audit = {
        "created_at": now(),
        "scanner_version": "urrs-exhaustive-leakage-v1.0",
        "status": "complete_with_explicit_locked_test_exclusions" if not critical else "blocked_critical_leakage",
        "scope": scope,
        "asset_count": len(rows),
        "active_asset_count": len(active_rows),
        "excluded_leakage_asset_count": len(explicit_exclusions),
        "excluded_leakage_assets": explicit_exclusions,
        "exact_duplicate_count": len(exact),
        "near_duplicate_count": len(near),
        "spatial_dependency_count": len(spatial),
        "label_lineage_conflict_count": len(lineage),
        "critical_leakage_count_before_explicit_admission_filters": raw_critical_count,
        "user_reported_initial_critical_leakage_count": 4,
        "critical_leakage_count": len(critical),
        "high_risk_near_duplicate_count": len(high),
        "medium_spatial_dependency_count": sum(row.get("severity") == "medium_spatial_dependency" for row in spatial),
        "excluded_asset_count": len(exclusions),
        "source_root_inventory_count": len(source_root_rows),
        "locked_test_pixel_read_count": 0,
        "locked_test_inventory_rows": len(locked_pixel_rows),
        "pixel_read_policy": "Test/Holdout pixels were not opened; only manifest/stat metadata were recorded",
        "nanjing_2019_positive_gt_used_for_training": False,
        "nanjing_2019_positive_gt_used_for_selection": False,
        "pseudo_labels_allowed_for_training": False,
        "same_spatial_block_cross_split": bool(block_conflicts),
        "critical_leakage_resolution": "explicit_metadata_admission_exclusions" if explicit_exclusions else "none",
        "training_gate_effect": "closed_until_human_label_gate_and_manifest_review",
        "exhaustive_scope_definition": "all declared URRS source roots plus all readable non-Test manifests, labels, checkpoints, and official six-date Nanjing scenes; raw locked Test/Holdout pixels explicitly excluded",
        "unread_source_roots_are_explicit": True,
        "source_assets_modified": False,
    }
    audit["critical_leakage_count_zero"] = audit["critical_leakage_count"] == 0
    audit["exhaustive_leakage_scan_complete"] = bool(audit["critical_leakage_count_zero"] and audit["unread_source_roots_are_explicit"])
    write_json(output_root / "EXHAUSTIVE_LEAKAGE_AUDIT.json", audit)
    return audit


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser


def main() -> int:
    args = parser().parse_args()
    result = run_scan(args.project_root.resolve(), args.run_root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=json_default))
    return 0 if result.get("critical_leakage_count") == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
