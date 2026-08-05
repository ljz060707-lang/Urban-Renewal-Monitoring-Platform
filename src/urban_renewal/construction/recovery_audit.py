from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_FALSE_RECOVERY_ROOTS = (
    ("WHU_Building", "test", Path(r"D:\Data Set\WHU_Building\test\image")),
    ("WHU_Building", "train", Path(r"D:\Data Set\WHU_Building\train\image")),
    ("DSIFN", "train_t1", Path(r"D:\Data Set\DSIFN-Dataset\train\t1")),
    ("DSIFN", "train_t2", Path(r"D:\Data Set\DSIFN-Dataset\train\t2")),
    ("DSIFN", "validation_t1", Path(r"D:\Data Set\DSIFN-Dataset\val\t1")),
    ("DSIFN", "validation_t2", Path(r"D:\Data Set\DSIFN-Dataset\val\t2")),
    ("DSIFN", "predicted", Path(r"D:\Data Set\DSIFN-Dataset\predicted")),
    ("DSIFN", "test_t1", Path(r"D:\Data Set\DSIFN-Dataset\test\t1")),
    ("DSIFN", "test_t2", Path(r"D:\Data Set\DSIFN-Dataset\test\t2")),
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["candidate_id", "status"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _missing_stems(ucs_root: Path) -> tuple[set[str], dict[str, Any]]:
    image_dir = ucs_root / "images_tif"
    label_dir = ucs_root / "label"
    image_stems = {path.stem for path in image_dir.glob("*.tif")} if image_dir.is_dir() else set()
    label_stems = {path.stem for path in label_dir.glob("*.png")} if label_dir.is_dir() else set()
    missing = label_stems - image_stems
    return missing, {
        "ucs_root": str(ucs_root.resolve()),
        "image_dir": str(image_dir.resolve()),
        "label_dir": str(label_dir.resolve()),
        "image_stem_count": len(image_stems),
        "label_stem_count": len(label_stems),
        "missing_stem_count": len(missing),
        "unmatched_label_stems": sorted(missing, key=lambda value: int(value) if value.isdigit() else value),
        "read_policy": "directory entries and file metadata only; no image or label pixels read",
    }


def _candidate_rows(missing: set[str], roots) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    root_audit: list[dict[str, Any]] = []
    for dataset, split, root in roots:
        root = Path(root)
        candidates = []
        if root.is_dir():
            for path in sorted(root.iterdir(), key=lambda value: value.name.casefold()):
                if not path.is_file() or path.suffix.casefold() == ".png" or path.stem not in missing:
                    continue
                candidates.append(path)
                rows.append({
                    "candidate_id": f"{dataset.lower()}_{split}_{path.stem}_{len(rows):04d}",
                    "missing_ucs_stem": path.stem,
                    "source_dataset": dataset,
                    "source_split": split,
                    "source_path": str(path.resolve()),
                    "extension": path.suffix.casefold(),
                    "file_size_bytes": path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                    "stem_collision_with_ucs": True,
                    "valid_ucs_recovery_candidate": False,
                    "rejection_code": "cross_dataset_numeric_stem_collision",
                    "rejection_reason": "same numeric stem exists in WHU/DSIFN, but source dataset semantics are not UCS imagery",
                    "pixel_read": False,
                    "source_modified": False,
                })
        root_audit.append({
            "dataset": dataset,
            "split": split,
            "path": str(root),
            "exists": root.is_dir(),
            "matched_missing_stem_candidates": len(candidates),
            "candidate_stems": sorted({path.stem for path in candidates}, key=lambda value: int(value) if value.isdigit() else value),
            "pixel_reads": 0,
        })
    return rows, root_audit


def audit_false_recovery_candidates(
    project_root: Path,
    ucs_root: Path,
    *,
    output_dir: Path | None = None,
    roots=DEFAULT_FALSE_RECOVERY_ROOTS,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    ucs_root = Path(ucs_root).resolve()
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / "urrs" / "audits" / f"ucs_missing_image_resolution_{stamp}"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing, missing_audit = _missing_stems(ucs_root)
    rows, root_audit = _candidate_rows(missing, roots)
    by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stem[row["missing_ucs_stem"]].append(row)
    source_counts = Counter((row["source_dataset"], row["source_split"]) for row in rows)
    collision_rows = []
    for stem in sorted(missing, key=lambda value: int(value) if value.isdigit() else value):
        candidates = by_stem.get(stem, [])
        collision_rows.append({
            "missing_ucs_stem": stem,
            "candidate_count": len(candidates),
            "candidate_datasets": sorted({row["source_dataset"] for row in candidates}),
            "candidate_paths": [row["source_path"] for row in candidates],
            "all_rejected": bool(candidates) and all(not row["valid_ucs_recovery_candidate"] for row in candidates),
            "rejection_code": "cross_dataset_numeric_stem_collision" if candidates else "no_candidate_found",
        })
    collision_audit = {
        "status": "FORMAL_FALSE_RECOVERY_EXCLUSION_COMPLETE",
        "candidate_selection": "numeric stem collision with canonical UCS unresolved labels; path metadata only",
        "missing_ucs_stem_count": len(missing),
        "candidate_count": len(rows),
        "candidate_unique_stem_count": len({row["missing_ucs_stem"] for row in rows}),
        "candidate_collision_stem_count": sum(bool(item["candidate_count"]) for item in collision_rows),
        "candidate_paths_by_source": [
            {"dataset": dataset, "split": split, "count": count}
            for (dataset, split), count in sorted(source_counts.items())
        ],
        "root_audit": root_audit,
        "stem_audit": collision_rows,
        "whu_test_reads": 0,
        "dsifn_test_reads": 0,
        "dsifn_predicted_reads": 0,
        "total_pixel_reads": 0,
        "source_data_modified": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_csv(output_dir / "FALSE_RECOVERY_CANDIDATES.csv", rows)
    _write_json(output_dir / "CROSS_DATASET_STEM_COLLISION_AUDIT.json", {"missing_inventory": missing_audit, **collision_audit})
    final_status = {
        "status": "FORMAL_FALSE_RECOVERY_EXCLUSION_COMPLETE",
        "valid_ucs_recovery_candidates": 0,
        "rejected_cross_dataset_candidates": len(rows),
        "remaining_missing_ucs_images": len(missing),
        "missing_image_status": "true_local_absence",
        "rejection_reason": "cross_dataset_numeric_stem_collision",
        "candidate_roots": [item for item in root_audit],
        "source_counts": collision_audit["candidate_paths_by_source"],
        "prohibited_actions": [
            "read WHU Test pixels",
            "read DSIFN Test pixels",
            "read DSIFN predicted pixels as UCS imagery",
            "copy or resize WHU/DSIFN files into UCS",
            "fabricate or reconstruct the 214 missing UCS images",
            "modify any original file under D:\\Data Set",
        ],
        "whu_test_reads": 0,
        "dsifn_test_reads": 0,
        "dsifn_predicted_reads": 0,
        "total_pixel_reads": 0,
        "source_data_modified": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "UCS_MISSING_IMAGE_FINAL_STATUS.json", final_status)
    return {
        "status": "success",
        "output_dir": str(output_dir),
        "candidate_count": len(rows),
        "missing_ucs_images": len(missing),
        "valid_ucs_recovery_candidates": 0,
        "rejected_cross_dataset_candidates": len(rows),
        "pixel_reads": 0,
        "source_data_modified": 0,
    }


__all__ = ["DEFAULT_FALSE_RECOVERY_ROOTS", "audit_false_recovery_candidates"]
