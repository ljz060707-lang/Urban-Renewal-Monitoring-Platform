"""Import the current Nanjing 66-block label package into URRS Lite Review.

This is a review-admission operation only.  It copies already-created
non-Holdout label-package chips into the local review data root, registers all
66 block split decisions, creates regular review entries for 45 train and 11
validation blocks, and records the 10 Holdout blocks as sealed metadata only.
It never opens Holdout chips, raw scenes, checkpoints, or Registry files.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from urrs_lite_review.core import (  # noqa: E402
    DATA_ROOT,
    ReviewDB,
    atomic_write_json,
    geometry_bbox,
    geometry_centroid,
    project_id,
    stable_hash,
    transform_geometry,
    write_csv,
)


RUN_NAME = "nanjing_domain_adaptation_stable_tracking_20260805_123034"
RUN_ROOT = PROJECT_ROOT / "outputs" / "inference" / RUN_NAME
PACKAGE_ROOT = RUN_ROOT / "05_label_package"
REVIEW_PROJECT = "nanjing_building_domain_20260805"
DB_PROJECT = project_id(REVIEW_PROJECT)
DATES = ["2015-01-04", "2016-08-18", "2017-12-11", "2018-02-08", "2019-03-16", "2022-12-02"]
SPLIT_TO_DB = {"training": "train", "validation": "validation", "holdout": "holdout"}
POLYGON_ACTIONS = ["accept", "delete", "add", "correct"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def wkt_polygon_to_geojson(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text.upper().startswith("POLYGON"):
        raise ValueError(f"only POLYGON WKT is supported for block import: {text[:40]}")
    body = text[text.find("((") + 2 : text.rfind("))")]
    rings: list[list[list[float]]] = []
    for ring_text in body.split("),("):
        points: list[list[float]] = []
        for pair in ring_text.split(","):
            parts = pair.strip().split()
            if len(parts) < 2:
                raise ValueError(f"invalid WKT coordinate: {pair!r}")
            points.append([float(parts[0]), float(parts[1])])
        if points[0] != points[-1]:
            points.append(points[0])
        rings.append(points)
    geometry = {"type": "Polygon", "coordinates": rings}
    return transform_geometry(geometry)


def copy_chip(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    return {"path": str(destination.resolve()), "sha256": stable_hash(destination.read_bytes()), "source": str(source.resolve())}


def choose_pilot(rows: list[dict[str, Any]], split: str, count: int) -> list[str]:
    """Choose deterministic coverage-diverse blocks, independent of metrics."""
    candidates = [row for row in rows if row["split"] == split]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        tag = (row.get("planned_coverage_tags") or "unclassified").split(";")[0]
        buckets[tag].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: stable_hash([split, row["block_id"]]))
    selected: list[str] = []
    tags = sorted(buckets)
    while len(selected) < count and tags:
        changed = False
        for tag in tags:
            if buckets[tag]:
                selected.append(buckets[tag].pop(0)["block_id"])
                changed = True
                if len(selected) >= count:
                    break
        if not changed:
            break
    if len(selected) != count:
        raise RuntimeError(f"unable to select {count} Pilot blocks from {split}")
    return selected


def gate_details(block_rows: list[dict[str, Any]], pilot_training: list[str], pilot_validation: list[str]) -> dict[str, Any]:
    return {
        "project": REVIEW_PROJECT,
        "status": "blocked_pending_human_labels",
        "requirements": {
            "training_blocks": 12,
            "validation_blocks": 4,
            "dates_per_block": 6,
            "complete_background": True,
            "valid_mask": True,
            "ignore_mask_supported": True,
            "polygon_actions": POLYGON_ACTIONS,
            "export_statuses": ["human_verified", "human_corrected"],
            "validation_blind_first": True,
            "holdout_sealed": True,
        },
        "counts": {
            "imported_blocks": len(block_rows),
            "training_blocks": sum(row["split"] == "training" for row in block_rows),
            "validation_blocks": sum(row["split"] == "validation" for row in block_rows),
            "holdout_blocks": sum(row["split"] == "holdout" for row in block_rows),
            "pilot_training_blocks": len(pilot_training),
            "pilot_validation_blocks": len(pilot_validation),
            "pilot_date_chips": (len(pilot_training) + len(pilot_validation)) * len(DATES),
            "regular_holdout_review_entries": 0,
        },
        "pilot_blocks": {"training": pilot_training, "validation": pilot_validation},
        "training_command": {"command_id": "TR1_HEAD_ONLY", "confirmation": "START PILOT", "execute_automatically": False},
        "holdout_policy": {"sealed": True, "pixels_opened": 0, "regular_review_entry": False, "exported": False},
    }


def import_review() -> dict[str, Any]:
    selection_path = PACKAGE_ROOT / "NANJING_LABEL_BLOCK_SELECTION.csv"
    manifest_path = PACKAGE_ROOT / "LABEL_PACKAGE_MANIFEST.json"
    if not selection_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("label package manifest or block selection is missing")
    package_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    block_rows = read_csv_rows(selection_path)
    if len(block_rows) != 66 or len({row["block_id"] for row in block_rows}) != 66:
        raise ValueError("the current label package must contain 66 unique blocks")
    if {row["split"] for row in block_rows} != {"training", "validation", "holdout"}:
        raise ValueError("the current label package must contain training, validation, and holdout splits")
    split_counts = {split: sum(row["split"] == split for row in block_rows) for split in ("training", "validation", "holdout")}
    if split_counts != {"training": 45, "validation": 11, "holdout": 10}:
        raise ValueError(f"target 45/11/10 split was not preserved: {split_counts}")

    pilot_training = choose_pilot(block_rows, "training", 12)
    pilot_validation = choose_pilot(block_rows, "validation", 4)
    pilot_set = set(pilot_training + pilot_validation)
    review_chip_root = DATA_ROOT / "chips" / DB_PROJECT
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    review_chip_root.mkdir(parents=True, exist_ok=True)

    db = ReviewDB()
    try:
        db.add_project(REVIEW_PROJECT, "Nanjing Building Domain Adaptation 2026-08-05", DATES, "66 frozen spatial blocks; model references remain non-ground-truth and Holdout has no regular review entry.")
        db.add_queue(REVIEW_PROJECT, "building_block", "Nanjing 66-Block Review")
        db.add_queue(REVIEW_PROJECT, "building_pilot", "Nanjing Pilot: 12 Training + 4 Validation")
        block_admission: list[dict[str, Any]] = []
        pilot_block_records: list[dict[str, Any]] = []
        pilot_date_records: list[dict[str, Any]] = []
        holdout_records: list[dict[str, Any]] = []
        created_samples = 0

        for row in block_rows:
            block_id = row["block_id"]
            package_split = row["split"]
            db_split = SPLIT_TO_DB[package_split]
            geometry = wkt_polygon_to_geojson(row["geometry_wkt"])
            cx, cy = geometry_centroid(geometry)
            # This also locks all 66 split decisions into split_registry.  No
            # Holdout sample or chip is created below.
            db.add_block(REVIEW_PROJECT, block_id, cx, cy, split_override=db_split)
            is_pilot = block_id in pilot_set
            is_regular_review = package_split != "holdout"
            sample_id = f"njda_{package_split}_{block_id}" if is_regular_review else None
            date_chips: dict[str, dict[str, Any]] = {}
            if is_regular_review:
                for date in DATES:
                    source = PACKAGE_ROOT / "chips" / package_split / block_id / date / "image.png"
                    destination = review_chip_root / block_id / f"{date}.png"
                    chip = copy_chip(source, destination)
                    date_chips[date] = chip
                    if is_pilot:
                        pilot_date_records.append({
                            "sample_id": sample_id,
                            "block_id": block_id,
                            "package_split": package_split,
                            "db_split": db_split,
                            "scene_date": date,
                            "blind_first": package_split == "validation",
                            "model_visible": package_split == "training",
                            "chip_path": chip["path"],
                        })
                reference = {
                    "model_reference_only": True,
                    "model_reference_status": "model_reference_not_ground_truth",
                    "reference_prediction": str((PACKAGE_ROOT / "reference_predictions" / package_split / block_id / "frozen_building_reference_six_dates.png").resolve()),
                    "full20_reference": str((PACKAGE_ROOT / "full20_reference" / package_split / block_id / "full20_reference_six_dates.png").resolve()),
                }
                metadata = {
                    "review_mode": "six_date_spatial_block",
                    "package_block_id": block_id,
                    "package_split": package_split,
                    "planned_coverage_tags": row.get("planned_coverage_tags"),
                    "date_chips": date_chips,
                    "complete_background_required": True,
                    "valid_mask_required": True,
                    "ignore_mask_supported": True,
                    "polygon_actions": POLYGON_ACTIONS,
                    "model_reference_status": "model_reference_not_ground_truth",
                    "pilot_queue": "pilot" if is_pilot else "full_review_queue",
                    "pilot_role": "training" if block_id in pilot_training else ("validation" if block_id in pilot_validation else None),
                }
                created_samples += int(db.add_sample(
                    sample_id=sample_id,
                    project=REVIEW_PROJECT,
                    module="building_pilot" if is_pilot else "building_block",
                    scene_date="multi_date",
                    block=block_id,
                    source_asset=None,
                    chip_path=Path(date_chips[DATES[0]]["path"]),
                    source_geometry=geometry,
                    reference=reference,
                    metadata=metadata,
                    blind_first=package_split == "validation",
                    model_visible=package_split == "training",
                    split_override=db_split,
                ))
                if is_pilot:
                    pilot_block_records.append({
                        "sample_id": sample_id,
                        "block_id": block_id,
                        "split": package_split,
                        "db_split": db_split,
                        "blind_first": package_split == "validation",
                        "model_visible": package_split == "training",
                        "date_count": len(DATES),
                        "planned_coverage_tags": row.get("planned_coverage_tags"),
                        "chip_root": str(review_chip_root / block_id),
                    })
            else:
                holdout_records.append({
                    "block_id": block_id,
                    "package_split": package_split,
                    "db_split": db_split,
                    "regular_review_entry": False,
                    "pixels_opened": 0,
                    "sealed": True,
                    "action": "metadata_registry_only",
                })
            block_admission.append({
                "block_id": block_id,
                "package_split": package_split,
                "db_split": db_split,
                "planned_coverage_tags": row.get("planned_coverage_tags"),
                "geometry_wkt_metric_crs": row.get("geometry_wkt"),
                "geometry_crs": "EPSG:32650",
                "regular_review_entry": is_regular_review,
                "sample_id": sample_id,
                "pilot_queue": "training" if block_id in pilot_training else ("validation" if block_id in pilot_validation else None),
                "blind_first": package_split == "validation" if is_regular_review else True,
                "holdout_sealed": package_split == "holdout",
            })

        details = gate_details(block_rows, pilot_training, pilot_validation)
        gate_id = f"{DB_PROJECT}:pilot"
        db.conn.execute(
            "INSERT OR REPLACE INTO training_gates(gate_id,gate_name,status,details_json,checked_at,frozen) VALUES(?,?,?,?,?,?)",
            (gate_id, f"{REVIEW_PROJECT} Pilot Gate", "blocked", json.dumps(details, ensure_ascii=False), __import__("urrs_lite_review.core", fromlist=["now_iso"]).now_iso(), 0),
        )
        db.conn.execute("INSERT OR REPLACE INTO app_settings(key,value,updated_at) VALUES(?,?,?)", ("active_review_project", DB_PROJECT, __import__("urrs_lite_review.core", fromlist=["now_iso"]).now_iso()))
        db.conn.commit()
        progress = db.progress(DB_PROJECT)
        db_counts = {
            "sample_rows": db.conn.execute("SELECT count(*) FROM samples WHERE project_id=?", (DB_PROJECT,)).fetchone()[0],
            "registered_spatial_blocks": db.conn.execute("SELECT count(*) FROM spatial_blocks WHERE project_id=?", (DB_PROJECT,)).fetchone()[0],
            "reviewable_blocks": db.conn.execute("SELECT count(DISTINCT spatial_block_id) FROM samples WHERE project_id=?", (DB_PROJECT,)).fetchone()[0],
            "holdout_samples": db.conn.execute("SELECT count(*) FROM samples WHERE project_id=? AND split='holdout'", (DB_PROJECT,)).fetchone()[0],
            "pilot_samples": db.conn.execute("SELECT count(*) FROM samples WHERE project_id=? AND module='building_pilot'", (DB_PROJECT,)).fetchone()[0],
            "validation_blind_first": db.conn.execute("SELECT count(*) FROM samples WHERE project_id=? AND split='validation' AND blind_first=1", (DB_PROJECT,)).fetchone()[0],
        }
    finally:
        db.close()

    manifest_dir = RUN_ROOT / "12_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_csv(manifest_dir / "BLOCK_SPLIT_ADMISSION.csv", block_admission)
    write_csv(manifest_dir / "SEALED_HOLDOUT_BLOCKS.csv", holdout_records)
    write_csv(manifest_dir / "PILOT_QUEUE_BLOCKS.csv", pilot_block_records)
    write_csv(manifest_dir / "PILOT_QUEUE_DATES.csv", pilot_date_records)
    review_protocol = {
        "project": REVIEW_PROJECT,
        "review_page": "URRS Lite Review",
        "model_reference_status": "model_reference_not_ground_truth",
        "reviewable_splits": ["train", "validation"],
        "holdout": {"sealed": True, "regular_review_entry": False, "pixel_reads": 0, "export": False},
        "validation": {"blind_first": True, "model_visible_before_first_decision": False},
        "six_date_review": {"dates": DATES, "dates_per_block": 6, "comparison_unit": "same spatial block"},
        "required_human_fields": {"complete_background": True, "valid_mask": True, "ignore_mask": "when applicable", "polygon_actions": POLYGON_ACTIONS},
        "export": {"allowed_annotation_statuses": ["human_verified", "human_corrected"], "candidate_not_ground_truth": True},
        "training": {"pilot_gate_required": True, "package_after_gate": "TR1_HEAD_ONLY", "start_confirmation": "START PILOT", "automatic_start": False},
        "external_network": "disabled",
    }
    atomic_write_json(manifest_dir / "URRS_LITE_REVIEW_PROTOCOL.json", review_protocol)
    pilot_manifest = {
        "created_at": __import__("urrs_lite_review.core", fromlist=["now_iso"]).now_iso(),
        "project": REVIEW_PROJECT,
        "database_project_id": DB_PROJECT,
        "source_label_package_manifest": str(manifest_path.resolve()),
        "source_label_package_manifest_sha256": stable_hash(manifest_path.read_bytes()),
        "imported_block_count": len(block_admission),
        "reviewable_block_count": len(block_admission) - len(holdout_records),
        "holdout_block_count": len(holdout_records),
        "target_split": {"training": 45, "validation": 11, "holdout": 10},
        "actual_split": split_counts,
        "pilot": {"training_blocks": 12, "validation_blocks": 4, "dates_per_block": 6, "date_tasks": len(pilot_date_records), "blocks_file": str((manifest_dir / "PILOT_QUEUE_BLOCKS.csv").resolve()), "dates_file": str((manifest_dir / "PILOT_QUEUE_DATES.csv").resolve())},
        "db_counts": db_counts,
        "holdout_policy": {"sealed": True, "regular_review_entries": 0, "pixels_opened": 0},
        "review_protocol": str((manifest_dir / "URRS_LITE_REVIEW_PROTOCOL.json").resolve()),
        "pilot_gate": str((manifest_dir / "PILOT_GATE.json").resolve()),
        "status": "review_page_ready_pending_human_pilot_reviews",
    }
    atomic_write_json(manifest_dir / "PILOT_GATE.json", details)
    atomic_write_json(manifest_dir / "PILOT_QUEUE_MANIFEST.json", pilot_manifest)
    atomic_write_json(manifest_dir / "TR1_HEAD_ONLY_TRAINING_PACKAGE_POLICY.json", {
        "status": "not_generated_pilot_gate_pending",
        "generated": False,
        "command_id": "TR1_HEAD_ONLY",
        "start_confirmation": "START PILOT",
        "training_started": False,
        "holdout_reads": 0,
        "automatic_start": False,
        "reason": "Pilot Gate requires human_verified or human_corrected six-date review and complete background/valid-mask decisions.",
    })
    atomic_write_json(manifest_dir / "LABEL_PACKAGE_MANIFEST_IMPORTED.json", {
        "source": str(manifest_path.resolve()),
        "source_sha256": stable_hash(manifest_path.read_bytes()),
        "imported_blocks": len(block_admission),
        "reviewable_blocks": len(block_admission) - len(holdout_records),
        "sealed_holdout_blocks": len(holdout_records),
        "database_project": DB_PROJECT,
        "no_holdout_pixels_opened": True,
    })
    return {"success": True, "project": REVIEW_PROJECT, "project_id": DB_PROJECT, "pilot": details["counts"], "db_counts": db_counts, "manifest": str((manifest_dir / "PILOT_QUEUE_MANIFEST.json").resolve())}


def main(argv: list[str] | None = None) -> int:
    global RUN_ROOT, PACKAGE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args(argv)
    RUN_ROOT = args.run_root.resolve()
    PACKAGE_ROOT = RUN_ROOT / "05_label_package"
    print(json.dumps(import_review(), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
