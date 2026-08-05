"""Create explicit NOT-ACCEPTED height outputs for a Tier-C match gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATES = [
    "2015-01-04",
    "2016-08-18",
    "2017-12-11",
    "2018-02-08",
    "2019-03-16",
    "2022-12-02",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-audit", type=Path, required=True)
    parser.add_argument("--height-training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    match = read_json(args.match_audit.resolve())
    if str(match.get("tier", "")).upper() != "C":
        raise RuntimeError("this guard is only valid for a Tier C GloBFP/RSBuilding match")
    training_root = args.height_training_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    training_manifest = read_json(training_root / "HEIGHT_MODEL_TRAINING_MANIFEST.json")
    invariance = read_json(training_root / "HEIGHT_BRANCH_INVARIANCE_AUDIT.json")
    common = {
        "created_at_utc": now(),
        "status": "NOT_ACCEPTED_TIER_C_MATCH_GATE",
        "tier": "C",
        "reason": "GloBFP/RSBuilding matching quality is Tier C; Nanjing height external acceptance and six-date inference are forbidden",
        "training_model_available": True,
        "height_model_id": training_manifest.get("selected_model"),
        "height_model_sha256": training_manifest.get("model_sha256"),
        "nanjing_height_labels_used_for_training": False,
        "nanjing_height_labels_used_for_model_selection": False,
        "locked_test_pixel_reads": 0,
        "source_data_modified": False,
    }
    acceptance = {
        **common,
        "external_acceptance_scope": "not executed",
        "matching_summary": {
            "globfp_feature_count": match.get("globfp_feature_count"),
            "rsbuilding_2019_count": match.get("rsbuilding_2019_count"),
            "rsbuilding_2022_count": match.get("rsbuilding_2022_count"),
            "matched_both_count": match.get("matched_both_count"),
            "stable_building_count": match.get("stable_building_count"),
            "temporal_conflict_count": match.get("temporal_conflict_count"),
            "median_footprint_iou": match.get("median_footprint_iou"),
            "median_centroid_distance_m": match.get("median_centroid_distance_m"),
        },
        "teacher_agreement_metrics_against_GloBFP_estimated_height": None,
        "not_generated": [
            "NANJING_HEIGHT_EXTERNAL_ACCEPTANCE.csv",
            "NANJING_HEIGHT_EXTERNAL_ACCEPTANCE.gpkg",
            "HEIGHT_RESIDUAL_MAP.png",
            "HEIGHT_SCATTER.png",
            "HEIGHT_ERROR_BY_SIZE.png",
            "HEIGHT_ERROR_BY_HEIGHT.png",
        ],
    }
    write_json(output_root / "04_height_acceptance" / "NANJING_HEIGHT_EXTERNAL_ACCEPTANCE.json", acceptance)
    write_json(output_root / "04_height_acceptance" / "NOT_GENERATED_ARTIFACTS.json", {
        "status": "NOT_GENERATED_TIER_C",
        "artifacts": acceptance["not_generated"],
        "created_at_utc": now(),
    })
    for date in DATES:
        write_json(output_root / "05_height_inference" / date / "HEIGHT_INFERENCE_MANIFEST.json", {
            **common,
            "date": date,
            "status": "NOT_RUN_TIER_C_MATCH_GATE",
            "scientific_status": "model_derived_building_height_candidate_not_accepted_for_nanjing",
            "not_generated": ["building_height_candidate.gpkg", "building_height_candidate.tif", "building_volume_proxy.gpkg"],
        })
    write_json(output_root / "05_height_inference" / "HEIGHT_INFERENCE_MANIFEST.json", {
        **common,
        "valid_dates": DATES,
        "inference_count": 0,
        "status": "NOT_RUN_TIER_C_MATCH_GATE",
    })
    write_json(output_root / "06_height_tracking" / "HEIGHT_TEMPORAL_STABILITY_AUDIT.json", {
        **common,
        "status": "NOT_RUN_TIER_C_MATCH_GATE",
        "track_count": 0,
        "not_generated": ["BUILDING_HEIGHT_TRACKS.csv", "BUILDING_HEIGHT_TRACKS.gpkg", "BUILDING_VOLUME_CHANGE_BY_PAIR.csv"],
    })
    write_json(output_root / "06_height_tracking" / "NOT_GENERATED_ARTIFACTS.json", {
        "status": "NOT_GENERATED_TIER_C",
        "artifacts": ["BUILDING_HEIGHT_TRACKS.csv", "BUILDING_HEIGHT_TRACKS.gpkg", "BUILDING_VOLUME_CHANGE_BY_PAIR.csv"],
        "created_at_utc": now(),
    })
    write_json(output_root / "HEIGHT_BRANCH_INVARIANCE_AUDIT.json", invariance)
    write_json(output_root / "NANJING_HEIGHT_EXTERNAL_ACCEPTANCE.json", acceptance)
    write_json(output_root / "HEIGHT_INFERENCE_MANIFEST.json", read_json(output_root / "05_height_inference" / "HEIGHT_INFERENCE_MANIFEST.json"))
    write_json(output_root / "HEIGHT_TEMPORAL_STABILITY_AUDIT.json", read_json(output_root / "06_height_tracking" / "HEIGHT_TEMPORAL_STABILITY_AUDIT.json"))
    receipt = {
        **common,
        "status": "COMPLETE_TIER_C_HEIGHT_BRANCH_DEGRADED",
        "output_root": str(output_root),
        "height_model_training_root": str(training_root),
        "nanjing_external_acceptance": "NOT_ACCEPTED_TIER_C_MATCH_GATE",
        "six_date_inference": "NOT_RUN_TIER_C_MATCH_GATE",
        "baseline_hash_match_count": invariance.get("baseline_hash_match_count"),
        "baseline_hash_total_count": invariance.get("baseline_hash_total_count"),
        "baseline_hash_invariance_passed": invariance.get("baseline_hash_invariance_passed"),
    }
    write_json(output_root / "HEIGHT_BRANCH_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
