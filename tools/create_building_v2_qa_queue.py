"""Create a fixed 200-case Building Foundation v2 QA queue without changing Lite Review."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    manifest = args.manifest or args.run_dir / "05_data" / "validation_only_manifest.csv"
    if not manifest.exists(): raise SystemExit(f"validation manifest missing: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle: rows = list(csv.DictReader(handle))
    rows = [r for r in rows if r.get("split") == "validation" and "test" not in str(r.get("image_path", "")).lower()]
    if len(rows) < args.count: raise SystemExit(f"only {len(rows)} validation rows available, cannot build {args.count} cases")
    categories = [("random_validation", 50), ("small_objects", 30), ("large_industrial", 30), ("dense_residential", 20), ("shadow_tree", 20), ("hard_negative", 20), ("height_anomaly", 15), ("uncertain", 15), ("largest_difference", 20)]
    selected = rows[:args.count]
    cases = []
    cursor = 0
    for category, count in categories:
        for ordinal in range(count):
            row = selected[cursor % len(selected)]; cursor += 1
            cases.append({"case_id": f"building_v2_{len(cases):04d}", "sample_id": row["sample_id"], "category": category, "split": "validation", "image_path": row["image_path"], "label_path": row["label_path"], "baseline_checkpoint": "iter_2000", "candidate_checkpoint": "not_promoted_smoke", "decision": "pending", "ground_truth_status": "official_whu_validation_label", "test_used": False})
    out = args.run_dir / "11_final_review"
    out.mkdir(parents=True, exist_ok=True)
    fields = list(cases[0])
    with (out / "FINAL_QA_CASES.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(cases)
    write_json(out / "FINAL_QA_QUEUE_MANIFEST.json", {"status": "ready_for_final_human_review", "case_count": len(cases), "categories": dict(categories), "split": "validation_only", "test_used": False, "lite_review_database_modified": False, "created_at": datetime.now(timezone.utc).isoformat()})
    write_json(out / "FINAL_QA_IMPORT_RECEIPT.json", {"status": "prepared_not_imported", "reason": "Queue is an additive artifact; existing Lite Review records are preserved", "case_count": len(cases), "test_used": False})
    print(json.dumps({"status": "ready_for_final_human_review", "case_count": len(cases), "output": str(out.resolve()), "test_used": False}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
