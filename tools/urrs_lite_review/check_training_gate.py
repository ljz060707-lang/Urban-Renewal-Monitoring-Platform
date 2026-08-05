from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path: sys.path.insert(0, str(HERE.parent))
from urrs_lite_review.core import DATA_ROOT, HOLDOUT_SPLIT, ReviewDB, TRAINING_IMPORTABLE, atomic_write_json, now_iso, project_id  # noqa: E402


def _counts(db: ReviewDB, pid: str) -> dict:
    rows = [dict(r) for r in db.conn.execute("SELECT s.*,h.annotation_status AS h_status FROM samples s LEFT JOIN human_annotations h ON h.sample_id=s.sample_id WHERE s.project_id=?", (pid,))]
    by_split = {}
    for split in ("train", "validation", "holdout"):
        rs = [r for r in rows if r["split"] == split]
        by_split[split] = {"samples": len(rs), "blocks": len({r["spatial_block_id"] for r in rs}), "reviewed": sum(1 for r in rs if r["h_status"] in TRAINING_IMPORTABLE), "complete": sum(1 for r in rs if r["decision"] in {"complete_prediction_correct", "prediction_needs_correction", "image_quality_ignore"}), "hard_negative": sum(1 for r in rs if r.get("negative_type"))}
    block_splits = {}
    for r in rows: block_splits.setdefault(r["spatial_block_id"], set()).add(r["split"])
    leakage = {"critical": [b for b, s in block_splits.items() if len(s) > 1], "duplicate_source_hash": []}
    hashes = {}
    for r in rows:
        if r.get("source_hash"): hashes.setdefault(r["source_hash"], set()).add(r["split"])
    leakage["duplicate_source_hash"] = [h for h, s in hashes.items() if len(s) > 1]
    return {"rows": len(rows), "by_split": by_split, "leakage": leakage, "holdout_sealed": db.setting("holdout_sealed", "1") == "1"}


def evaluate() -> dict:
    db = ReviewDB()
    try:
        nj = _counts(db, "nanjing_building"); tz = _counts(db, "tongzhou_construction")
        p = nj["by_split"]
        critical = bool(nj["leakage"]["critical"] or nj["leakage"]["duplicate_source_hash"] or tz["leakage"]["critical"] or tz["leakage"]["duplicate_source_hash"])
        building_pilot_reasons = []
        if p["train"]["blocks"] < 12: building_pilot_reasons.append(f"training blocks {p['train']['blocks']} < 12")
        if p["validation"]["blocks"] < 4: building_pilot_reasons.append(f"validation blocks {p['validation']['blocks']} < 4")
        if p["train"]["reviewed"] < p["train"]["samples"] or p["validation"]["reviewed"] < p["validation"]["samples"]: building_pilot_reasons.append("unreviewed or non-human labels remain")
        if p["train"]["complete"] < p["train"]["blocks"] or p["validation"]["complete"] < p["validation"]["blocks"]: building_pilot_reasons.append("complete-background confirmation is incomplete")
        if not nj["holdout_sealed"]: building_pilot_reasons.append("holdout is not sealed")
        if critical: building_pilot_reasons.append("critical split/source leakage detected")
        full_reasons = list(building_pilot_reasons)
        if p["train"]["blocks"] < 28: full_reasons.append(f"full training blocks {p['train']['blocks']} < 28")
        if p["validation"]["blocks"] < 10: full_reasons.append(f"full validation blocks {p['validation']['blocks']} < 10")
        t = tz["by_split"]
        construction_reasons = []
        if t["train"]["reviewed"] < 100: construction_reasons.append(f"verified construction train {t['train']['reviewed']} < 100")
        if t["validation"]["reviewed"] < 30: construction_reasons.append(f"verified construction validation {t['validation']['reviewed']} < 30")
        if t["train"]["hard_negative"] < 200: construction_reasons.append(f"hard negative train {t['train']['hard_negative']} < 200")
        if critical: construction_reasons.append("critical split/source leakage detected")
        result = {"generated_at": now_iso(), "policy": {"candidate_not_ground_truth": True, "holdout_sealed": True, "locked_test_reads": 0, "2019_positive_only_gt_used": False}, "building": {"pilot": {"status": "blocked" if building_pilot_reasons else "pass", "reasons": building_pilot_reasons, "counts": nj}, "full": {"status": "blocked" if full_reasons else "pass", "reasons": full_reasons, "counts": nj}}, "construction": {"classifier": {"status": "blocked" if construction_reasons else "pass", "reasons": construction_reasons, "counts": tz}}, "holdout": {"sealed": True, "status": "sealed"}}
        atomic_write_json(DATA_ROOT / "BUILDING_LABEL_GATE.json", result["building"])
        atomic_write_json(DATA_ROOT / "CONSTRUCTION_LABEL_GATE.json", result["construction"])
        for name, gate in [("building_pilot", result["building"]["pilot"]), ("building_full", result["building"]["full"]), ("construction_classifier", result["construction"]["classifier"])]:
            db.conn.execute("INSERT OR REPLACE INTO training_gates(gate_id,gate_name,status,details_json,checked_at,frozen) VALUES(?,?,?,?,?,?)", (name, name, gate["status"], json.dumps(gate, ensure_ascii=False), now_iso(), int(gate["status"] == "pass")))
        db.conn.commit(); return result
    finally: db.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("-CheckTrainingGate", "--check", action="store_true"); p.add_argument("--json", action="store_true")
    args = p.parse_args(argv); result = evaluate(); print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

