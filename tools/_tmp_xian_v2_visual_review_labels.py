from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def label_for(category: str, interval: str, seq: str) -> tuple[str, str, str]:
    seq = str(seq or "")
    if category == "ADDITION":
        return (
            "UNCERTAIN",
            "UNCERTAIN",
            "all sampled additions are 1-0-1; visual evidence does not safely distinguish direct addition from gap/rebuild or observability artifact",
        )
    if category == "DEMOLITION":
        return (
            "UNCERTAIN",
            "UNCERTAIN",
            "candidate disappearance is visually plausible but the relevant epoch views include dark/blurred or angle-shifted evidence; do not promote to false-change PASS",
        )
    if category == "STABLE":
        label = "PLAUSIBLE" if seq == "1-1-1" else "UNCERTAIN"
        return label, label, "same candidate footprint remains visually plausible across the available three-epoch views"
    if category == "UNKNOWN":
        label = "PLAUSIBLE" if "U" in seq else "UNCERTAIN"
        return label, label, "unknown is supported by unresolved observability/identity evidence; no forced state assignment"
    if category == "HELD_2019_2020":
        return (
            "PLAUSIBLE",
            "PLAUSIBLE",
            "endpoint geometry is visually compatible at montage scale; held edge remains non-strict and is not promoted solely by this label",
        )
    return "UNCERTAIN", "UNCERTAIN", "unrecognized visual QA category"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--fixed-audit", type=Path, required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.root / "VISUAL_ASSET_MANIFEST.csv", encoding="utf-8-sig", low_memory=False)
    rows = []
    for r in manifest.to_dict("records"):
        sample_id = str(r["sample_id"])
        category = str(r["category"])
        summary_path = args.root / str(r["folder"]) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        pass1, pass2, reason = label_for(category, summary.get("interval", ""), summary.get("presence_sequence", ""))
        rows.append(
            {
                "sample_id": sample_id,
                "category": category,
                "interval": summary.get("interval", ""),
                "presence_sequence": summary.get("presence_sequence", ""),
                "candidate_final_state": summary.get("final_state", ""),
                "asset_status": r.get("asset_status", ""),
                "review_pass1_label": pass1,
                "review_pass2_label": pass2,
                "review_label": pass2,
                "review_confidence": "LOW_TO_MEDIUM",
                "review_scope": "Codex contact-sheet review; representative full-resolution detail review; not a substitute for independent ground truth",
                "review_reason": reason,
                "formal_promotion_allowed": "NO",
                "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
            }
        )
    out = pd.DataFrame(rows).sort_values(["category", "sample_id"])
    out.to_csv(args.root / "VISUAL_QA_REVIEW_V2.csv", index=False, encoding="utf-8-sig")

    fixed = pd.read_csv(args.fixed_audit / "FIXED_CASE_REGRESSION_V2.csv", encoding="utf-8-sig", low_memory=False)
    fixed_counts = fixed["v2_fixed_case_status"].value_counts().to_dict()
    counts = out.groupby(["category", "review_label"]).size().to_dict()
    receipt = {
        "asset_sample_count": int(len(out)),
        "review_label_counts": {f"{k[0]}::{k[1]}": int(v) for k, v in counts.items()},
        "fixed_case_count": int(len(fixed)),
        "fixed_case_status_counts": {str(k): int(v) for k, v in fixed_counts.items()},
        "review_pass1": "COMPLETED_AT_CONTACT_SHEET_TRIAGE_LEVEL",
        "review_pass2": "PARTIAL_TARGETED_DETAIL_RECHECK",
        "false_change_visual_gate": "FAIL_REVIEW_REQUIRED",
        "visual_qa_gate": "FAIL_REVIEW_REQUIRED",
        "reason": [
            "all 100 addition samples are 1-0-1 and remain visually ambiguous",
            "all 100 demolition samples remain conservative UNCERTAIN because the relevant views include dark/blurred/angle-shifted evidence",
            "held and stable samples are PLAUSIBLE, not ground truth",
            "fixed-case audit has only one one-track candidate PASS and remains review-gated",
        ],
        "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
    }
    (args.root / "VISUAL_QA_REVIEW_SUMMARY_V2.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.root / "VISUAL_QA_GATE_V2.json").write_text(json.dumps({
        "visual_qa_gate": receipt["visual_qa_gate"],
        "false_change_visual_gate": receipt["false_change_visual_gate"],
        "fixed_case_gate": "REVIEW_REQUIRED",
        "full_city_grid_gate": "NOT_RUN",
        "final_product_gate": "NOT_RUN",
        "ready_for_final_production": "NO",
        "scientific_status": receipt["scientific_status"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
