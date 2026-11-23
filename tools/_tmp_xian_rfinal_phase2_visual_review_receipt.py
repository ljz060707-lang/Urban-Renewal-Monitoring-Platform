"""Record the Phase 2 identity-edge visual triage without promoting edges.

This is an audit-only script.  It reads the generated visual sample index and
writes a small, append-only review receipt.  It never changes the frozen
Geometry, IdentityGraph, Presence, State, or product outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


MANUAL = {
    # Targeted contact-sheet review performed in this turn.  These labels are
    # deliberately conservative and are not promotion decisions.
    "EDGE_2018_2019_001": ("REJECT_GEOMETRY_FALSE_POSITIVE", "candidate follows a tree/vegetation strip rather than a building footprint"),
    "EDGE_2018_2019_002": ("REJECT_GEOMETRY_FALSE_POSITIVE", "candidate follows a narrow vegetation strip and adjacent seam"),
    "EDGE_2018_2019_003": ("REJECT_CANDIDATE_NOT_SAME_OBJECT", "candidate is a roof/vegetation fragment inconsistent with the source object"),
    "EDGE_2018_2019_004": ("REJECT_GEOMETRY_OR_ALIGNMENT", "source and candidate outlines are not the same stable building object"),
    "EDGE_2018_2019_008": ("UNCERTAIN_SPLIT_OR_BOUNDARY", "large roof and adjacent tree/building boundaries require object-level adjudication"),
    "EDGE_2018_2019_009": ("UNCERTAIN_SPLIT_OR_BOUNDARY", "candidate appears to cover a different boundary portion; not safe to promote"),
    "EDGE_2018_2019_041": ("PLAUSIBLE_SAME_ROOF_NEEDS_COMPONENT_REVIEW", "candidate follows the same large roof, but relation is N:1"),
    "EDGE_2018_2019_042": ("PLAUSIBLE_SAME_ROOF_DIRECT_SUPPORT_ONLY", "candidate follows the same large roof in direct 2018/2020 support"),
    "EDGE_2019_2020_001": ("UNCERTAIN_PARALLAX_OR_FRAGMENT", "candidate is a roof fragment under strong view/geometry change"),
    "EDGE_2019_2020_002": ("REJECT_CANDIDATE_NOT_SAME_OBJECT", "candidate polygon expands to a different roof texture/object"),
    "EDGE_2019_2020_003": ("REJECT_CANDIDATE_NOT_SAME_OBJECT", "candidate polygon is inconsistent with the source roof fragment"),
    "EDGE_2019_2020_005": ("UNCERTAIN_IDENTITY_GEOMETRY", "candidate and source are in a dense multi-roof scene; same-object identity is not visually decisive"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "sample_id", "interval", "physical_track_id", "scene_key", "area_m2",
        "identity_queue_status", "visual_label", "visual_reason",
        "review_level", "promotion_decision", "formal_outputs_modified",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual-root", type=Path, required=True)
    args = ap.parse_args()
    visual_root = args.visual_root
    index_path = visual_root / "IDENTITY_EDGE_VISUAL_SAMPLE_INDEX.csv"
    rows = read_csv(index_path)
    labels: list[dict[str, str]] = []
    for row in rows:
        sample_id = row.get("sample_id", "")
        label, reason = MANUAL.get(
            sample_id,
            ("CONTACT_SHEET_TRIAGE_PENDING", "not manually adjudicated in this targeted review"),
        )
        review_level = "TARGETED_MANUAL_CONTACT_SHEET" if sample_id in MANUAL else "PENDING_TARGETED_REVIEW"
        labels.append({
            "sample_id": sample_id,
            "interval": row.get("interval", ""),
            "physical_track_id": row.get("physical_track_id", ""),
            "scene_key": row.get("scene_key", ""),
            "area_m2": row.get("area_m2", ""),
            "identity_queue_status": row.get("identity_queue_status", ""),
            "visual_label": label,
            "visual_reason": reason,
            "review_level": review_level,
            "promotion_decision": "DO_NOT_PROMOTE",
            "formal_outputs_modified": "false",
        })

    labels_path = visual_root / "PHASE2_VISUAL_REVIEW_LABELS.csv"
    write_csv(labels_path, labels)
    manual_labels = [r for r in labels if r["review_level"] == "TARGETED_MANUAL_CONTACT_SHEET"]
    counts = Counter(r["visual_label"] for r in manual_labels)
    interval_counts = Counter(r["interval"] for r in manual_labels)
    receipt = {
        "task": "XIAN-RFINAL-PHASE2-VISUAL-REVIEW-RECEIPT",
        "status": "FAIL_REVIEW_REQUIRED",
        "review_scope": "identity-edge candidate contact sheets",
        "sample_count": len(labels),
        "targeted_manual_sample_count": len(manual_labels),
        "pending_sample_count": len(labels) - len(manual_labels),
        "manual_label_counts": dict(counts),
        "manual_interval_counts": dict(interval_counts),
        "promotion_count": 0,
        "promotion_gate": "FAIL_REVIEW_REQUIRED",
        "disposition": "ROLLBACK_CANDIDATE_ONLY",
        "reason": "targeted samples include geometry false positives, non-same-object edges, and unresolved split/merge or parallax cases; metrics alone do not establish identity",
        "formal_outputs_modified": False,
        "geometry_formal_unchanged": True,
        "tracking_formal_unchanged": True,
        "presence_formal_unchanged": True,
        "state_formal_unchanged": True,
        "products_formal_unchanged": True,
        "next_action": "retain candidates as review-only and continue geometry-vs-identity diagnosis; do not start Presence/State promotion",
    }
    (visual_root / "PHASE2_VISUAL_REVIEW_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Xi'an R-FINAL Phase 2 identity-edge visual review",
        "",
        "结论：本轮候选边不推广，正式结果未修改。",
        "",
        f"- 候选样本总数：{len(labels)}",
        f"- 定向人工接触表复核：{len(manual_labels)}",
        f"- 尚待复核：{len(labels) - len(manual_labels)}",
        f"- 标签计数：{dict(counts)}",
        "- promotion_count: 0",
        "- formal_outputs_modified: false",
        "",
        "指标达标只能说明几何/邻域候选相似，不能排除树带、道路、屋顶纹理、split/merge 或视差误配。",
        "在 IdentityGraph 候选边未通过完整视觉与结构一致性 Gate 前，不进入 Presence/State 正式回写。",
    ]
    (visual_root / "PHASE2_VISUAL_REVIEW_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
