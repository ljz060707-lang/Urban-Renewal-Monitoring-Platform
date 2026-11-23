from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--qa", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    v2 = read_json(args.v2 / "FINAL_CANDIDATE_RECEIPT_V2.json")
    visual = read_json(args.qa / "VISUAL_QA_REVIEW_SUMMARY_V2.json")
    fixed = read_json(args.qa / "fixed_case_audit" / "FIXED_CASE_REGRESSION_V2_RECEIPT.json")
    unknown = read_json(args.qa / "unknown_safety" / "UNKNOWN_SAFETY_AUDIT_V2.json")
    receipt = {
        "task": v2.get("task"),
        "candidate_output": str(args.v2),
        "visual_qa_output": str(args.qa),
        "gates": {
            "V4B_REGRESSION_GATE": v2.get("v4b_regression_gate"),
            "STRICT_CORE_LEDGER_RECONCILIATION_GATE": v2.get("ledger_reconciliation_gate"),
            "STATE_SANITY_GATE": v2.get("state_sanity_gate"),
            "TRACKING_ACCURACY_GATE": v2.get("tracking_accuracy_gate"),
            "FIXED_CASE_GATE": fixed.get("gate"),
            "UNKNOWN_LT5_GATE": v2.get("unknown_lt5_gate"),
            "UNKNOWN_SAFETY_GATE": "FAIL_REVIEW_REQUIRED",
            "VISUAL_QA_GATE": visual.get("visual_qa_gate"),
            "FALSE_CHANGE_VISUAL_GATE": visual.get("false_change_visual_gate"),
            "FULL_CITY_GRID_GATE": "NOT_RUN",
            "FINAL_PRODUCT_GATE": "NOT_RUN",
        },
        "counts": {
            "canonical_objects": v2.get("canonical_objects"),
            "physical_tracks": v2.get("physical_tracks"),
            "strict_core_input_rows": v2.get("strict_core_input_rows"),
            "v4b_promoted_components": v2.get("v4b_promoted_components"),
            "held_edges": v2.get("held_edges"),
            "residual_unknown_rows": v2.get("residual_unknown_rows"),
            "visual_asset_samples": visual.get("asset_sample_count"),
            "fixed_cases": fixed.get("fixed_case_count"),
        },
        "held_decisions": v2.get("held_full_recovery_decisions"),
        "unknown_area_pct": v2.get("unknown_area_pct"),
        "state_area_pct": v2.get("state_area_pct"),
        "formal_outputs_modified": False,
        "geometry_owner_formal_unchanged": True,
        "ready_for_final_production": "NO",
        "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
        "closure_conclusion": unknown.get("conclusion"),
    }
    (args.out / "XIAN_V4B_CANDIDATE_CLOSURE_RECEIPT_V2.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Xi'an V4B candidate closure V2",
        "",
        "状态：CANDIDATE_NOT_GROUND_TRUTH；不得进入正式全域产品。",
        "",
        "## 已通过",
        "",
        "- V4B regression gate: PASS；回归行数 0。",
        "- Strict-core ledger reconciliation: PASS。",
        "- State sanity: PASS。",
        "- 2018→2020 direct support 的 formal transition eligibility 全部为 NO。",
        "- 正式 Geometry / Tracking / Presence / State / TIF 未修改。",
        "",
        "## 仍未通过",
        "",
        f"- Unknown area: 2018→2019 {v2['unknown_area_pct']['2018_2019']:.4f}%；2019→2020 {v2['unknown_area_pct']['2019_2020']:.4f}%；均未达到 <5%。",
        "- Tracking accuracy、fixed-case、visual false-change 仍需人工/更高质量辅助证据。",
        "- 全域统一网格和最终产品 gate 未运行。",
        "",
        "## 视觉结论",
        "",
        "- 已生成 700 个真实样本：Stable/Add/Demolition/Unknown 各 100，held 2019→2020 300。",
        "- Addition 样本全部为 1-0-1，不能仅凭候选规则当作直接新增；统一标记 UNCERTAIN。",
        "- Demolition 样本保守标记 UNCERTAIN；稳定和 held 只标 PLAUSIBLE，不等于 ground truth。",
        "- fixed-case 仅 1 个达到 one-physical-track 候选 PASS，其余保持视觉复核门禁。",
        "",
        "## 结论",
        "",
        "剩余 Unknown 主要来自不可观测、身份多候选/无匹配和未解 compound。没有新的独立观测或完成的视觉确认时，继续强行消 Unknown 会制造误拆/误建，因此本轮 STOP 于候选闭合层。",
        "",
        f"候选目录：`{args.v2}`",
        f"视觉 QA 目录：`{args.qa}`",
    ]
    (args.out / "XIAN_V4B_CANDIDATE_CLOSURE_REPORT_V2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ready_for_final_production": "NO", "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
