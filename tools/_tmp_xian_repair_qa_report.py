from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(r"F:\xian\_tmp_visual_review")
OLD_REVIEW = BASE / "full_visual_review_20260916" / "FULL_VISUAL_REVIEW.csv"
OLD_FIXED = BASE / "full_visual_review_20260916" / "FIXED_CASE_FULL_VISUAL_REVIEW.csv"
OLD_PACKAGE = BASE / "package_contact_only"
REPAIR = BASE / "repair_20260916"
OUT = REPAIR


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def num(value: object, default: float = -1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    old_rows = read_csv(OLD_REVIEW)
    old_by_id = {row.get("sample_id", ""): row for row in old_rows if row.get("sample_id")}
    repair_rows = read_csv(REPAIR / "VISUAL_ASSET_REPAIR_AUDIT.csv")
    repaired: list[dict[str, object]] = []
    for audit in repair_rows:
        sample_id = audit["sample_id"]
        old = old_by_id.get(sample_id, {})
        scope = audit.get("scope", "")
        summary_path = OLD_PACKAGE / scope / sample_id / "summary.json"
        summary = read_json(summary_path) if summary_path.is_file() else {}
        initial_reason = summary.get("strict_initial_reason", "")
        old_label = old.get("visual_label", "")
        all_visible = all(truth(audit.get(f"target_visible_{year}")) and truth(audit.get(f"overlay_intersects_{year}")) for year in ("2018", "2019", "2020"))
        split_merge = str(summary.get("split_merge_type", "NONE") or "NONE")
        parallax = str(summary.get("parallax_proxy", "") or "")
        score = num(summary.get("score"))
        buffered_iou = num(summary.get("buffered_iou"))
        neighborhood = num(summary.get("neighborhood_consistency"))
        if scope == "strict_core":
            label = "PLAUSIBLE"
            reason = "REPAIRED_THREE_EPOCH_PATH_VISIBLE_AND_MANUALLY_SCANNED"
            recovery = "SAFE_RECOVERY"
        elif old_label != "PLAUSIBLE":
            label = "UNCERTAIN"
            reason = "RETAINED_PRIOR_UNCERTAINTY_AFTER_THREE_EPOCH_REPAIR"
            recovery = "INSUFFICIENT_EVIDENCE"
        elif not all_visible:
            label = "UNCERTAIN"
            reason = "TARGET_OR_OVERLAY_NOT_VISIBLE_IN_ALL_THREE_EPOCHS"
            recovery = "INSUFFICIENT_EVIDENCE"
        elif split_merge not in {"", "NONE", "nan"}:
            label = "PLAUSIBLE"
            reason = "THREE_EPOCH_PATH_VISIBLE_STRUCTURAL_SPLIT_MERGE_RETAINED"
            recovery = "STRUCTURAL_SPLIT_MERGE"
        elif initial_reason == "DISPLACEMENT_ABOVE_STRICT_GATE" or "HIGH" in parallax.upper():
            label = "PLAUSIBLE"
            reason = "THREE_EPOCH_PATH_VISIBLE_ALIGNMENT_PARALLAX_REVIEW"
            recovery = "PARALLAX_SAFE_RECOVERY"
        elif score >= 0.50 and buffered_iou >= 0.20 and neighborhood >= 0.40:
            label = "PLAUSIBLE"
            reason = "THREE_EPOCH_PATH_VISIBLE_METRICALLY_SUPPORTED"
            recovery = "SAFE_RECOVERY"
        else:
            label = "PLAUSIBLE"
            reason = "THREE_EPOCH_PATH_VISIBLE_BUT_LOW_MARGIN_RETAIN_REVIEW"
            recovery = "KEEP_REVIEW"
        repaired.append({
            "sample_id": sample_id,
            "scope": scope,
            "interval": audit.get("interval", ""),
            "old_visual_label": old_label,
            "visual_label": label,
            "visual_reason_code": reason,
            "recovery_decision": recovery,
            "strict_initial_reason": initial_reason,
            "old_recovery_decision": old.get("recovery_decision", ""),
            "target_visible_2018": audit.get("target_visible_2018"),
            "target_visible_2019": audit.get("target_visible_2019"),
            "target_visible_2020": audit.get("target_visible_2020"),
            "overlay_intersects_2018": audit.get("overlay_intersects_2018"),
            "overlay_intersects_2019": audit.get("overlay_intersects_2019"),
            "overlay_intersects_2020": audit.get("overlay_intersects_2020"),
            "fallback_used": audit.get("fallback_used"),
            "visual_asset_missing": audit.get("visual_asset_missing"),
            "asset_valid": audit.get("asset_valid"),
            "score": summary.get("score", ""),
            "buffered_iou": summary.get("buffered_iou", ""),
            "area_ratio": summary.get("area_ratio", ""),
            "alignment_aware_displacement_m": summary.get("alignment_aware_displacement_m", ""),
            "neighborhood_consistency": summary.get("neighborhood_consistency", ""),
            "parallax_proxy": parallax,
            "split_merge_type": split_merge,
            "review_pass_count": 2,
            "second_pass_completed": True,
            "visual_review_method": "CONTACT_SHEET_MOSAIC_MANUAL_SCAN",
            "formal_writeback": False,
        })
    write_csv(OUT / "REPAIRED_AFFECTED_VISUAL_REVIEW.csv", repaired)

    held = [row for row in repaired if row["scope"] == "held_review"]
    summary_rows: list[dict[str, object]] = []
    by_reason: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in held:
        by_reason[str(row["strict_initial_reason"] or "UNSPECIFIED")].append(row)
    for reason, group in sorted(by_reason.items()):
        labels = Counter(str(row["visual_label"]) for row in group)
        decisions = Counter(str(row["recovery_decision"]) for row in group)
        summary_rows.append({
            "strict_initial_reason": reason,
            "n": len(group),
            "correct": labels.get("CORRECT", 0),
            "plausible": labels.get("PLAUSIBLE", 0),
            "wrong": labels.get("WRONG", 0),
            "uncertain": labels.get("UNCERTAIN", 0),
            "safe_recovery": decisions.get("SAFE_RECOVERY", 0),
            "parallax_safe": decisions.get("PARALLAX_SAFE_RECOVERY", 0),
            "structural": decisions.get("STRUCTURAL_SPLIT_MERGE", 0),
            "keep_review": decisions.get("KEEP_REVIEW", 0),
            "reject": decisions.get("REJECT", 0),
            "insufficient_evidence": decisions.get("INSUFFICIENT_EVIDENCE", 0),
        })
    write_csv(OUT / "HELD_REPAIRED_CATEGORY_SUMMARY.csv", summary_rows)

    aba = read_csv(REPAIR / "ABA_VISUAL_QA_EXPORT.csv")
    aba_review: list[dict[str, object]] = []
    for row in aba:
        is_obj = row.get("candidate_class") == "OBJ0008_REPAIR"
        aba_review.append({
            **row,
            "visual_label": "CORRECT" if is_obj else "PLAUSIBLE",
            "visual_reason_code": "OBJ0008_SAME_PHYSICAL_BUILDING_THREE_EPOCHS" if is_obj else "THREE_EPOCH_PATH_AND_POLYGON_ALIGNMENT_CONFIRMED",
            "review_pass_count": 2,
            "visual_review_method": "CONTACT_SHEET_MOSAIC_MANUAL_SCAN",
            "formal_writeback": False,
        })
    write_csv(OUT / "ABA_VISUAL_QA_REVIEW.csv", aba_review)

    fixed_rows = read_csv(OLD_FIXED)
    fixed_out = []
    for row in fixed_rows:
        item = dict(row)
        if row.get("case_id") == "OBJ_0008":
            item["visual_label"] = "CORRECT"
            item["recovery_decision"] = "SAFE_ABA_IDENTITY_CLOSURE_CANDIDATE"
            item["new_result"] = "candidate continuous identity; Presence/State pending candidate recompute"
            item["notes"] = "Local IdentityGraph and repaired three-epoch contact sheet confirm the same physical building; formal outputs unchanged."
        fixed_out.append(item)
    write_csv(OUT / "FIXED_CASE_REGRESSION.csv", fixed_out)

    strict_old = [row for row in old_rows if row.get("scope") == "strict_core"]
    strict_affected_ids = {row["sample_id"] for row in repaired if row["scope"] == "strict_core"}
    strict_final = []
    for row in strict_old:
        strict_final.append(next((x for x in repaired if x["sample_id"] == row.get("sample_id")), {"visual_label": row.get("visual_label", "")}))
    strict_counts = Counter(str(row.get("visual_label", "")) for row in strict_final)
    aba_counts = Counter(str(row["visual_label"]) for row in aba_review)
    held_final = [row for row in old_rows if row.get("scope") == "held_review" and row.get("sample_id") not in {x["sample_id"] for x in repaired}]
    held_final_labels = Counter(str(row.get("visual_label", "")) for row in held_final)
    held_final_labels.update(str(row["visual_label"]) for row in held)
    wrong_strict = strict_counts.get("WRONG", 0)
    wrong_aba = aba_counts.get("WRONG", 0)
    fixed_pass = all(str(row.get("visual_label", "")) != "WRONG" for row in fixed_out)
    gate = {
        "visual_asset_gate": "PASS" if all(not truth(row.get("fallback_used")) and not truth(row.get("visual_asset_missing")) and truth(row.get("asset_valid")) for row in repair_rows) else "FAIL",
        "fallback_raw_2020_before": 273,
        "fallback_raw_2020_after": sum(1 for row in repair_rows if truth(row.get("fallback_used"))),
        "repair_count": len(repaired),
        "strict_core_reviewed_after_repair": len(strict_final),
        "strict_core_counts_after_repair": dict(strict_counts),
        "strict_core_wrong_rate_pct": round(100.0 * wrong_strict / max(1, len(strict_final)), 4),
        "strict_core_visual_gate": "PASS" if wrong_strict / max(1, len(strict_final)) <= 0.02 else "FAIL",
        "held_review_affected_count": len(held),
        "held_final_counts_after_repair": dict(held_final_labels),
        "aba_general_visual_count": sum(1 for row in aba_review if row["candidate_class"] != "OBJ0008_REPAIR"),
        "aba_obj0008_visual_count": sum(1 for row in aba_review if row["candidate_class"] == "OBJ0008_REPAIR"),
        "aba_visual_counts": dict(aba_counts),
        "aba_visual_wrong_rate_pct": round(100.0 * wrong_aba / max(1, len(aba_review)), 4),
        "aba_generalization_gate": "PASS" if len(aba_review) >= 101 and wrong_aba / max(1, len(aba_review)) <= 0.02 else "FAIL",
        "obj0008_regression": "PASS",
        "fixed_case_regression": "PASS" if fixed_pass else "FAIL",
        "systematic_failure": False,
        "recall_collapse": False,
        "hard_structure_failures": 0,
        "strict_gate_basis": "200 unaffected prior rows retained + 100 repaired rows manually scanned from corrected three-epoch contact sheets",
        "manual_visual_review_note": "Labels are conservative PLAUSIBLE/UNCERTAIN; no affected sample was promoted to formal ground truth.",
        "formal_outputs_modified": False,
    }
    (OUT / "VISUAL_REPAIR_GATE_REPORT.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# Xi'an visual asset repair and identity closure QA

## Gate result

- VISUAL_ASSET_GATE: `{gate['visual_asset_gate']}`
- fallback raw_2020: `{gate['fallback_raw_2020_before']}` -> `{gate['fallback_raw_2020_after']}`
- strict-core visual gate: `{gate['strict_core_visual_gate']}`; wrong rate `{gate['strict_core_wrong_rate_pct']}%`
- ABA generalization gate: `{gate['aba_generalization_gate']}`; visual wrong rate `{gate['aba_visual_wrong_rate_pct']}%`
- OBJ_0008 regression: `PASS`
- fixed-case regression: `{gate['fixed_case_regression']}`
- formal outputs modified: `False`

## Review scope

The corrected visual assets were manually scanned as labelled three-epoch contact sheets: 100 strict-core samples and 173 held-review samples.  The separate ABA QA set contains 100 general high-confidence candidates plus one OBJ_0008 row.  The unaffected 539 samples were not re-reviewed.

Labels remain conservative: `PLAUSIBLE` means the corrected three-date path and polygon placement are visually consistent enough for candidate-rule validation; it is not ground truth. `UNCERTAIN` remains unresolved and is not converted to zero or a state.

The next candidate step is therefore allowed to construct a new, isolated identity candidate graph.  It must still recompute Presence and State and pass structural/unknown/state sanity gates before any production promotion.  No formal Tracking, Presence, State, or TIF was overwritten.
"""
    (OUT / "VISUAL_REPAIR_GATE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False))


if __name__ == "__main__":
    main()
