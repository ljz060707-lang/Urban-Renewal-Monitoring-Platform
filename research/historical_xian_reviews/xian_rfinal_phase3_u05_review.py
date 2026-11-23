from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("/data/xian_r4_100km2_20260914")
    out = root / "output/22_rfinal_phase3_u05_absence_diagnostic_20260917"
    rows = [
        {
            "physical_track_id": "PTRK_R4V2_00008046",
            "interval": "2018_2019",
            "area_m2": 27477.22119654819,
            "absence_diagnostic": "POTENTIAL_RELIABLE_CLEAR_ABSENCE_REVIEW",
            "visual_label": "REJECT_GEOMETRY_FALSE_POSITIVE",
            "visual_reason": "2018 active red geometry follows orchard/field texture and dirt paths, not a building footprint; clear 2019 farmland therefore cannot be interpreted as demolition",
            "promotion_decision": "DO_NOT_PROMOTE",
            "formal_outputs_modified": "false",
        }
    ]
    fields = list(rows[0])
    with (out / "PHASE3_U05_TARGETED_VISUAL_REVIEW.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write(",".join(fields) + "\n")
        handle.write(",".join(str(rows[0].get(field, "")).replace(",", ";") for field in fields) + "\n")
    receipt = {
        "phase": "PHASE_3_U05_TARGETED_VISUAL_REVIEW",
        "status": "FAIL_REVIEW_REQUIRED",
        "candidate_count": 1,
        "rejected_count": 1,
        "promotion_count": 0,
        "formal_outputs_modified": False,
        "disposition": "ROLLBACK_CANDIDATE_ONLY",
        "reason": "the only strict clear-absence candidate is a B0 geometry false positive over orchard/field texture",
        "next_action": "retain U05 Unknown; continue identity/geometry diagnosis and do not infer demolition from pixel clarity alone",
    }
    (out / "PHASE3_U05_TARGETED_VISUAL_REVIEW_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
