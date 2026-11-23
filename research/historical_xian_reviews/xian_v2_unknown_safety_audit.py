from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    receipt = json.loads((args.v2 / "FINAL_CANDIDATE_RECEIPT_V2.json").read_text(encoding="utf-8-sig"))
    d = pd.read_csv(args.v2 / "RESIDUAL_UNKNOWN_PHASE_B_V2.csv", encoding="utf-8-sig", low_memory=False)
    rows = []
    pct_map = receipt["unknown_area_pct"]
    rationale = {
        "U05_UNOBSERVABLE": "no reliable absence/presence observation; cannot safely assign a formal state",
        "U01_IDENTITY_MULTIPLE_OR_NO_MATCH": "identity competitor or no-match remains; nearest candidate would violate IdentityGraph gate",
        "U07_COMPOUND_UNRESOLVED": "compound/split/merge structure not resolved; child-level state would be fabricated",
        "U08_INSUFFICIENT_EVIDENCE": "evidence is incomplete; requires targeted visual or supplemental observation",
        "U03_PARALLAX": "view-angle/parallax uncertainty remains; polygon cannot be warped into a new fact",
    }
    for (interval, cause), g in d.groupby(["interval", "unknown_cause"], dropna=False):
        area = float(g["area_m2"].sum())
        unknown_pct = float(pct_map[interval])
        denominator = area / (unknown_pct / 100.0) if unknown_pct else None
        rows.append(
            {
                "interval": interval,
                "unknown_cause": str(cause),
                "rows": int(len(g)),
                "area_m2": area,
                "share_of_residual_unknown_pct": area / float(d.loc[d.interval.eq(interval), "area_m2"].sum()) * 100.0,
                "current_unknown_area_pct": unknown_pct,
                "area_required_to_reach_5pct_m2": max(0.0, float(d.loc[d.interval.eq(interval), "area_m2"].sum()) - denominator * 0.05),
                "safe_to_auto_close": "NO",
                "closure_gate": "BLOCKED_PENDING_NEW_EVIDENCE",
                "reason": rationale.get(str(cause), "unclassified residual unknown"),
            }
        )
    out = pd.DataFrame(rows).sort_values(["interval", "area_m2"], ascending=[True, False])
    out.to_csv(args.out / "UNKNOWN_SAFETY_AUDIT_V2.csv", index=False, encoding="utf-8-sig")
    summary = {
        "unknown_lt5_gate": receipt["unknown_lt5_gate"],
        "residual_unknown_rows": int(len(d)),
        "residual_unknown_area_m2": float(d["area_m2"].sum()),
        "safe_auto_close_area_m2": 0.0,
        "dominant_blockers": [
            "U05_UNOBSERVABLE",
            "U01_IDENTITY_MULTIPLE_OR_NO_MATCH",
            "U07_COMPOUND_UNRESOLVED",
        ],
        "conclusion": "residual Unknown is not safely closable from the frozen three-date inputs and candidate identity evidence; do not force Presence 0 or formal State",
        "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
    }
    (args.out / "UNKNOWN_SAFETY_AUDIT_V2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
