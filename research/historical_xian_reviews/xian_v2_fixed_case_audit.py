from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--fixed-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    obj = pd.read_csv(
        args.v2 / "FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv",
        usecols=["object_uid", "physical_track_id", "compound_track_id"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    tracks = pd.read_csv(
        args.v2 / "FINAL_CANDIDATE_PHYSICAL_TRACKS_V2.csv",
        encoding="utf-8-sig",
        low_memory=False,
    ).set_index("physical_track_id")
    state = pd.read_csv(
        args.v2 / "STATE_V2_FROM_SCRATCH.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )

    by_uid = obj.groupby("object_uid", sort=False)["physical_track_id"].agg(lambda s: sorted(set(s)))
    rows = []
    for case_dir in sorted(p for p in args.fixed_root.iterdir() if p.is_dir()):
        summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8-sig"))
        case_id = summary.get("case_id", case_dir.name.removeprefix("CASE_"))
        requested = list(summary.get("current_object_uids") or [])
        mapped = {}
        missing = []
        for uid in requested:
            vals = by_uid.get(uid, [])
            if vals:
                mapped[uid] = vals
            else:
                missing.append(uid)
        physical_ids = sorted({x for vals in mapped.values() for x in vals})
        seqs = []
        states = []
        for pid in physical_ids:
            if pid in tracks.index:
                seqs.append(f"{pid}:{tracks.loc[pid, 'presence_sequence']}")
            m = state["physical_track_id"].eq(pid)
            for value in state.loc[m, "state"].astype(str).tolist():
                states.append(f"{pid}:{value}")
        if not requested and summary.get("unknown_reason") == "FIELD_NON_BUILDING":
            status = "NON_BUILDING_NO_MAPPING"
            note = "fixed case is explicitly non-building; no synthetic track created"
        elif not requested:
            status = "NO_CASE_OBJECT_KEYS"
            note = "fixed-case package has no object key; no PASS fabricated"
        elif missing:
            status = "PARTIAL_OBJECT_MAPPING"
            note = f"missing_object_uids={len(missing)}; no PASS fabricated"
        elif case_id == "OBJ_0008" and len(physical_ids) == 1:
            status = "PASS_CANDIDATE_ONE_PHYSICAL_TRACK"
            note = "all fixed-case canonical objects resolve to one V2 physical track"
        else:
            status = "MAPPED_REQUIRES_VISUAL_REVIEW"
            note = "object mapping exists, but fixed-case scientific acceptance remains visual-review gated"
        rows.append(
            {
                "case_id": case_id,
                "requested_object_count": len(requested),
                "mapped_object_count": len(mapped),
                "missing_object_count": len(missing),
                "v2_physical_track_ids": ";".join(physical_ids),
                "v2_presence_sequences": ";".join(seqs),
                "v2_states": ";".join(states),
                "old_result": summary.get("old_result", ""),
                "candidate_v1_result": summary.get("new_result", ""),
                "v2_fixed_case_status": status,
                "note": note,
            }
        )
    out = pd.DataFrame(rows).sort_values("case_id")
    out.to_csv(args.out / "FIXED_CASE_REGRESSION_V2.csv", index=False, encoding="utf-8-sig")
    receipt = {
        "fixed_case_count": len(out),
        "pass_candidate_count": int(out.v2_fixed_case_status.eq("PASS_CANDIDATE_ONE_PHYSICAL_TRACK").sum()),
        "mapped_requires_visual_review_count": int(out.v2_fixed_case_status.eq("MAPPED_REQUIRES_VISUAL_REVIEW").sum()),
        "gate": "REVIEW_REQUIRED",
        "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
    }
    (args.out / "FIXED_CASE_REGRESSION_V2_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
