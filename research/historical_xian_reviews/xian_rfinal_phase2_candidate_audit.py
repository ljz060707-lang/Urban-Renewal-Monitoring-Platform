from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


BASE = Path("/data/xian_r4_100km2_20260914")
V2 = BASE / "output/16_candidate_identitygraph_v2_20260916"
FORMAL = BASE / "output/西安_R4_辅助影像接入前_三期三态候选版_20260914"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "output/19_rfinal_phase2_observability_diagnostic_20260916"
    q = pd.read_csv(out / "phase2_fresh_observability_candidates.csv", low_memory=False)
    q["area_m2"] = pd.to_numeric(q["area_m2"], errors="coerce").fillna(0.0)

    tr = pd.read_csv(V2 / "FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv", usecols=["physical_track_id", "object_uid", "building_id", "identity_sources"], low_memory=False)
    tr["physical_track_id"] = tr["physical_track_id"].astype(str)
    tr["object_uid"] = tr["object_uid"].astype(str)
    tr["building_id"] = tr["building_id"].astype(str)
    track_buildings = tr.groupby("physical_track_id")["building_id"].apply(lambda s: ";".join(sorted(set(s)))).to_dict()
    track_objects = tr.groupby("physical_track_id")["object_uid"].apply(lambda s: set(s)).to_dict()

    presence = pd.read_csv(FORMAL / "05_presence_state/PRESENCE_1_0_U.csv", usecols=["building_id", "scene_key", "presence", "date_support_status", "observability_status"], low_memory=False)
    presence["building_id"] = presence["building_id"].astype(str)
    pidx = {(str(r.building_id), str(r.scene_key)): r for r in presence.itertuples(index=False)}

    held = pd.read_csv(V2 / "HELD_FULL_RECOVERY_DECISION.csv", usecols=["source_object_id", "target_object_id", "decision", "identity_conflict", "split_merge_type"], low_memory=False)
    held = held[held["decision"].eq("KEEP_REVIEW")]
    held_object_ids = set(held["source_object_id"].astype(str)) | set(held["target_object_id"].astype(str))
    held_conflict_tracks = {pid for pid, objs in track_objects.items() if objs & held_object_ids}

    rows = []
    for r in q.itertuples(index=False):
        pid = str(r.physical_track_id)
        buildings = [x for x in track_buildings.get(pid, "").split(";") if x]
        entries = [pidx[(b, str(r.scene_key))] for b in buildings if (b, str(r.scene_key)) in pidx]
        statuses = sorted(set(str(e.date_support_status) for e in entries))
        obs = sorted(set(str(e.observability_status) for e in entries))
        identity_sources = str(r.identity_sources) if not pd.isna(r.identity_sources) else ""
        source_present = bool(identity_sources.strip())
        no_compound_or_parallax = not any(k in identity_sources.upper() for k in ("COMPOUND", "PARALLAX"))
        date_assigned = bool(entries) and all(str(e.date_support_status) == "FOOTPRINT_ASSIGNED" for e in entries)
        no_held_endpoint = pid not in held_conflict_tracks
        no_identity_competitor = source_present and no_compound_or_parallax and no_held_endpoint
        promotion_eligible = bool(r.formal_zero_candidate) and date_assigned and no_identity_competitor
        reason = []
        if not bool(r.formal_zero_candidate): reason.append("PIXEL_NOT_CLEAR")
        if not entries: reason.append("NO_FORMAL_DATE_SUPPORT_ROW")
        elif not date_assigned: reason.append("DATE_SUPPORT_NOT_FOOTPRINT_ASSIGNED")
        if not source_present: reason.append("IDENTITY_SOURCE_EMPTY")
        if not no_compound_or_parallax: reason.append("COMPOUND_OR_PARALLAX")
        if not no_held_endpoint: reason.append("HELD_REVIEW_ENDPOINT")
        rows.append({**r._asdict(), "building_ids": ";".join(buildings), "formal_date_support_statuses": ";".join(statuses), "formal_observability_statuses": ";".join(obs), "date_assigned_gate": date_assigned, "identity_source_present_gate": source_present, "no_compound_or_parallax_gate": no_compound_or_parallax, "no_held_endpoint_gate": no_held_endpoint, "no_identity_competitor_gate": no_identity_competitor, "promotion_eligible_zero_candidate": promotion_eligible, "audit_reason": ";".join(reason) if reason else "ALL_AUDIT_GATES_PASS_CANDIDATE_ONLY"})

    result = pd.DataFrame(rows)
    result.to_csv(out / "phase2_candidate_audit.csv", index=False, encoding="utf-8-sig")
    eligible = result[result["promotion_eligible_zero_candidate"]].copy()
    summary = result.groupby(["interval", "promotion_eligible_zero_candidate"], dropna=False).agg(rows=("physical_track_id", "size"), area_m2=("area_m2", "sum")).reset_index()
    summary.to_csv(out / "phase2_candidate_audit_summary.csv", index=False, encoding="utf-8-sig")
    receipt = {
        "phase": "PHASE_2_CANDIDATE_DATE_IDENTITY_AUDIT",
        "input_rows": int(len(result)),
        "promotion_eligible_candidate_rows": int(len(eligible)),
        "promotion_eligible_candidate_area_m2_by_interval": {str(k): float(v) for k, v in eligible.groupby("interval")["area_m2"].sum().items()},
        "audit_gates": {
            "fresh_pixel_clear": "REQUIRED",
            "formal_date_support_all_footprint_assigned": "REQUIRED",
            "identity_source_nonempty": "REQUIRED",
            "no_compound_or_parallax_source": "REQUIRED",
            "no_held_review_endpoint": "REQUIRED",
        },
        "held_review_tracks_excluded": int(len(held_conflict_tracks)),
        "formal_outputs_modified": False,
        "promotion_status": "NOT_PROMOTED_PENDING_TARGETED_VISUAL_QA_AND_STATE_DIFF",
    }
    (out / "PHASE2_CANDIDATE_AUDIT_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
