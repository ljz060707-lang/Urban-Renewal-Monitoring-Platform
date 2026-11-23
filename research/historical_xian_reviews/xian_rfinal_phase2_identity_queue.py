from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/data/xian_r4_100km2_20260914")
V2 = BASE / "output/16_candidate_identitygraph_v2_20260916"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "output/19_rfinal_phase2_observability_diagnostic_20260916"
    audit = pd.read_csv(out / "phase2_candidate_audit.csv", low_memory=False)
    audit = audit[audit["promotion_eligible_zero_candidate"].astype(str).str.lower().eq("true")].copy()
    audit["area_m2"] = pd.to_numeric(audit["area_m2"], errors="coerce").fillna(0.0)
    audit["physical_track_id"] = audit["physical_track_id"].astype(str)
    audit["scene_key"] = audit["scene_key"].astype(str)
    keys = audit[["physical_track_id", "scene_key", "interval", "presence_sequence", "area_m2", "x", "y"]].drop_duplicates()

    tr = pd.read_csv(V2 / "FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv", usecols=["physical_track_id", "object_uid"], low_memory=False)
    tr["physical_track_id"] = tr["physical_track_id"].astype(str)
    tr["object_uid"] = tr["object_uid"].astype(str)
    pid_objects = tr.groupby("physical_track_id")["object_uid"].apply(set).to_dict()
    object_pid = tr.drop_duplicates("object_uid").set_index("object_uid")["physical_track_id"].to_dict()
    selected_objects = set().union(*(pid_objects.get(pid, set()) for pid in keys["physical_track_id"]))

    use = [
        "record_type", "edge_stage", "source_object_id", "target_object_id", "from_scene", "to_scene",
        "score", "buffered_iou", "area_ratio", "alignment_aware_displacement_m", "local_neighborhood_consistency",
        "candidate_relation", "split_merge_type", "identity_conflict", "primary_source", "transition_eligible",
    ]
    graph_path = V2 / "FINAL_CANDIDATE_IDENTITY_GRAPH_V2.csv"
    pieces = []
    for chunk in pd.read_csv(graph_path, usecols=use, low_memory=False, chunksize=250000):
        chunk = chunk[chunk["source_object_id"].isin(selected_objects) | chunk["target_object_id"].isin(selected_objects)].copy()
        if not chunk.empty:
            pieces.append(chunk)
    if pieces:
        edges = pd.concat(pieces, ignore_index=True)
    else:
        edges = pd.DataFrame(columns=use)
    edges = edges[edges["source_object_id"].notna() & edges["target_object_id"].notna()].copy()
    for c in ["score", "buffered_iou", "area_ratio", "alignment_aware_displacement_m", "local_neighborhood_consistency"]:
        edges[c] = pd.to_numeric(edges[c], errors="coerce")
    edges["identity_conflict"] = edges["identity_conflict"].fillna("NO").astype(str).str.upper()
    edges["source_pid"] = edges["source_object_id"].map(object_pid)
    edges["target_pid"] = edges["target_object_id"].map(object_pid)

    rows = []
    for key in keys.itertuples(index=False):
        pid = str(key.physical_track_id); scene = str(key.scene_key)
        sub = edges[(edges["from_scene"].eq(scene) | edges["to_scene"].eq(scene)) & ((edges["source_pid"].eq(pid)) | (edges["target_pid"].eq(pid)))]
        candidates = []
        for r in sub.itertuples(index=False):
            source_scene = str(r.from_scene); target_scene = str(r.to_scene)
            if source_scene == scene:
                candidate_object = str(r.source_object_id); other_object = str(r.target_object_id); candidate_pid = r.source_pid; other_pid = r.target_pid
            elif target_scene == scene:
                candidate_object = str(r.target_object_id); other_object = str(r.source_object_id); candidate_pid = r.target_pid; other_pid = r.source_pid
            else:
                continue
            if str(other_pid) != pid or str(candidate_pid) == pid or not str(candidate_pid) or str(candidate_pid) == "nan":
                continue
            rec = {
                "candidate_object_id": candidate_object,
                "candidate_physical_track_id": str(candidate_pid),
                "source_object_id": str(r.source_object_id),
                "target_object_id": str(r.target_object_id),
                "from_scene": source_scene,
                "to_scene": target_scene,
                "score": float(r.score) if np.isfinite(r.score) else 0.0,
                "buffered_iou": float(r.buffered_iou) if np.isfinite(r.buffered_iou) else 0.0,
                "area_ratio": float(r.area_ratio) if np.isfinite(r.area_ratio) else 0.0,
                "displacement_m": float(r.alignment_aware_displacement_m) if np.isfinite(r.alignment_aware_displacement_m) else 9999.0,
                "neighborhood": float(r.local_neighborhood_consistency) if np.isfinite(r.local_neighborhood_consistency) else 0.0,
                "candidate_relation": str(r.candidate_relation),
                "split_merge_type": str(r.split_merge_type),
                "identity_conflict": str(r.identity_conflict),
                "primary_source": str(r.primary_source),
            }
            rec["regular_safe"] = bool(rec["score"] >= 0.58 and rec["buffered_iou"] >= 0.55 and rec["area_ratio"] >= 0.55 and rec["neighborhood"] >= 0.60 and rec["displacement_m"] <= 50.0 and rec["identity_conflict"] != "YES")
            rec["parallax_safe"] = bool(rec["score"] >= 0.55 and rec["buffered_iou"] >= 0.40 and rec["area_ratio"] >= 0.45 and rec["neighborhood"] >= 0.50 and 45.0 < rec["displacement_m"] <= 90.0 and rec["identity_conflict"] != "YES")
            candidates.append(rec)
        cdf = pd.DataFrame(candidates)
        if cdf.empty:
            rows.append({**key._asdict(), "candidate_edge_count": 0, "candidate_track_count": 0, "regular_safe_count": 0, "parallax_safe_count": 0, "best_score": 0.0, "best_iou": 0.0, "best_area_ratio": 0.0, "best_displacement_m": 9999.0, "best_neighborhood": 0.0, "identity_queue_status": "NO_GRAPH_EDGE"})
            continue
        cdf = cdf.drop_duplicates(["candidate_object_id", "candidate_physical_track_id", "source_object_id", "target_object_id"])
        regular = cdf[cdf["regular_safe"]]
        parallax = cdf[cdf["parallax_safe"]]
        if len(regular) == 1 and regular["candidate_physical_track_id"].nunique() == 1:
            status = "REGULAR_SAFE_EDGE_NEEDS_VISUAL_REVIEW"
        elif len(parallax) == 1 and parallax["candidate_physical_track_id"].nunique() == 1:
            status = "PARALLAX_SAFE_EDGE_NEEDS_VISUAL_REVIEW"
        elif len(regular) > 1 or len(parallax) > 1 or (cdf["identity_conflict"] == "YES").any() or cdf["candidate_relation"].str.contains("AMBIGUOUS", na=False).any():
            status = "AMBIGUOUS_OR_CONFLICT_RETAIN_UNKNOWN"
        else:
            status = "REVIEW_ONLY_NO_VALIDATED_EDGE"
        best = cdf.sort_values(["score", "buffered_iou", "neighborhood"], ascending=False).iloc[0]
        rows.append({
            **key._asdict(),
            "candidate_edge_count": int(len(cdf)),
            "candidate_track_count": int(cdf["candidate_physical_track_id"].nunique()),
            "regular_safe_count": int(len(regular)),
            "parallax_safe_count": int(len(parallax)),
            "best_score": float(best["score"]),
            "best_iou": float(best["buffered_iou"]),
            "best_area_ratio": float(best["area_ratio"]),
            "best_displacement_m": float(best["displacement_m"]),
            "best_neighborhood": float(best["neighborhood"]),
            "best_candidate_physical_track_id": str(best["candidate_physical_track_id"]),
            "best_candidate_object_id": str(best["candidate_object_id"]),
            "best_candidate_relation": str(best["candidate_relation"]),
            "best_split_merge_type": str(best["split_merge_type"]),
            "best_identity_conflict": str(best["identity_conflict"]),
            "identity_queue_status": status,
        })
    result = pd.DataFrame(rows).sort_values("area_m2", ascending=False)
    result.to_csv(out / "phase2_identity_recovery_queue.csv", index=False, encoding="utf-8-sig")
    status_summary = result.groupby(["interval", "identity_queue_status"], dropna=False).agg(rows=("physical_track_id", "size"), area_m2=("area_m2", "sum")).reset_index()
    status_summary.to_csv(out / "phase2_identity_recovery_queue_summary.csv", index=False, encoding="utf-8-sig")
    receipt = {
        "phase": "PHASE_2_IDENTITY_RECOVERY_QUEUE_AFTER_ALL_OBJECT_VISUAL_AUDIT",
        "input_candidate_rows": int(len(keys)),
        "graph_edges_loaded_for_selected_objects": int(len(edges)),
        "queue_rows": int(len(result)),
        "status_summary": status_summary.to_dict(orient="records"),
        "safe_edge_promotion": "NOT_RUN",
        "formal_outputs_modified": False,
        "rule": "composite alignment-aware score, buffered IoU, area ratio, neighborhood, displacement and conflict; no nearest-centroid-only promotion",
        "next_action": "visual-review highest-area REGULAR/PARALLAX safe candidates; retain ambiguous/split-merge competitors as Unknown",
    }
    (out / "PHASE2_IDENTITY_QUEUE_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
