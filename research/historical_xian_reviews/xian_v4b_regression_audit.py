"""Read-only diagnostics for the Xi'an V4B -> candidate regression.

This script only reads frozen/formal inputs and the isolated candidate-v1
output.  It writes an isolated audit directory; it never edits Geometry,
formal Tracking, formal Presence/State, or formal products.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


SCENES = ("xa180429", "xa190101", "xa201231")
TRANSITIONS = (
    ("2018_2019", "xa180429", "xa190101"),
    ("2019_2020", "xa190101", "xa201231"),
)


def s(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def f(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def b(value: Any) -> bool:
    return s(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_values(value: Any) -> list[str]:
    text = s(value).strip()
    if not text:
        return []
    return [x.strip().strip("'\"") for x in re.split(r"[;,|]", text) if x.strip()]


def parse_dict(value: Any) -> dict[str, Any]:
    text = s(value).strip()
    if not text:
        return {}
    try:
        value = ast.literal_eval(text)
        return value if isinstance(value, dict) else {}
    except (SyntaxError, ValueError):
        return {}


def component_union_area(row: dict[str, Any], interval: str) -> float:
    d = parse_dict(row.get("scene_union_area_m2"))
    if not d:
        d = parse_dict(row.get("scene_area_m2"))
    a, z = ("xa180429", "xa190101") if interval == "2018_2019" else ("xa190101", "xa201231")
    return max(f(d.get(a)), f(d.get(z)), f(row.get("union_area_m2")))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in rows)


def edge_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        s(row.get("from_scene", row.get("source_epoch"))),
        s(row.get("to_scene", row.get("target_epoch"))),
        s(row.get("source_object_id")),
        s(row.get("target_object_id")),
    )


def strict_edges(formal_edges: Path) -> set[tuple[str, str, str, str]]:
    columns = [
        "from_scene", "to_scene", "source_object_id", "target_object_id",
        "score", "buffered_iou", "area_ratio", "local_neighborhood_consistency",
        "alignment_aware_displacement_m", "candidate_relation", "split_merge_type",
        "identity_conflict", "accepted_for_identity", "score_gap_source",
        "score_gap_target", "mutual_best",
    ]
    rows: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(formal_edges).iter_batches(batch_size=250_000, columns=columns):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            row = {k: data[k][i] for k in columns}
            if not b(row["accepted_for_identity"]):
                continue
            if b(row["identity_conflict"]):
                continue
            if s(row["candidate_relation"]) != "1:1":
                continue
            if s(row["split_merge_type"]) not in {"", "NONE"}:
                continue
            if f(row["score"]) < 0.60 or f(row["buffered_iou"]) < 0.55 or f(row["area_ratio"]) < 0.55:
                continue
            if f(row["local_neighborhood_consistency"]) < 0.60 or f(row["alignment_aware_displacement_m"]) > 50:
                continue
            rows.append(row)
    src = Counter((s(r["from_scene"]), s(r["source_object_id"])) for r in rows)
    dst = Counter((s(r["to_scene"]), s(r["target_object_id"])) for r in rows)
    keep: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = edge_key(row)
        unique = src[(s(row["from_scene"]), s(row["source_object_id"]))] == 1 and dst[(s(row["to_scene"]), s(row["target_object_id"]))] == 1
        margin = min(f(row["score_gap_source"]), f(row["score_gap_target"])) >= 0.15
        if unique or (margin and b(row["mutual_best"])):
            keep.add(key)
    src2 = Counter((a, b_, c) for a, b_, c, _ in keep)
    dst2 = Counter((a, b_, d) for a, b_, _, d in keep)
    return {key for key in keep if src2[(key[0], key[1], key[2])] == 1 and dst2[(key[0], key[1], key[3])] == 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--v4b-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise RuntimeError(f"refusing to overwrite existing directory: {out}")
    out.mkdir(parents=True)

    formal = args.formal_root
    v4b = args.v4b_root
    cand = args.candidate_root
    repair = args.repair_root

    tracking = pd.read_csv(formal / "04_identity_graph" / "tracking_objects_unfiltered.csv", encoding="utf-8-sig", low_memory=False,
                           usecols=["scene_key", "building_id", "object_uid", "area_m2"])
    tracking["object_uid"] = tracking["object_uid"].astype(str)
    tracking["scene_key"] = tracking["scene_key"].astype(str)
    tracking["area_m2"] = pd.to_numeric(tracking["area_m2"], errors="coerce").fillna(0.0)
    object_map = tracking.set_index("object_uid").to_dict("index")

    candidate = pd.read_csv(cand / "FINAL_CANDIDATE_TRACKING_OBJECTS.csv", encoding="utf-8-sig", low_memory=False)
    candidate["object_uid"] = candidate["object_uid"].astype(str)
    candidate["physical_track_id"] = candidate["physical_track_id"].astype(str)
    candidate["identity_sources"] = candidate["identity_sources"].fillna("").astype(str)
    candidate_by_uid = candidate.set_index("object_uid").to_dict("index")
    candidate_by_pid = candidate.groupby("physical_track_id", sort=False)

    presence = pd.read_csv(cand / "CANDIDATE_PRESENCE_FROM_SCRATCH.csv", encoding="utf-8-sig", low_memory=False)
    presence["physical_track_id"] = presence["physical_track_id"].astype(str)
    presence_by_pid_scene = {(s(r.physical_track_id), s(r.scene_key)): r._asdict() for r in presence.itertuples(index=False)}
    tracks = pd.read_csv(cand / "FINAL_CANDIDATE_PHYSICAL_TRACKS.csv", encoding="utf-8-sig", low_memory=False)
    tracks["physical_track_id"] = tracks["physical_track_id"].astype(str)
    track_by_pid = tracks.set_index("physical_track_id").to_dict("index")
    states = pd.read_csv(cand / "CANDIDATE_STATE_EVENTS_FROM_SCRATCH.csv", encoding="utf-8-sig", low_memory=False)
    states["physical_track_id"] = states["physical_track_id"].astype(str)
    state_by_pid_interval = {(s(r.physical_track_id), s(r.interval)): r._asdict() for r in states.itertuples(index=False)}

    comp_cols = [
        "component_id", "canonical_compound_track_id", "component_status", "node_count", "scene_count",
        "scene_object_count_2018", "scene_object_count_2019", "scene_object_count_2020", "scene_union_area_m2",
        "structural_types", "candidate_relations", "presence_values_2018", "presence_values_2019",
        "presence_values_2020", "object_uids", "resolution_reasons", "identity_confidence", "union_area_m2",
    ]
    compounds = pd.read_csv(v4b / "COMPOUND_TRACKS.csv", encoding="utf-8-sig", low_memory=False, usecols=lambda c: c in comp_cols)
    resolved = compounds[compounds["component_status"].isin(["RESOLVED_CONTINUOUS_IDENTITY_COMPONENT", "RESOLVED_GAP_RETURN"])].copy()

    # Exact V4B resolved-state lookup, when available in the resolved detail tables.
    resolved_state: dict[tuple[str, str], str] = {}
    for name in ("RESOLVED_SPLIT_MERGE.csv", "RESOLVED_MULTIPLE_CANDIDATES.csv", "RESOLVED_GAP_RETURN.csv"):
        path = v4b / name
        if not path.exists():
            continue
        detail = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, usecols=lambda c: c in {"component_id", "transition", "resolved_formal_state"})
        for row in detail.itertuples(index=False):
            cid, interval, state = s(getattr(row, "component_id", "")), s(getattr(row, "transition", "")), s(getattr(row, "resolved_formal_state", ""))
            if cid and interval and state:
                resolved_state[(cid, interval)] = state

    component_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    reason_area_components: defaultdict[str, dict[str, float]] = defaultdict(lambda: {"components": 0.0, "area_m2": 0.0})
    reason_objects: Counter[str] = Counter()
    expected_sources = {
        "RESOLVED_CONTINUOUS_IDENTITY_COMPONENT": "V4B_STRUCTURAL_CLOSURE|COMPOUND",
        "RESOLVED_GAP_RETURN": "V4B_STRUCTURAL_CLOSURE|GAP_RETURN",
    }

    def component_presence(cid_uids: list[str], scene: str) -> str:
        vals = []
        for uid in cid_uids:
            row = candidate_by_uid.get(uid)
            if row is None:
                continue
            p = presence_by_pid_scene.get((s(row.get("physical_track_id")), scene))
            if p:
                vals.append(s(p.get("presence")))
        if "1" in vals:
            return "1"
        if "U" in vals:
            return "U"
        if vals and all(v == "0" for v in vals):
            return "0"
        return "U"

    def fallback_v4b_state(row: dict[str, Any], interval: str) -> str:
        exact = resolved_state.get((s(row.get("component_id")), interval), "")
        if exact:
            return exact
        if interval == "2018_2019":
            if f(row.get("scene_object_count_2018")) <= 0 or f(row.get("scene_object_count_2019")) <= 0:
                return "NOT_APPLICABLE"
        else:
            if f(row.get("scene_object_count_2019")) <= 0 or f(row.get("scene_object_count_2020")) <= 0:
                return "NOT_APPLICABLE"
        a, z = (s(row.get("presence_values_2018")), s(row.get("presence_values_2019"))) if interval == "2018_2019" else (s(row.get("presence_values_2019")), s(row.get("presence_values_2020")))
        vals = parse_values(a) + parse_values(z)
        if "U" in vals:
            return "STABLE" if "1" in vals else "UNKNOWN"
        if "0" in vals and "1" in vals:
            return "ADDITION_OR_DEMOLITION"
        if vals and all(v == "1" for v in vals):
            return "STABLE"
        return "UNKNOWN"

    for raw in resolved.to_dict("records"):
        cid = s(raw.get("component_id"))
        uids = [u for u in s(raw.get("object_uids")).split(";") if u]
        member_rows = [candidate_by_uid.get(uid) for uid in uids]
        member_rows = [r for r in member_rows if r is not None]
        missing_uids = [uid for uid in uids if uid not in candidate_by_uid]
        pids = sorted({s(r.get("physical_track_id")) for r in member_rows if s(r.get("physical_track_id"))})
        identity_sources = sorted({source for r in member_rows for source in s(r.get("identity_sources")).split(";") if source})
        structural = s(raw.get("structural_types"))
        is_compound = any(x in structural.upper() for x in ("SPLIT", "MERGE")) or "SPLIT" in s(raw.get("candidate_relations")).upper() or "MERGE" in s(raw.get("candidate_relations")).upper()
        expected_source = expected_sources.get(s(raw.get("component_status")), "V4B_STRUCTURAL_CLOSURE")
        comp_area = max(component_union_area(raw, interval) for interval, _, _ in TRANSITIONS)
        candidate_presence = "-".join(component_presence(uids, scene) for scene in SCENES)
        candidate_states: dict[str, str] = {}
        regression_intervals: list[str] = []
        causes: set[str] = set()
        for interval, source_scene, target_scene in TRANSITIONS:
            expected = fallback_v4b_state(raw, interval)
            states_here = [s(state_by_pid_interval.get((pid, interval), {}).get("state")) for pid in pids]
            states_here = [x for x in states_here if x]
            if not states_here:
                observed = "MISSING"
            elif len(set(states_here)) == 1:
                observed = states_here[0]
            else:
                observed = "MIXED"
            candidate_states[interval] = observed
            regressed = expected not in {"UNKNOWN", "", "NOT_APPLICABLE"} and (observed in {"UNKNOWN_CONFLICT", "MISSING", "MIXED"} or len(pids) != 1)
            if not regressed:
                continue
            regression_intervals.append(interval)
            if missing_uids:
                causes.add("R01_V4B_EDGE_NOT_IMPORTED")
            if not pids:
                causes.add("R02_COMPONENT_IMPORTED_BUT_NOT_PROMOTED")
            elif len(pids) > 1:
                causes.add("R04_PRESENCE_REBUILT_AS_SEPARATE_CHILD_TRACKS" if is_compound else "R02_COMPONENT_IMPORTED_BUT_NOT_PROMOTED")
            if observed in {"UNKNOWN_CONFLICT", "MISSING", "MIXED"}:
                if len(pids) == 1 and not any("COMPOUND" in x or "V4B" in x or "GAP_RETURN" in x for x in identity_sources):
                    causes.add("R03_COMPOUND_TRACK_NOT_CONSUMED_BY_PRESENCE")
                elif any("STRICT_CORE" in x for x in identity_sources) and not any("COMPOUND" in x or "V4B" in x for x in identity_sources):
                    causes.add("R06_STRICT_CORE_PRECEDENCE_OVERRODE_COMPONENT")
                else:
                    causes.add("R05_STATE_DECODER_REQUIRES_CHILD_1TO1")
            if not causes:
                causes.add("R10_OTHER")
        primary = sorted(causes)[0] if causes else ""
        promoted = len(pids) == 1 and not missing_uids
        component_bug = "YES" if s(raw.get("component_status")).startswith("RESOLVED") and len(pids) != 1 else "NO"
        component_rows.append({
            "component_id": cid,
            "canonical_compound_track_id": s(raw.get("canonical_compound_track_id")),
            "component_status": s(raw.get("component_status")),
            "member_count_2018": s(raw.get("scene_object_count_2018")),
            "member_count_2019": s(raw.get("scene_object_count_2019")),
            "member_count_2020": s(raw.get("scene_object_count_2020")),
            "object_uid_count": len(uids),
            "candidate_object_count": len(member_rows),
            "missing_object_count": len(missing_uids),
            "candidate_physical_track_count": len(pids),
            "candidate_physical_track_ids": ";".join(pids),
            "promoted_to_physical_identity": "YES" if promoted else "NO",
            "physical_track_count_after_promotion": len(pids),
            "presence_sequence": candidate_presence,
            "state_2018_2019": candidate_states.get("2018_2019", ""),
            "state_2019_2020": candidate_states.get("2019_2020", ""),
            "v4b_expected_state_2018_2019": fallback_v4b_state(raw, "2018_2019"),
            "v4b_expected_state_2019_2020": fallback_v4b_state(raw, "2019_2020"),
            "identity_sources_present": ";".join(identity_sources),
            "identity_sources_missing": expected_source if not all(x in identity_sources for x in expected_source.split("|")) else "",
            "regression_intervals": ";".join(regression_intervals),
            "regression_reason": primary,
            "component_promotion_bug": component_bug,
            "component_union_area_m2": comp_area,
        })
        if not regression_intervals:
            continue
        for uid in uids:
            c = candidate_by_uid.get(uid, {})
            pid = s(c.get("physical_track_id"))
            for interval in regression_intervals:
                st = state_by_pid_interval.get((pid, interval), {})
                pseq = s(track_by_pid.get(pid, {}).get("presence_sequence"))
                reason = primary
                diff_rows.append({
                    "component_id": cid,
                    "old_compound_track": s(raw.get("canonical_compound_track_id")),
                    "v4b_component_status": s(raw.get("component_status")),
                    "v4b_resolution_reasons": s(raw.get("resolution_reasons")),
                    "v4b_structural_types": structural,
                    "old_object_uid": uid,
                    "old_scene": s(object_map.get(uid, {}).get("scene_key")),
                    "old_area_m2": f(object_map.get(uid, {}).get("area_m2")),
                    "final_physical_track_id": pid,
                    "final_presence_sequence": pseq,
                    "final_presence": pseq,
                    "interval": interval,
                    "v4b_expected_state": fallback_v4b_state(raw, interval),
                    "final_state": s(st.get("state")),
                    "final_unknown_reason": s(st.get("state_reason")),
                    "identity_sources_present": ";".join(identity_sources),
                    "identity_sources_missing": expected_source if not all(x in identity_sources for x in expected_source.split("|")) else "",
                    "state_decoder_path": s(st.get("state_reason")),
                    "candidate_physical_track_count": len(pids),
                    "component_union_area_m2": component_union_area(raw, interval),
                    "regression_reason": reason,
                })
                reason_objects[reason] += 1
        for reason in causes or {"R10_OTHER"}:
            reason_area_components[reason]["components"] += 1
            reason_area_components[reason]["area_m2"] += component_union_area(raw, regression_intervals[0]) if regression_intervals else 0.0

    component_fields = list(component_rows[0]) if component_rows else ["component_id"]
    write_csv(out / "V4B_COMPONENT_PROMOTION_AUDIT.csv", component_rows, component_fields)
    diff_fields = list(diff_rows[0]) if diff_rows else ["component_id"]
    write_csv(out / "V4B_REGRESSION_OBJECT_DIFF.csv", diff_rows, diff_fields)
    total_reg_area = sum(v["area_m2"] for v in reason_area_components.values())
    summary_rows = []
    for reason in sorted(set(reason_area_components) | set(reason_objects)):
        v = reason_area_components[reason]
        summary_rows.append({
            "regression_reason": reason,
            "component_count": int(v["components"]),
            "object_interval_row_count": int(reason_objects[reason]),
            "component_area_m2": v["area_m2"],
            "component_area_pct_of_reason_components": 100.0 * v["area_m2"] / total_reg_area if total_reg_area else 0.0,
        })
    write_csv(out / "V4B_REGRESSION_REASON_SUMMARY.csv", summary_rows, list(summary_rows[0]) if summary_rows else ["regression_reason"])

    # Strict-ledger reconciliation: use the exact strict-core predicate and
    # retain the missing keys instead of hiding them behind source priority.
    strict = strict_edges(formal / "04_identity_graph" / "identity_edges.parquet")
    ledger = pd.read_csv(cand / "FINAL_IDENTITY_SOURCE_LEDGER.csv", encoding="utf-8-sig", low_memory=False)
    ledger["candidate_source"] = ledger["candidate_source"].fillna("").astype(str)
    ledger_keys = [edge_key(r) for r in ledger.to_dict("records")]
    strict_ledger_keys = {edge_key(r) for r in ledger.to_dict("records") if "STRICT_CORE" in s(r.get("candidate_source"))}
    all_ledger_keys = set(ledger_keys)
    strict_relabelled = {k for k in strict if k in all_ledger_keys and k not in strict_ledger_keys}
    missing = sorted(strict - all_ledger_keys)
    # The unique-key set is used for traceability, but preserve the exact
    # input row count separately because formal Parquet can contain duplicate
    # metric rows for one logical edge.
    formal_strict_row_count = 0
    formal_held_row_count = 0
    formal_edges_for_counts = pq.ParquetFile(formal / "04_identity_graph" / "identity_edges.parquet")
    count_columns = ["from_scene", "to_scene", "source_object_id", "target_object_id", "score", "buffered_iou", "area_ratio", "local_neighborhood_consistency", "alignment_aware_displacement_m", "candidate_relation", "split_merge_type", "identity_conflict", "accepted_for_identity", "score_gap_source", "score_gap_target", "mutual_best"]
    strict_row_keys: list[tuple[str, str, str, str]] = []
    for batch in formal_edges_for_counts.iter_batches(batch_size=250_000, columns=count_columns):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            row = {k: data[k][i] for k in count_columns}
            if not b(row["accepted_for_identity"]) or b(row["identity_conflict"]) or s(row["candidate_relation"]) != "1:1" or s(row["split_merge_type"]) not in {"", "NONE"}:
                continue
            is_strict = f(row["score"]) >= 0.60 and f(row["buffered_iou"]) >= 0.55 and f(row["area_ratio"]) >= 0.55 and f(row["local_neighborhood_consistency"]) >= 0.60 and f(row["alignment_aware_displacement_m"]) <= 50
            if is_strict:
                formal_strict_row_count += 1
                strict_row_keys.append(edge_key(row))
            else:
                formal_held_row_count += 1
    recon = [{
        "input_strict_edges": formal_strict_row_count,
        "input_strict_unique_keys": len(strict),
        "ledger_strict_rows": int(sum("STRICT_CORE" in s(x) for x in ledger["candidate_source"])),
        "ledger_strict_deduplicated_rows": len(strict_ledger_keys),
        "relabelled_rows": len(strict_relabelled),
        "missing_rows": len(missing),
        "missing_explained": "NO" if missing else "YES",
    }]
    write_csv(out / "STRICT_CORE_LEDGER_RECONCILIATION.csv", recon, list(recon[0]))
    write_csv(out / "STRICT_CORE_LEDGER_MISSING.csv", [{"from_scene": a, "to_scene": b_, "source_object_id": c, "target_object_id": d} for a, b_, c, d in missing], ["from_scene", "to_scene", "source_object_id", "target_object_id"])

    # Audit all 2018->2020 records in the existing ledger.  This is diagnostic
    # only; the V2 runner will force transition_eligible=NO for these edges.
    direct = ledger[(ledger["source_epoch"].astype(str) == "xa180429") & (ledger["target_epoch"].astype(str) == "xa201231")].copy()
    direct_summary = direct.groupby(["candidate_source", "transition_eligible"], dropna=False).size().reset_index(name="count")
    write_csv(out / "DIRECT_2018_2020_SUPPORT_AUDIT.csv", direct_summary.to_dict("records"), ["candidate_source", "transition_eligible", "count"])

    # Current candidate held-recovery coverage, using the exact 91,152 formal
    # accepted-but-not-strict rows.
    held_rows: list[dict[str, Any]] = []
    formal_edges = pq.ParquetFile(formal / "04_identity_graph" / "identity_edges.parquet")
    columns = ["from_scene", "to_scene", "source_object_id", "target_object_id", "score", "buffered_iou", "area_ratio", "local_neighborhood_consistency", "alignment_aware_displacement_m", "candidate_relation", "split_merge_type", "identity_conflict", "accepted_for_identity", "score_gap_source", "score_gap_target", "mutual_best"]
    for batch in formal_edges.iter_batches(batch_size=250_000, columns=columns):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            row = {k: data[k][i] for k in columns}
            if not b(row["accepted_for_identity"]):
                continue
            if (
                not b(row["identity_conflict"])
                and s(row["candidate_relation"]) == "1:1"
                and s(row["split_merge_type"]) in {"", "NONE"}
                and f(row["score"]) >= 0.60
                and f(row["buffered_iou"]) >= 0.55
                and f(row["area_ratio"]) >= 0.55
                and f(row["local_neighborhood_consistency"]) >= 0.60
                and f(row["alignment_aware_displacement_m"]) <= 50
            ):
                continue
            held_rows.append(row)
    write_csv(out / "HELD_EDGE_BASELINE.csv", held_rows, columns)

    receipt = {
        "resolved_v4b_components": len(resolved),
        "v4b_regression_component_count": sum(1 for r in component_rows if r["regression_intervals"]),
        "v4b_regression_object_interval_rows": len(diff_rows),
        "component_promotion_bug_count": sum(r["component_promotion_bug"] == "YES" for r in component_rows),
        "strict_core_edges": len(strict),
        "strict_ledger_rows": int(recon[0]["ledger_strict_rows"]),
        "strict_ledger_missing_rows": len(missing),
        "held_edges": formal_held_row_count,
        "direct_2018_2020_ledger_rows": len(direct),
        "formal_outputs_modified": False,
    }
    (out / "REGRESSION_AUDIT_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
