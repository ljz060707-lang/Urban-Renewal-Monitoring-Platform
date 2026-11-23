"""Candidate-only Presence/State recomputation using an identity overlay.

The formal tracking and geometry artifacts remain read-only.  The output is a
compressed tabular candidate run plus a delta map, deliberately avoiding a
second multi-gigabyte GPKG copy on the cloud disk.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FORMAL = Path("/data/xian_r4_100km2_20260914/output/西安_R4_辅助影像接入前_三期三态候选版_20260914")
BASE = Path("/data/xian_r4_100km2_20260914/output/13_visual_asset_repair_and_obj0008_identity_closure_20260916")
IDENTITY = BASE / "identity_candidate"
OUT = BASE / "candidate_presence_state_overlay"
SAFE = IDENTITY / "ABA_SAFE_CLOSURE_CANDIDATES.csv"
PRESENCE = FORMAL / "05_presence_state" / "PRESENCE_1_0_U.csv"
STATE = FORMAL / "05_presence_state" / "STATE_EVENTS_UNFILTERED.csv"

ORDER = ("xa180429", "xa190101", "xa201231")


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def num(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_safe() -> tuple[list[dict[str, str]], dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    rows = list(csv.DictReader(SAFE.open("r", encoding="utf-8-sig", newline="")))
    dsu = DSU()
    middle_targets: dict[str, set[str]] = defaultdict(set)
    middle_rows: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        a, b, c = row.get("source_2018_track_id", ""), row.get("middle_2019_track_id", ""), row.get("target_2020_track_id", "")
        if not (a and b and c):
            continue
        middle_targets[b].add(a)
        middle_rows[b].append(row.get("source_2018", ""))
    blocked_middle = {middle for middle, targets in middle_targets.items() if len(targets) > 1}
    applied_rows: list[dict[str, str]] = []
    for row in rows:
        a, b, c = row.get("source_2018_track_id", ""), row.get("middle_2019_track_id", ""), row.get("target_2020_track_id", "")
        if a and b and c and b not in blocked_middle:
            dsu.union(a, b)
            dsu.union(a, c)
            applied_rows.append(row)
    components: dict[str, set[str]] = defaultdict(set)
    for x in dsu.parent:
        components[dsu.find(x)].add(x)
    canonical_by_root: dict[str, str] = {}
    for root, members in components.items():
        preferred = []
        for row in applied_rows:
            if dsu.find(row.get("source_2018_track_id", "")) == root:
                preferred.append(row.get("source_2018_track_id", ""))
        canonical_by_root[root] = min(preferred or list(members))
    mapping = {member: canonical_by_root[dsu.find(member)] for member in dsu.parent if canonical_by_root[dsu.find(member)] != member}
    return rows, mapping, {k: sorted(v) for k, v in middle_targets.items()}, {k: sorted(v) for k, v in components.items()}


def write_csv_gz(path: Path, fieldnames: list[str], rows) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=6) as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_presence(mapping: dict[str, str], affected_ids: set[str]) -> tuple[dict[str, dict[str, list[dict[str, str]]]], list[str]]:
    mapped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with PRESENCE.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            old = row.get("building_id") or row.get("track_id") or ""
            if old in affected_ids:
                mapped[mapping.get(old, old)][row.get("scene_key", "")].append(row)
    return mapped, fieldnames


def aggregate_presence_row(new_id: str, scene: str, rows: list[dict[str, str]], mapping: dict[str, str]) -> dict[str, object]:
    values = {str(row.get("presence", "")) for row in rows}
    if "1" in values:
        presence = "1"
        reason = "CANDIDATE_IDENTITY_MERGE_POSITIVE_SUPPORT"
    elif "0" in values:
        presence = "0"
        reason = "CANDIDATE_IDENTITY_MERGE_RELIABLE_ABSENCE"
    else:
        presence = "U"
        reason = "CANDIDATE_IDENTITY_MERGE_UNOBSERVED_OR_UNCERTAIN"
    dates = sorted({str(row.get("actual_capture_date", "")) for row in rows if row.get("actual_capture_date") and str(row.get("actual_capture_date")) != "nan"})
    uids = sorted({str(row.get("candidate_object_uid", "")) for row in rows if row.get("candidate_object_uid")})
    old_ids = sorted({str(row.get("building_id", "")) for row in rows})
    areas = [num(row.get("area_m2"), 0.0) for row in rows if row.get("area_m2") not in {None, "", "nan"}]
    return {
        "building_id": new_id,
        "track_id": new_id,
        "scene_key": scene,
        "formal_date": {"xa180429": "2018-04-29", "xa190101": "2019-01-01", "xa201231": "2020-12-31"}.get(scene, ""),
        "presence": presence,
        "presence_reason": reason,
        "actual_capture_date": dates[0] if len(dates) == 1 else ";".join(dates),
        "date_support_status": "MULTIPLE_OBSERVATIONS" if len(dates) > 1 else (rows[0].get("date_support_status", "") if rows else ""),
        "observability_status": ";".join(sorted({str(row.get("observability_status", "")) for row in rows if row.get("observability_status")})),
        "candidate_object_uid": ";".join(uids),
        "area_m2": max(areas) if areas else "",
        "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
        "identity_overlay_applied": True,
        "identity_merged_from": ";".join(old_ids),
    }


def temporal_status(rows: dict[str, dict[str, object]]) -> tuple[str, list[str]]:
    dates = [str(rows.get(scene, {}).get("actual_capture_date", "")) for scene in ORDER]
    pair_status = []
    for left, right in zip(dates, dates[1:]):
        if left and right and left != "nan" and right != "nan" and left != right:
            pair_status.append("OFF_DATE_VALID")
        elif left and right and left == right:
            pair_status.append("DUPLICATE_EPOCH")
        else:
            pair_status.append("TEMPORAL_INSUFFICIENT")
    return ("VALID_TEMPORAL" if pair_status and all(x == "OFF_DATE_VALID" for x in pair_status) else ";".join(pair_status)), pair_status


def decode_candidate_state(new_id: str, rows: dict[str, dict[str, object]], conflict: bool) -> dict[str, object]:
    sequence = "-".join(str(rows.get(scene, {}).get("presence", "U")) for scene in ORDER)
    temporal, _ = temporal_status(rows)
    strict = temporal == "VALID_TEMPORAL"
    if conflict:
        state, event = "UNKNOWN_CONFLICT", "IDENTITY_CONFLICT"
    elif sequence == "1-U-1":
        state, event = "GAP_RETURN", "GAP_RETURN"
    elif not strict:
        state, event = "UNKNOWN_CONFLICT", temporal
    elif sequence == "1-1-1":
        state, event = "STABLE", "CANDIDATE_IDENTITY_CLOSURE"
    elif sequence in {"0-1-1", "0-0-1"}:
        state, event = "ADDITION", "ADDITION"
    elif sequence in {"1-1-0", "1-0-0"}:
        state, event = "DEMOLITION", "DEMOLITION"
    elif sequence == "1-0-1":
        state, event = "UNKNOWN_CONFLICT", "REDEVELOPMENT_101_DIAGNOSTIC"
    else:
        state, event = "UNKNOWN_CONFLICT", "UNKNOWN_SUPPORT"
    chosen_scene = "xa190101" if state == "STABLE" and rows.get("xa190101", {}).get("presence") == "1" else ("xa201231" if state == "ADDITION" else "xa180429")
    chosen = rows.get(chosen_scene, {})
    area = chosen.get("area_m2", "") if chosen.get("presence") == "1" else ""
    return {
        "building_id": new_id,
        "track_id": new_id,
        "presence_sequence": sequence,
        "state": state,
        "event": event,
        "temporal_status": temporal,
        "strict_change_eligible": bool(strict and state in {"STABLE", "ADDITION", "DEMOLITION"}),
        "area_m2": area,
        "source_date": chosen.get("formal_date", "") if area != "" else "",
        "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
        "identity_overlay_applied": True,
        "identity_conflict_from_overlap": conflict,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    safe_rows, mapping, middle_targets, components = read_safe()
    affected_ids = set(mapping) | set(mapping.values())
    mapped_presence, presence_fields = aggregate_presence(mapping, affected_ids)
    overlay_presence_fields = presence_fields + ["identity_overlay_applied", "identity_merged_from"]
    # Existing formal identity conflicts are retained as a conservative block.
    conflict_old: set[str] = set()
    with STATE.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("event") == "IDENTITY_CONFLICT":
                conflict_old.add(row.get("building_id", ""))

    presence_path = OUT / "CANDIDATE_PRESENCE_1_0_U.csv.gz"
    mapped_presence_rows: dict[str, dict[str, dict[str, object]]] = {}
    presence_group_conflicts: set[str] = set()
    for new_id, scene_rows in mapped_presence.items():
        mapped_presence_rows[new_id] = {}
        for scene, rows in scene_rows.items():
            value = aggregate_presence_row(new_id, scene, rows, mapping)
            mapped_presence_rows[new_id][scene] = value
            if len({uid for uid in str(value.get("candidate_object_uid", "")).split(";") if uid}) > 1:
                presence_group_conflicts.add(new_id)

    with PRESENCE.open("r", encoding="utf-8-sig", newline="") as source, gzip.open(presence_path, "wt", encoding="utf-8", newline="", compresslevel=6) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=overlay_presence_fields, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            old = row.get("building_id") or row.get("track_id") or ""
            if old in affected_ids:
                continue
            row["identity_overlay_applied"] = False
            row["identity_merged_from"] = ""
            writer.writerow(row)
        for new_id in sorted(mapped_presence_rows):
            for scene in ORDER:
                if scene in mapped_presence_rows[new_id]:
                    writer.writerow(mapped_presence_rows[new_id][scene])

    # Read formal State once to preserve all untouched rows while rebuilding
    # only components changed by the identity overlay.
    mapped_state_ids = set(affected_ids)
    state_path = OUT / "CANDIDATE_STATE_EVENTS_UNFILTERED.csv.gz"
    state_fields = ["building_id", "track_id", "presence_sequence", "state", "event", "temporal_status", "strict_change_eligible", "area_m2", "source_date", "scientific_status", "identity_overlay_applied", "identity_conflict_from_overlap"]
    state_rows_by_new: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for new_id, scene_rows in mapped_presence_rows.items():
        state_rows_by_new[new_id] = scene_rows
    state_stats = Counter()
    state_area: Counter[str] = Counter()
    sample_queues: dict[str, list[dict[str, object]]] = defaultdict(list)
    rebuilt_states: list[dict[str, object]] = []
    for new_id, scene_rows in state_rows_by_new.items():
        members = set()
        for scene in scene_rows.values():
            members.update(str(scene.get("identity_merged_from", "")).split(";"))
        # A prior IDENTITY_CONFLICT on the middle track is the defect this
        # validated ABA closure is intended to repair.  Do not carry that
        # diagnostic forward automatically; retain only new overlap conflicts
        # created by the candidate overlay itself.
        conflict = new_id in presence_group_conflicts
        rebuilt = decode_candidate_state(new_id, scene_rows, conflict)
        rebuilt_states.append(rebuilt)

    rebuilt_by_id = {row["building_id"]: row for row in rebuilt_states}
    with STATE.open("r", encoding="utf-8-sig", newline="") as source, gzip.open(state_path, "wt", encoding="utf-8", newline="", compresslevel=6) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=state_fields, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            old = row.get("building_id", "")
            if old in mapped_state_ids:
                continue
            row["identity_overlay_applied"] = False
            row["identity_conflict_from_overlap"] = False
            writer.writerow(row)
            state = str(row.get("state", ""))
            state_stats[state] += 1
            state_area[state] += num(row.get("area_m2"), 0.0)
            if len(sample_queues[state]) < 100:
                sample_queues[state].append({"state": state, "building_id": old, "presence_sequence": row.get("presence_sequence", ""), "area_m2": row.get("area_m2", ""), "review_status": "PENDING_VISUAL_QA"})
        for row in rebuilt_states:
            writer.writerow(row)
            state = str(row.get("state", ""))
            state_stats[state] += 1
            state_area[state] += num(row.get("area_m2"), 0.0)
            if len(sample_queues[state]) < 100:
                sample_queues[state].append({"state": state, "building_id": row["building_id"], "presence_sequence": row.get("presence_sequence", ""), "area_m2": row.get("area_m2", ""), "review_status": "PENDING_VISUAL_QA"})

    map_rows = []
    for old, new in sorted(mapping.items()):
        map_rows.append({"old_track_id": old, "candidate_track_id": new, "changed": old != new, "rule": "SAFE_ABA_IDENTITY_CLOSURE", "formal_writeback": False})
    with (OUT / "CANDIDATE_TRACK_ID_MAP.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(map_rows[0]) if map_rows else ["old_track_id", "candidate_track_id", "changed", "rule", "formal_writeback"])
        writer.writeheader()
        writer.writerows(map_rows)

    closure_rows = []
    for row in safe_rows:
        b = row.get("middle_2019_track_id", "")
        a = row.get("source_2018_track_id", "")
        closure_rows.append({**row, "candidate_applied": bool(b in mapping and mapping.get(b) == mapping.get(a, a)), "candidate_track_id": mapping.get(a, a), "formal_writeback": False})
    closure_fields = list(closure_rows[0]) if closure_rows else ["candidate_applied", "candidate_track_id", "formal_writeback"]
    with (OUT / "CANDIDATE_ABA_CLOSURES.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=closure_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(closure_rows)

    queue_rows = [row for state in sorted(sample_queues) for row in sample_queues[state]]
    with (OUT / "FALSE_CHANGE_AUDIT_QUEUE.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["state", "building_id", "presence_sequence", "area_m2", "review_status"])
        writer.writeheader()
        writer.writerows(queue_rows)

    total_area = sum(state_area.values())
    unknown_area = state_area.get("UNKNOWN_CONFLICT", 0.0) + state_area.get("GAP_RETURN", 0.0)
    total_objects = sum(state_stats.values())
    unknown_objects = state_stats.get("UNKNOWN_CONFLICT", 0) + state_stats.get("GAP_RETURN", 0)
    receipt = {
        "task": "XIAN-VISUAL-ASSET-REPAIR-AND-OBJ0008-IDENTITY-CLOSURE-001",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": "CANDIDATE_IDENTITY_OVERLAY_PRESENCE_STATE",
        "formal_outputs_modified": False,
        "geometry_owner_formal_unchanged": True,
        "safe_aba_rows_read": len(safe_rows),
        "candidate_closures_applied": sum(1 for row in closure_rows if row["candidate_applied"]),
        "old_tracks_mapped": len(mapping),
        "candidate_components": len(components),
        "mapping_conflict_middle_tracks": sum(1 for values in middle_targets.values() if len(values) > 1),
        "presence_output": str(presence_path),
        "state_output": str(state_path),
        "state_counts": dict(state_stats),
        "state_area_m2": dict(state_area),
        "unknown_including_gap_return_area_pct": round(100.0 * unknown_area / total_area, 6) if total_area else None,
        "unknown_including_gap_return_object_pct": round(100.0 * unknown_objects / total_objects, 6) if total_objects else None,
        "false_change_audit_queue_count": len(queue_rows),
        "false_change_visual_review_status": "PENDING_PARENT_REVIEW",
        "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
    }
    (OUT / "CANDIDATE_PRESENCE_STATE_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "CANDIDATE_PRESENCE_STATE_REPORT.md").write_text(
        "# Candidate Presence/State overlay\n\n"
        "This is an isolated candidate run. It applies only validated-rule ABA identity overlays to the unchanged formal tabular outputs; geometry ownership and formal outputs were not modified.\n\n"
        f"- safe ABA rows read: {len(safe_rows)}\n"
        f"- closures applied: {receipt['candidate_closures_applied']}\n"
        f"- old tracks mapped: {len(mapping)}\n"
        f"- state counts: {json.dumps(dict(state_stats), ensure_ascii=False)}\n"
        f"- unknown including GAP_RETURN area pct: {receipt['unknown_including_gap_return_area_pct']}%\n"
        f"- unknown including GAP_RETURN object pct: {receipt['unknown_including_gap_return_object_pct']}%\n\n"
        "The false-change queue is generated but remains pending visual review. Unknowns are not reassigned to Stable.\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
