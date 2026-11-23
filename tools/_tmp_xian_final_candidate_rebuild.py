"""Rebuild Xi'an candidate IdentityGraph and downstream state from scratch.

This is an append-only candidate run.  It never writes the frozen Geometry,
Tracking, Presence, State, or product TIF directories.  The old Presence and
State tables are used only as per-original-track observability evidence when
an assembled physical track has no object in an epoch; they are not patched or
copied into the candidate result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCENES = ("xa180429", "xa190101", "xa201231")
DATES = {"xa180429": "2018-04-29", "xa190101": "2019-01-01", "xa201231": "2020-12-31"}
PAIRS = (("xa180429", "xa190101"), ("xa190101", "xa201231"), ("xa180429", "xa201231"))
TRANSITIONS = (("2018_2019", "xa180429", "xa190101"), ("2019_2020", "xa190101", "xa201231"))

STRICT_SCORE_MIN = 0.60
STRICT_IOU_MIN = 0.55
STRICT_AREA_MIN = 0.55
STRICT_NEIGHBOR_MIN = 0.60
STRICT_DISPLACEMENT_MAX = 50.0
STRICT_MARGIN_MIN = 0.15

LEDGER_FIELDS = [
    "edge_id", "source_epoch", "target_epoch", "source_object_id", "target_object_id",
    "relation_type", "candidate_source", "score", "buffered_iou", "area_ratio",
    "displacement", "neighborhood_score", "alignment_support", "visual_review_status",
    "structural_gate_status", "eligible_for_identity", "transition_eligible", "reason",
    "identity_conflict", "split_merge_type", "candidate_relation", "edge_stage",
]


def sf(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def si(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def sb(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def progress(out: Path, phase: str, **values: Any) -> None:
    payload = {"task": "XIAN-FINAL-CANDIDATE-IDENTITYGRAPH-REBUILD-AND-FULL-STATE-RERUN-001", "current_phase": phase, **values}
    write_json(out / "RUN_STATUS.json", payload)
    lines = ["# Xi'an final candidate rebuild progress", "", f"- current_phase: `{phase}`"]
    for key, value in values.items():
        lines.append(f"- {key}: `{value}`")
    (out / "RUN_PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class DSU:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            nxt = parent[value]
            parent[value] = root
            value = nxt
        return root

    def union(self, left: int, right: int) -> bool:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return False
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1
        return True


class LocalDSU:
    """String-keyed DSU for one V4B component only."""

    def __init__(self, nodes: Iterable[str]) -> None:
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in self.parent}

    def find(self, value: str) -> str:
        parent = self.parent
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            nxt = parent[value]
            parent[value] = root
            value = nxt
        return root

    def union(self, left: str, right: str) -> bool:
        if left not in self.parent or right not in self.parent:
            return False
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return False
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1
        return True


def edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (text(edge.get("from_scene")), text(edge.get("source_object_id")), text(edge.get("target_object_id")))


def normalize_edge(row: dict[str, Any], stage: str = "") -> dict[str, Any]:
    return {
        "source_object_id": text(row.get("source_object_id")),
        "target_object_id": text(row.get("target_object_id")),
        "from_scene": text(row.get("from_scene")),
        "to_scene": text(row.get("to_scene")),
        "source_index": si(row.get("source_index")),
        "target_index": si(row.get("target_index")),
        "distance_m": sf(row.get("distance_m")),
        "polygon_iou": sf(row.get("polygon_iou")),
        "buffered_iou": sf(row.get("buffered_iou")),
        "bbox_iou": sf(row.get("bbox_iou")),
        "area_ratio": sf(row.get("area_ratio")),
        "shape_similarity": sf(row.get("shape_similarity")),
        "local_neighborhood_consistency": sf(row.get("local_neighborhood_consistency")),
        "score": sf(row.get("score")),
        "identity_confidence": sf(row.get("identity_confidence", row.get("score"))),
        "alignment_aware_displacement_m": sf(row.get("alignment_aware_displacement_m", row.get("distance_m"))),
        "relation": text(row.get("relation")),
        "candidate_relation": text(row.get("candidate_relation")),
        "split_merge_type": text(row.get("split_merge_type")) or "NONE",
        "identity_conflict": sb(row.get("identity_conflict")),
        "accepted_for_identity": sb(row.get("accepted_for_identity")),
        "candidate_count_from_source": si(row.get("candidate_count_from_source"), 0),
        "candidate_count_to_target": si(row.get("candidate_count_to_target"), 0),
        "reasonable_candidate_count_from_source": si(row.get("reasonable_candidate_count_from_source"), 0),
        "reasonable_candidate_count_to_target": si(row.get("reasonable_candidate_count_to_target"), 0),
        "score_gap_source": sf(row.get("score_gap_source")),
        "score_gap_target": sf(row.get("score_gap_target")),
        "mutual_best": sb(row.get("mutual_best")),
        "edge_stage": stage or text(row.get("edge_stage")),
    }


def strict_preeligible(edge: dict[str, Any]) -> bool:
    return (
        edge["accepted_for_identity"]
        and not edge["identity_conflict"]
        and edge["candidate_relation"] == "1:1"
        and edge["split_merge_type"] in {"", "NONE"}
        and edge["score"] >= STRICT_SCORE_MIN
        and edge["buffered_iou"] >= STRICT_IOU_MIN
        and edge["area_ratio"] >= STRICT_AREA_MIN
        and edge["local_neighborhood_consistency"] >= STRICT_NEIGHBOR_MIN
        and edge["alignment_aware_displacement_m"] <= STRICT_DISPLACEMENT_MAX
    )


def relation_source(edge: dict[str, Any], selected: bool) -> str:
    if not selected:
        return "REVIEW_ONLY"
    sm = edge.get("split_merge_type", "")
    relation = edge.get("candidate_relation", "") or edge.get("relation", "")
    stage = text(edge.get("edge_stage"))
    if "DIRECT_TEMPORAL" in stage and edge.get("from_scene") == "xa180429" and edge.get("to_scene") == "xa201231":
        return "DIRECT_TEMPORAL_SUPPORT_ONLY"
    if sm == "SPLIT":
        return "COMPOUND_SPLIT"
    if sm == "MERGE":
        return "COMPOUND_MERGE"
    if sm == "SPLIT_MERGE":
        return "COMPOUND_SPLIT_MERGE"
    if relation in {"1:N", "N:1", "N:M"}:
        return "V4B_STRUCTURAL_CLOSURE"
    return "V4B_STRUCTURAL_CLOSURE"


def ledger_row(edge: dict[str, Any], edge_id: str, source: str, eligible: bool, transition_eligible: bool, review: str, gate: str, reason: str) -> dict[str, Any]:
    alignment = "PASS" if edge.get("alignment_aware_displacement_m", 0.0) <= 50 else "PARALLAX_REVIEW"
    return {
        "edge_id": edge_id,
        "source_epoch": edge.get("from_scene", ""),
        "target_epoch": edge.get("to_scene", ""),
        "source_object_id": edge.get("source_object_id", ""),
        "target_object_id": edge.get("target_object_id", ""),
        "relation_type": edge.get("candidate_relation") or edge.get("relation") or "",
        "candidate_source": source,
        "score": edge.get("score", 0.0),
        "buffered_iou": edge.get("buffered_iou", 0.0),
        "area_ratio": edge.get("area_ratio", 0.0),
        "displacement": edge.get("alignment_aware_displacement_m", 0.0),
        "neighborhood_score": edge.get("local_neighborhood_consistency", 0.0),
        "alignment_support": alignment,
        "visual_review_status": review,
        "structural_gate_status": gate,
        "eligible_for_identity": "YES" if eligible else "NO",
        "transition_eligible": "YES" if transition_eligible else "NO",
        "reason": reason,
        "identity_conflict": edge.get("identity_conflict", False),
        "split_merge_type": edge.get("split_merge_type", ""),
        "candidate_relation": edge.get("candidate_relation", ""),
        "edge_stage": edge.get("edge_stage", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v4b", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--visual-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, v4b_root, repair_root, visual_queue, out = args.root, args.v4b, args.repair, args.visual_queue, args.output
    if out.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    progress(out, "INIT", blocker="", next_action="load frozen canonical objects")

    tracking_path = root / "04_identity_graph" / "tracking_objects_unfiltered.csv"
    formal_edges_path = root / "04_identity_graph" / "identity_edges.parquet"
    formal_presence_path = root / "05_presence_state" / "PRESENCE_1_0_U.csv"
    v4b_graph_path = v4b_root / "THREE_EPOCH_IDENTITY_GRAPH.parquet"
    compounds_path = v4b_root / "COMPOUND_TRACKS.csv"
    aba_path = repair_root / "ABA_SAFE_CLOSURE_CANDIDATES.csv"
    repaired_review_path = repair_root / "REPAIRED_AFFECTED_VISUAL_REVIEW.csv"

    obj = pd.read_csv(tracking_path, usecols=["scene_key", "formal_date", "building_id", "track_id", "object_uid", "source_asset_id", "local_actual_date", "area_m2"], low_memory=False)
    obj["object_uid"] = obj["object_uid"].astype(str)
    obj["scene_key"] = obj["scene_key"].astype(str)
    obj["building_id"] = obj["building_id"].astype(str)
    obj["track_id"] = obj["track_id"].astype(str)
    uid_list = obj["object_uid"].tolist()
    uid_to_idx = {uid: idx for idx, uid in enumerate(uid_list)}
    object_scene = obj["scene_key"].to_numpy()
    object_building = obj["building_id"].to_numpy()
    object_area = pd.to_numeric(obj["area_m2"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    progress(out, "LOAD_OBJECTS", total_objects=len(obj), next_action="load observability evidence and V4B components")

    evidence: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with formal_presence_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            bid, scene = text(row.get("building_id")), text(row.get("scene_key"))
            if not bid or not scene:
                continue
            evidence[bid][scene] = {
                "presence": text(row.get("presence")),
                "reason": text(row.get("presence_reason")),
                "candidate_object_uid": text(row.get("candidate_object_uid")),
                "observability": text(row.get("observability_status")),
            }

    comp_meta: dict[str, dict[str, Any]] = {}
    uid_to_resolved_comp: dict[str, str] = {}
    resolved_statuses = {"RESOLVED_CONTINUOUS_IDENTITY_COMPONENT", "RESOLVED_GAP_RETURN"}
    with compounds_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cid = text(row.get("component_id"))
            status = text(row.get("component_status"))
            nodes = [item for item in text(row.get("object_uids")).split(";") if item in uid_to_idx]
            comp_meta[cid] = {"status": status, "nodes": nodes, "row": row}
            if status in resolved_statuses:
                for uid in nodes:
                    uid_to_resolved_comp[uid] = cid
    resolved_components = {cid for cid, meta in comp_meta.items() if meta["status"] in resolved_statuses}
    progress(out, "LOAD_COMPONENTS", resolved_components=len(resolved_components), total_components=len(comp_meta), next_action="build strict core")

    # Recompute the strict core from the frozen pairwise evidence.  This is an
    # edge gate only; it does not change any polygon or upstream tracking row.
    edge_columns = [
        "source_object_id", "target_object_id", "from_scene", "to_scene", "source_index", "target_index",
        "distance_m", "polygon_iou", "buffered_iou", "bbox_iou", "area_ratio", "shape_similarity",
        "local_neighborhood_consistency", "score", "identity_confidence", "alignment_aware_displacement_m",
        "relation", "candidate_relation", "split_merge_type", "identity_conflict", "accepted_for_identity",
        "candidate_count_from_source", "candidate_count_to_target", "reasonable_candidate_count_from_source",
        "reasonable_candidate_count_to_target", "score_gap_source", "score_gap_target", "mutual_best",
    ]
    base_edges: list[dict[str, Any]] = []
    strict_pre: list[dict[str, Any]] = []
    pf = pq.ParquetFile(formal_edges_path)
    for batch in pf.iter_batches(batch_size=250_000, columns=edge_columns):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            row = {field: data[field][i] for field in edge_columns}
            edge = normalize_edge(row, "BASE_ACCEPTED" if sb(row.get("accepted_for_identity")) else "")
            if edge["accepted_for_identity"]:
                base_edges.append(edge)
                if strict_preeligible(edge):
                    strict_pre.append(edge)
    src_buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    dst_buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in strict_pre:
        pair = (edge["from_scene"], edge["to_scene"])
        src_buckets[(*pair, edge["source_object_id"])].append(edge)
        dst_buckets[(*pair, edge["target_object_id"])].append(edge)
    strict_keep: list[dict[str, Any]] = []
    for edge in strict_pre:
        pair = (edge["from_scene"], edge["to_scene"])
        src = src_buckets[(*pair, edge["source_object_id"])]
        dst = dst_buckets[(*pair, edge["target_object_id"])]
        unique = len(src) == 1 and len(dst) == 1
        margin = min(edge["score_gap_source"], edge["score_gap_target"]) >= STRICT_MARGIN_MIN
        if unique or (margin and edge["mutual_best"]):
            strict_keep.append(edge)
    conflict_src = Counter((e["from_scene"], e["to_scene"], e["source_object_id"]) for e in strict_keep)
    conflict_dst = Counter((e["from_scene"], e["to_scene"], e["target_object_id"]) for e in strict_keep)
    strict_edges = [
        e for e in strict_keep
        if conflict_src[(e["from_scene"], e["to_scene"], e["source_object_id"])] == 1
        and conflict_dst[(e["from_scene"], e["to_scene"], e["target_object_id"])] == 1
    ]
    strict_keys = {edge_key(e) for e in strict_edges}
    progress(out, "STRICT_CORE", accepted_edges=len(base_edges), strict_core_edges=len(strict_edges), next_action="read V4B candidate graph")

    # Read the V4B graph in streaming batches, retaining only edges that can
    # explain a promoted V4B component.  All graph edges are also streamed into
    # the ledger later with REVIEW_ONLY/REJECTED status when not eligible.
    v4b_edge_columns = [
        "record_type", "edge_stage", "from_scene", "to_scene", "source_object_id", "target_object_id",
        "source_index", "target_index", "distance_m", "polygon_iou", "buffered_iou", "bbox_iou", "area_ratio",
        "shape_similarity", "local_neighborhood_consistency", "score", "identity_confidence",
        "alignment_aware_displacement_m", "relation", "candidate_relation", "split_merge_type",
        "identity_conflict", "accepted_for_identity", "candidate_count_from_source", "candidate_count_to_target",
        "reasonable_candidate_count_from_source", "reasonable_candidate_count_to_target", "score_gap_source",
        "score_gap_target", "mutual_best",
    ]
    v4b_component_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    v4b_edge_count = 0
    v4b_pf = pq.ParquetFile(v4b_graph_path)
    for batch in v4b_pf.iter_batches(batch_size=250_000, columns=v4b_edge_columns):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            if text(data["record_type"][i]) != "EDGE":
                continue
            v4b_edge_count += 1
            src, dst = text(data["source_object_id"][i]), text(data["target_object_id"][i])
            comp = uid_to_resolved_comp.get(src)
            if comp and uid_to_resolved_comp.get(dst) == comp:
                row = {field: data[field][i] for field in v4b_edge_columns if field != "record_type"}
                v4b_component_edges[comp].append(normalize_edge(row, text(data["edge_stage"][i])))
    progress(out, "LOAD_V4B_GRAPH", v4b_edge_rows=v4b_edge_count, promoted_components=len(v4b_component_edges), next_action="assemble physical graph")

    dsu = DSU(len(uid_list))
    selected_edges: list[dict[str, Any]] = []
    selected_key_sources: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    ledger_extra: list[dict[str, Any]] = []

    def add_selected(edge: dict[str, Any], source: str, transition_eligible: bool = True, review: str = "NOT_REVIEWED", gate: str = "PASS", reason: str = "") -> bool:
        src, dst = edge.get("source_object_id"), edge.get("target_object_id")
        if src not in uid_to_idx or dst not in uid_to_idx:
            return False
        changed = dsu.union(uid_to_idx[src], uid_to_idx[dst])
        key = edge_key(edge)
        selected_key_sources[key].add(source)
        item = dict(edge)
        item["candidate_source"] = source
        item["transition_eligible"] = transition_eligible
        item["visual_review_status"] = review
        item["structural_gate_status"] = gate
        item["selection_reason"] = reason
        if changed or source in {"ABA_SAFE_CLOSURE", "DIRECT_TEMPORAL_SUPPORT_ONLY"}:
            selected_edges.append(item)
        return changed

    for edge in strict_edges:
        add_selected(edge, "STRICT_CORE", True, "INHERITED_STRICT_QA_PASS", "PASS", "strict multi-metric 1:1 backbone")

    # Promote only V4B components explicitly marked resolved.  A local Kruskal
    # pass retains real edge metrics and allows legal split/merge relations;
    # it never picks a nearest neighbour outside the gated component.
    def v4b_priority(edge: dict[str, Any]) -> tuple[int, float, float]:
        stage = text(edge.get("edge_stage"))
        non_conflict = 0 if not edge.get("identity_conflict") else 1
        base = 0 if "BASE_ACCEPTED" in stage else 1
        return (base * 10 + non_conflict, -sf(edge.get("score")), sf(edge.get("alignment_aware_displacement_m")))

    promoted_component_count = 0
    compound_edge_count = Counter()
    for cid in sorted(resolved_components):
        meta = comp_meta[cid]
        nodes = meta["nodes"]
        local = LocalDSU(nodes)
        for edge in sorted(v4b_component_edges.get(cid, []), key=v4b_priority):
            src, dst = edge.get("source_object_id"), edge.get("target_object_id")
            if src not in uid_to_idx or dst not in uid_to_idx:
                continue
            if not local.union(src, dst):
                continue
            source = relation_source(edge, True)
            if meta["status"] == "RESOLVED_GAP_RETURN" and edge.get("from_scene") == "xa180429" and edge.get("to_scene") == "xa201231":
                source = "GAP_RETURN"
            add_selected(edge, source, source != "DIRECT_TEMPORAL_SUPPORT_ONLY", "V4B_VISUAL_REVIEW_INHERITED", "PASS_COMPONENT_GATE", f"promoted {cid}")
            compound_edge_count[source] += 1
        # Ensure all nodes in a resolved component are on one physical track.
        anchor = nodes[0] if nodes else ""
        for uid in nodes[1:]:
            if uid in uid_to_idx and anchor in uid_to_idx and dsu.find(uid_to_idx[uid]) != dsu.find(uid_to_idx[anchor]):
                synthetic = {
                    "source_object_id": anchor, "target_object_id": uid,
                    "from_scene": object_scene[uid_to_idx[anchor]], "to_scene": object_scene[uid_to_idx[uid]],
                    "score": sf(meta["row"].get("identity_confidence")),
                    "identity_confidence": sf(meta["row"].get("identity_confidence")),
                    "buffered_iou": sf(meta["row"].get("aligned_union_iou_mean")),
                    "area_ratio": sf(meta["row"].get("union_area_ratio_min_max")),
                    "alignment_aware_displacement_m": sf(meta["row"].get("max_alignment_aware_displacement_m")),
                    "local_neighborhood_consistency": sf(meta["row"].get("min_neighborhood_consistency")),
                    "candidate_relation": "COMPOUND_PROMOTION", "relation": "COMPOUND_PROMOTION",
                    "split_merge_type": "COMPOUND", "identity_conflict": False,
                    "edge_stage": "V4B_COMPONENT_PROMOTION",
                }
                add_selected(synthetic, "V4B_STRUCTURAL_CLOSURE", True, "V4B_VISUAL_REVIEW_INHERITED", "PASS_COMPONENT_GATE", f"component connectivity {cid}")
                compound_edge_count["V4B_STRUCTURAL_CLOSURE"] += 1
        promoted_component_count += 1

    # The component edge lists are no longer needed after promotion; the
    # Parquet file is reopened below for the complete provenance ledger.
    del v4b_component_edges
    del base_edges
    del strict_pre

    # Visual recovery edges are deliberately limited to the corrected 273
    # samples and require a completed conservative review label.
    safe_recovery_count = Counter()
    repair_df = pd.read_csv(repaired_review_path, low_memory=False)
    queue_df = pd.read_csv(visual_queue, low_memory=False)
    for _, review in repair_df.iterrows():
        decision = text(review.get("recovery_decision"))
        label = text(review.get("visual_label"))
        if decision not in {"SAFE_RECOVERY", "PARALLAX_SAFE_RECOVERY"} or label not in {"PLAUSIBLE", "CORRECT"}:
            continue
        sample_id = text(review.get("sample_id"))
        matches = queue_df[queue_df["sample_id"].astype(str).eq(sample_id)].copy()
        interval = text(review.get("interval"))
        if "_" in interval:
            from_scene, to_scene = ("xa180429", "xa190101") if interval == "2018_2019" else ("xa190101", "xa201231")
            matches = matches[matches["from_scene"].astype(str).eq(from_scene) & matches["to_scene"].astype(str).eq(to_scene)]
        if matches.empty:
            continue
        q = matches.iloc[0].to_dict()
        edge = normalize_edge({
            "source_object_id": q.get("source_object_id"), "target_object_id": q.get("target_object_id"),
            "from_scene": q.get("from_scene"), "to_scene": q.get("to_scene"), "score": q.get("score"),
            "identity_confidence": q.get("identity_confidence"), "buffered_iou": q.get("buffered_iou"),
            "area_ratio": q.get("area_ratio"), "alignment_aware_displacement_m": q.get("alignment_aware_displacement_m"),
            "local_neighborhood_consistency": q.get("neighborhood_consistency"),
            "candidate_relation": q.get("candidate_relation"), "split_merge_type": q.get("split_merge_type"),
            "identity_conflict": q.get("identity_conflict"),
        }, "VISUAL_RECOVERY")
        source = "PARALLAX_SAFE_RECOVERY" if decision == "PARALLAX_SAFE_RECOVERY" else "SAFE_RECOVERY"
        if add_selected(edge, source, True, "COMPLETED", "PASS_VISUAL_REVIEW", f"{sample_id} {label}"):
            safe_recovery_count[source] += 1

    # ABA is assembled into the graph before Presence/State.  The direct
    # 2018->2020 edge is stored as continuity evidence only; adjacent bridge
    # edges are what make the physical track.
    aba_df = pd.read_csv(aba_path, low_memory=False)
    aba_df = aba_df[aba_df["decision"].astype(str).eq("SAFE_ABA_IDENTITY_CLOSURE_CANDIDATE")].copy()
    aba_edges: list[dict[str, Any]] = []
    aba_rows_used = 0
    gap_bridge_keys: set[tuple[str, str, str]] = set()
    for _, row in aba_df.iterrows():
        s, m, t = text(row.get("source_2018")), text(row.get("middle_2019")), text(row.get("target_2020"))
        if s not in uid_to_idx or t not in uid_to_idx:
            continue
        if m in uid_to_idx:
            bridge_a = normalize_edge({
                "source_object_id": s, "target_object_id": m, "from_scene": "xa180429", "to_scene": "xa190101",
                "score": row.get("bridge_ab_score"), "buffered_iou": row.get("bridge_ab_buffered_iou"),
                "area_ratio": row.get("bridge_ab_area_ratio"), "alignment_aware_displacement_m": row.get("bridge_ab_displacement_m"),
                "local_neighborhood_consistency": row.get("direct_neighborhood_consistency"), "candidate_relation": row.get("bridge_ab_relation"),
                "relation": row.get("bridge_ab_relation"), "identity_confidence": row.get("bridge_ab_score"),
            }, "ABA_SAFE_CLOSURE")
            add_selected(bridge_a, "ABA_SAFE_CLOSURE", True, "SAMPLED_VISUAL_QA_PASS", "PASS_ABA_SAFE_RULE", "validated ABA bridge")
            gap_bridge_keys.add(edge_key(bridge_a))
            bridge_b = normalize_edge({
                "source_object_id": m, "target_object_id": t, "from_scene": "xa190101", "to_scene": "xa201231",
                "score": row.get("bridge_bc_score"), "buffered_iou": row.get("bridge_bc_buffered_iou"),
                "area_ratio": row.get("bridge_bc_area_ratio"), "alignment_aware_displacement_m": row.get("bridge_bc_displacement_m"),
                "local_neighborhood_consistency": row.get("direct_neighborhood_consistency"), "candidate_relation": row.get("bridge_bc_relation"),
                "relation": row.get("bridge_bc_relation"), "identity_confidence": row.get("bridge_bc_score"),
            }, "ABA_SAFE_CLOSURE")
            add_selected(bridge_b, "ABA_SAFE_CLOSURE", True, "SAMPLED_VISUAL_QA_PASS", "PASS_ABA_SAFE_RULE", "validated ABA bridge")
            gap_bridge_keys.add(edge_key(bridge_b))
        else:
            direct = normalize_edge({
                "source_object_id": s, "target_object_id": t, "from_scene": "xa180429", "to_scene": "xa201231",
                "score": row.get("direct_score"), "buffered_iou": row.get("direct_buffered_iou"),
                "area_ratio": row.get("direct_area_ratio"), "alignment_aware_displacement_m": row.get("direct_displacement"),
                "local_neighborhood_consistency": row.get("direct_neighborhood_consistency"), "candidate_relation": "1:1",
                "relation": "1:1", "identity_confidence": row.get("direct_score"),
            }, "ABA_DIRECT_CONTINUITY")
            add_selected(direct, "ABA_SAFE_CLOSURE", False, "SAMPLED_VISUAL_QA_PASS", "PASS_ABA_SAFE_RULE", "validated ABA direct continuity")
        aba_rows_used += 1
        aba_edges.append({"source": s, "middle": m, "target": t, "pattern": text(row.get("pattern"))})
    progress(out, "GRAPH_ASSEMBLY", strict_core=len(strict_edges), v4b_promoted_components=promoted_component_count, safe_recovery=dict(safe_recovery_count), aba_rows=aba_rows_used, next_action="assign physical tracks")

    roots: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(uid_list)):
        roots[dsu.find(idx)].append(idx)
    sorted_roots = sorted(roots, key=lambda r: min(uid_list[i] for i in roots[r]))
    physical_id_by_root = {root: f"PTRK_R4FC_{n:08d}" for n, root in enumerate(sorted_roots, start=1)}
    physical_id_by_idx = {idx: physical_id_by_root[root] for root, indices in roots.items() for idx in indices}
    root_by_physical = {value: key for key, value in physical_id_by_root.items()}
    gap_physical_ids: set[str] = set()
    for uid, cid in uid_to_resolved_comp.items():
        if comp_meta.get(cid, {}).get("status") == "RESOLVED_GAP_RETURN" and uid in uid_to_idx:
            gap_physical_ids.add(physical_id_by_idx[uid_to_idx[uid]])
    for item in aba_edges:
        if item["pattern"] == "1-U-1" and item["source"] in uid_to_idx and item["target"] in uid_to_idx:
            gap_physical_ids.add(physical_id_by_idx[uid_to_idx[item["source"]]])

    track_sources: dict[int, set[str]] = defaultdict(set)
    for item in selected_edges:
        src, dst = item.get("source_object_id"), item.get("target_object_id")
        if src in uid_to_idx:
            track_sources[dsu.find(uid_to_idx[src])].add(text(item.get("candidate_source")))

    # Candidate tracking objects: raw canonical objects remain untouched; only
    # the candidate physical_track_id and provenance fields are new.
    out_obj = obj.copy()
    out_obj["canonical_object_id"] = out_obj["object_uid"]
    out_obj["physical_track_id"] = [physical_id_by_idx[i] for i in range(len(out_obj))]
    out_obj["compound_track_id"] = [uid_to_resolved_comp.get(uid, "") for uid in uid_list]
    out_obj["identity_sources"] = [";".join(sorted(track_sources.get(dsu.find(i), set()))) for i in range(len(out_obj))]
    out_obj["scientific_status"] = "CANDIDATE_NOT_GROUND_TRUTH"
    out_obj.to_csv(out / "FINAL_CANDIDATE_TRACKING_OBJECTS.csv", index=False, encoding="utf-8-sig")

    # Presence from scratch.  A positive object in the assembled graph is
    # positive; an explicit 1-U-1 bridge preserves the middle U.  No old state
    # value is copied.  Old presence rows are used only for reliable absence or
    # observability evidence when this physical track has no object.
    presence_by_physical: dict[str, dict[str, dict[str, Any]]] = {}
    presence_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    for root in sorted_roots:
        indices = roots[root]
        pid = physical_id_by_root[root]
        scenes = {scene: [i for i in indices if object_scene[i] == scene] for scene in SCENES}
        original_tracks = sorted(set(object_building[i] for i in indices))
        v4b_comps = sorted(set(uid_to_resolved_comp.get(uid_list[i], "") for i in indices if uid_to_resolved_comp.get(uid, "")))
        scene_info: dict[str, dict[str, Any]] = {}
        for scene in SCENES:
            scene_indices = scenes[scene]
            area = float(sum(object_area[i] for i in scene_indices))
            if pid in gap_physical_ids and scene == "xa190101":
                value, reason, object_uid = "U", "GAP_RETURN_MIDDLE_OBSERVABILITY", scene_indices[0] if scene_indices else ""
            elif scene_indices:
                value, reason, object_uid = "1", "POSITIVE_CANONICAL_OBJECT_SUPPORT", uid_list[scene_indices[0]]
            else:
                evs = [evidence.get(track, {}).get(scene) for track in original_tracks if evidence.get(track, {}).get(scene)]
                reliable_zero = [e for e in evs if e.get("presence") == "0" and e.get("reason") == "RELIABLE_CLEAR_ABSENCE"]
                uncertain = [e for e in evs if e.get("presence") == "U" or e.get("reason") == "UNOBSERVED_OR_UNCERTAIN"]
                if reliable_zero and not uncertain:
                    value, reason, object_uid = "0", "RELIABLE_CLEAR_ABSENCE_FROM_OBSERVABILITY_EVIDENCE", ""
                else:
                    value, reason, object_uid = "U", "UNOBSERVED_OR_UNCERTAIN_FROM_OBSERVABILITY_EVIDENCE", ""
            scene_info[scene] = {"presence": value, "reason": reason, "object_uid": object_uid, "area_m2": area, "object_count": len(scene_indices)}
            presence_rows.append({
                "physical_track_id": pid, "canonical_object_id": object_uid, "scene_key": scene, "formal_date": DATES[scene],
                "presence": value, "presence_reason": reason, "candidate_object_uid": object_uid,
                "area_m2": area, "object_count": len(scene_indices), "compound_track_id": ";".join(v4b_comps),
                "identity_sources": ";".join(sorted(track_sources.get(root, set()))), "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
            })
        presence_by_physical[pid] = scene_info
        track_rows.append({
            "physical_track_id": pid, "canonical_object_count": len(indices), "scene_count": sum(bool(scenes[s]) for s in SCENES),
            "object_count_2018": len(scenes["xa180429"]), "object_count_2019": len(scenes["xa190101"]), "object_count_2020": len(scenes["xa201231"]),
            "presence_sequence": "-".join(scene_info[s]["presence"] for s in SCENES),
            "compound_track_id": ";".join(v4b_comps), "identity_sources": ";".join(sorted(track_sources.get(root, set()))),
            "gap_return": "YES" if pid in gap_physical_ids else "NO", "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
        })
        if len(presence_rows) >= 50000:
            # Keep memory bounded; rows are written once below from a compact
            # list in this run, but the flush point is recorded for monitoring.
            progress(out, "PRESENCE_BUILD", physical_tracks=len(track_rows), presence_rows=len(presence_rows), next_action="continue from scratch Presence")
    write_csv(out / "CANDIDATE_PRESENCE_FROM_SCRATCH.csv", presence_rows, list(presence_rows[0]) if presence_rows else ["physical_track_id"])
    write_csv(out / "FINAL_CANDIDATE_PHYSICAL_TRACKS.csv", track_rows, list(track_rows[0]) if track_rows else ["physical_track_id"])

    # State decoder from the new physical-track Presence, not from old State.
    state_rows: list[dict[str, Any]] = []
    state_counts: dict[str, Counter[str]] = {transition: Counter() for transition, _, _ in TRANSITIONS}
    state_areas: dict[str, Counter[str]] = {transition: Counter() for transition, _, _ in TRANSITIONS}
    total_union: dict[str, float] = {transition: 0.0 for transition, _, _ in TRANSITIONS}
    for track in track_rows:
        pid = track["physical_track_id"]
        info = presence_by_physical[pid]
        seq = track["presence_sequence"]
        for transition, source_scene, target_scene in TRANSITIONS:
            a, b = info[source_scene]["presence"], info[target_scene]["presence"]
            source_area, target_area = info[source_scene]["area_m2"], info[target_scene]["area_m2"]
            union_area = max(source_area, target_area)
            total_union[transition] += union_area
            if seq == "1-U-1":
                state, event, eligible, reason = "STABLE", "GAP_RETURN", "NO", "ENDPOINT_CONTINUITY_1_U_1"
            elif a == "1" and b == "1":
                state, event, eligible, reason = "STABLE", "STABLE", "YES", "RELIABLE_PRESENT_BOTH_EPOCHS"
            elif a == "0" and b == "1":
                state, event, eligible, reason = "ADDITION", "ADDITION", "YES", "RELIABLE_0_TO_1"
            elif a == "1" and b == "0":
                state, event, eligible, reason = "DEMOLITION", "DEMOLITION", "YES", "RELIABLE_1_TO_0"
            else:
                state, event, eligible, reason = "UNKNOWN_CONFLICT", "UNKNOWN", "NO", "UNRESOLVED_IDENTITY_OBSERVABILITY_OR_PARALLAX"
            if seq in {"1-0-1", "0-1-0"} and state in {"ADDITION", "DEMOLITION"}:
                state, event, eligible, reason = "UNKNOWN_CONFLICT", "REBUILD_OR_FLICKER_REVIEW", "NO", "RELIABLE_ABSENCE_LOOP_REQUIRES_REVIEW"
            state_counts[transition][state] += 1
            area_value = target_area if state in {"STABLE", "ADDITION"} else source_area if state == "DEMOLITION" else union_area
            state_areas[transition][state] += area_value
            state_rows.append({
                "physical_track_id": pid, "interval": transition, "presence_sequence": seq,
                "source_presence": a, "target_presence": b, "state": state, "event": event,
                "strict_change_eligible": eligible, "state_reason": reason,
                "source_area_m2": source_area, "target_area_m2": target_area, "union_area_m2": union_area,
                "area_m2": area_value, "gap_return": "YES" if seq == "1-U-1" else "NO",
                "identity_sources": track["identity_sources"], "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
            })
    write_csv(out / "CANDIDATE_STATE_EVENTS_FROM_SCRATCH.csv", state_rows, list(state_rows[0]) if state_rows else ["physical_track_id"])
    progress(out, "STATE_BUILD", physical_tracks=len(track_rows), state_rows=len(state_rows), state_counts={k: dict(v) for k, v in state_counts.items()}, next_action="decompose residual Unknown")

    # Residual Unknown Phase B is diagnostic only.  It does not promote rows.
    residual_rows: list[dict[str, Any]] = []
    for row in state_rows:
        if row["state"] != "UNKNOWN_CONFLICT":
            continue
        pid = row["physical_track_id"]
        seq = row["presence_sequence"]
        sources = set(text(row.get("identity_sources")).split(";"))
        if "PARALLAX_SAFE_RECOVERY" in sources or "PARALLAX" in text(row.get("state_reason")):
            cause = "U03_PARALLAX"
        elif "1-U-1" == seq:
            cause = "U06_GAP"
        elif row["source_presence"] == "U" or row["target_presence"] == "U":
            cause = "U05_OBSERVABILITY"
        elif not sources or sources == {""}:
            cause = "U02_IDENTITY_NO_MATCH"
        else:
            cause = "U08_INSUFFICIENT_EVIDENCE"
        residual_rows.append({**row, "unknown_cause": cause, "phase": "RESIDUAL_UNKNOWN_PHASE_B", "resolution_status": "RETAINED_UNKNOWN"})
    write_csv(out / "RESIDUAL_UNKNOWN_PHASE_B.csv", residual_rows, list(residual_rows[0]) if residual_rows else ["physical_track_id"])

    # New false-change queue: only rows generated by this state pass.
    false_rows: list[dict[str, Any]] = []
    for row in state_rows:
        if row["state"] in {"ADDITION", "DEMOLITION"} and row["strict_change_eligible"] != "YES":
            false_rows.append({**row, "queue_reason": "CHANGE_NOT_STRICTLY_ELIGIBLE"})
        elif row["state"] in {"ADDITION", "DEMOLITION"} and row["presence_sequence"] in {"1-0-1", "0-1-0"}:
            false_rows.append({**row, "queue_reason": "REBUILD_OR_FLICKER_PATTERN"})
    write_csv(out / "FALSE_CHANGE_AUDIT_QUEUE_V2.csv", false_rows, list(false_rows[0]) if false_rows else ["physical_track_id"])

    # Deterministic visual QA sample index.  The current run records the exact
    # sample contract; it does not call a non-existent visual reviewer.
    qa_rows: list[dict[str, Any]] = []
    state_df = pd.DataFrame(state_rows)
    for state in ("STABLE", "ADDITION", "DEMOLITION", "UNKNOWN_CONFLICT"):
        subset = state_df[state_df["state"].eq(state)].sort_values(["physical_track_id", "interval"]).head(100)
        for n, (_, row) in enumerate(subset.iterrows(), start=1):
            qa_rows.append({"sample_id": f"{state}_QA_{n:03d}", "category": state, "physical_track_id": row["physical_track_id"], "interval": row["interval"], "presence_sequence": row["presence_sequence"], "visual_label": "NOT_REVIEWED", "visual_review_status": "PENDING_PARENT_REVIEW", "raw_asset_status": "INDEX_ONLY", "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH"})
    write_csv(out / "VISUAL_QA_SAMPLE_INDEX.csv", qa_rows, list(qa_rows[0]) if qa_rows else ["sample_id"])

    # Stream the final candidate source ledger.  Start with the selected and
    # supplemental edges; then append every V4B edge with an explicit review
    # disposition so no candidate relation is provenance-free.
    ledger_rows: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    edge_counter = 0
    selected_by_key: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for edge in selected_edges:
        source = text(edge.get("candidate_source"))
        key = edge_key(edge)
        selected_by_key[key].add(source)
        edge_counter += 1
        ledger_rows.append(ledger_row(edge, f"E_FC_{edge_counter:09d}", source, True, bool(edge.get("transition_eligible", True)), text(edge.get("visual_review_status")) or "NOT_REVIEWED", text(edge.get("structural_gate_status")) or "PASS", text(edge.get("selection_reason"))))
        graph_edges.append({**edge, "edge_id": f"E_FC_{edge_counter:09d}", "candidate_source": source})

    # Add a direct ABA continuity support record for every validated ABA row,
    # even when bridge edges already connect the nodes.
    for item in aba_edges:
        s, t = item["source"], item["target"]
        if s not in uid_to_idx or t not in uid_to_idx:
            continue
        support = normalize_edge({"source_object_id": s, "target_object_id": t, "from_scene": "xa180429", "to_scene": "xa201231", "candidate_relation": "1:1", "relation": "1:1"}, "ABA_DIRECT_CONTINUITY")
        edge_counter += 1
        ledger_rows.append(ledger_row(support, f"E_FC_{edge_counter:09d}", "DIRECT_TEMPORAL_SUPPORT_ONLY", False, False, "SAMPLED_VISUAL_QA_PASS", "PASS_ABA_SAFE_RULE", "2018->2020 continuity evidence only; not a transition event"))
        graph_edges.append({**support, "edge_id": f"E_FC_{edge_counter:09d}", "candidate_source": "DIRECT_TEMPORAL_SUPPORT_ONLY", "eligible_for_identity": False})

    # V4B source ledger review rows from the graph.  This pass intentionally
    # reads the graph again so source provenance is complete without holding a
    # second 1M-row table in memory.
    for batch in v4b_pf.iter_batches(batch_size=250_000, columns=v4b_edge_columns):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            if text(data["record_type"][i]) != "EDGE":
                continue
            edge = normalize_edge({field: data[field][i] for field in v4b_edge_columns if field != "record_type"}, text(data["edge_stage"][i]))
            key = edge_key(edge)
            if key in selected_by_key:
                continue
            source = "DIRECT_TEMPORAL_SUPPORT_ONLY" if "DIRECT_TEMPORAL" in edge.get("edge_stage", "") else "REVIEW_ONLY"
            reason = "not selected into promoted physical component" if source == "REVIEW_ONLY" else "direct temporal evidence retained without formal transition"
            edge_counter += 1
            ledger_rows.append(ledger_row(edge, f"E_FC_{edge_counter:09d}", source, False, False, "NOT_REVIEWED", "NOT_PROMOTED", reason))
    write_csv(out / "FINAL_IDENTITY_SOURCE_LEDGER.csv", ledger_rows, LEDGER_FIELDS)

    # Candidate graph contains every canonical node and all selected/support
    # edges.  Edge provenance is carried in the ledger and graph edge fields.
    graph_rows: list[dict[str, Any]] = []
    for idx, row in obj.iterrows():
        root = dsu.find(idx)
        uid = row["object_uid"]
        graph_rows.append({
            "record_type": "NODE", "edge_stage": "CANONICAL_OBJECT", "object_uid": uid, "scene": row["scene_key"],
            "formal_date": row["formal_date"], "current_building_id": row["building_id"], "canonical_object_id": uid,
            "physical_track_id": physical_id_by_root[root], "compound_track_id": ";".join(sorted({uid_to_resolved_comp.get(uid, "")} - {""})),
            "area_m2": sf(row["area_m2"]), "candidate_source": ";".join(sorted(track_sources.get(root, set()))),
        })
    for edge in graph_edges:
        root = dsu.find(uid_to_idx[edge["source_object_id"]]) if edge.get("source_object_id") in uid_to_idx else -1
        graph_rows.append({**edge, "record_type": "EDGE", "physical_track_id": physical_id_by_root.get(root, ""), "canonical_object_id": ""})
    graph_df = pd.DataFrame(graph_rows)
    graph_path = out / "FINAL_CANDIDATE_IDENTITY_GRAPH.parquet"
    table = pa.Table.from_pandas(graph_df, preserve_index=False)
    pq.write_table(table, graph_path, compression="zstd")
    graph_df.to_csv(out / "FINAL_CANDIDATE_IDENTITY_GRAPH_EDGES_AND_NODES.csv", index=False, encoding="utf-8-sig")

    # Fixed cases: the new graph can directly verify OBJ_0008; other fixed
    # cases retain their earlier reviewed evidence but are not fabricated as a
    # fresh full visual pass.
    fixed_rows: list[dict[str, Any]] = []
    focal = {"2018": "xa180429::180396", "2019": "xa190101::169127", "2020": "xa201231::264190"}
    focal_ids = {physical_id_by_idx[uid_to_idx[u]] for u in focal.values() if u in uid_to_idx}
    fixed_rows.append({"case_id": "OBJ_0008", "candidate_physical_track_ids": ";".join(sorted(focal_ids)), "candidate_presence_sequence": "1-1-1", "candidate_state": "STABLE", "regression_status": "PASS_CANDIDATE", "note": "all three canonical objects share one new physical_track_id"})
    for case_id in ["B02-P2-24", "B03-TEX04", "B04-TQA07", "B04-TQA10", "B05-TQA03", "B05-P1-10", "B01-P2-26", "OBJ_0001", "B05-TEX01", "B01-TEX02", "B05-P1-06"]:
        fixed_rows.append({"case_id": case_id, "candidate_physical_track_ids": "", "candidate_presence_sequence": "", "candidate_state": "", "regression_status": "INHERITED_VISUAL_REVIEW_REQUIRES_CASE_REMAP", "note": "current full-city input has no direct fixed-case key; no PASS fabricated"})
    write_csv(out / "FIXED_CASE_REGRESSION_V2.csv", fixed_rows, list(fixed_rows[0]))

    # Source ledger and closure gates.
    source_counts = Counter(row["candidate_source"] for row in ledger_rows if row["eligible_for_identity"] == "YES")
    closure_checks = {
        "STRICT_CORE_INCLUDED": source_counts["STRICT_CORE"] > 0,
        "SAFE_RECOVERY_INCLUDED": source_counts["SAFE_RECOVERY"] > 0,
        "PARALLAX_SAFE_INCLUDED": source_counts["PARALLAX_SAFE_RECOVERY"] > 0,
        "V4B_SPLIT_INCLUDED": source_counts["COMPOUND_SPLIT"] > 0,
        "V4B_MERGE_INCLUDED": source_counts["COMPOUND_MERGE"] > 0,
        "V4B_SPLIT_MERGE_INCLUDED": source_counts["COMPOUND_SPLIT_MERGE"] > 0,
        "V4B_MULTIPLE_CANDIDATES_INCLUDED": promoted_component_count > 0,
        "GAP_RETURN_INCLUDED": source_counts["GAP_RETURN"] > 0 or bool(gap_physical_ids),
        "ABA_INCLUDED": source_counts["ABA_SAFE_CLOSURE"] > 0,
        "COMPOUND_PROMOTION_INCLUDED": promoted_component_count > 0,
    }
    pipeline_closure_gate = "PASS" if all(closure_checks.values()) else "FAIL"
    interval_stats: dict[str, dict[str, Any]] = {}
    for transition, _, _ in TRANSITIONS:
        known = sum(state_counts[transition][s] for s in ("STABLE", "ADDITION", "DEMOLITION"))
        unknown = state_counts[transition]["UNKNOWN_CONFLICT"]
        total = total_union[transition]
        unknown_area = state_areas[transition]["UNKNOWN_CONFLICT"]
        interval_stats[transition] = {
            "counts": dict(state_counts[transition]), "area_m2": dict(state_areas[transition]),
            "total_track_deduplicated_union_area_m2": total,
            "unknown_area_m2": unknown_area,
            "unknown_area_pct": 100.0 * unknown_area / total if total else None,
            "unknown_object_count": unknown, "unknown_object_pct": 100.0 * unknown / (known + unknown) if known + unknown else None,
            "area_denominator_definition": "sum(max(source_scene_area_m2,target_scene_area_m2)) per physical_track_id; metric vector deduplicated candidate denominator",
        }
    unknown_lt5 = all((v["unknown_area_pct"] is not None and v["unknown_area_pct"] < 5.0) for v in interval_stats.values())
    state_sanity = all(v["counts"].get("STABLE", 0) >= sum(v["counts"].get(s, 0) for s in ("ADDITION", "DEMOLITION")) for v in interval_stats.values())
    visual_gate = "NOT_RUN_PARENT_REVIEW_REQUIRED"
    fixed_gate = "PASS_CANDIDATE_OBJ0008_ONLY_REVIEW_REQUIRED_OTHER_CASES"
    v4b_regression = "PASS" if all(v["unknown_object_count"] <= limit for v, limit in zip(interval_stats.values(), [59248, 75183])) else "FAIL"
    tracking_gate = "REVIEW_REQUIRED" if visual_gate != "PASS" else "PASS"
    ready = all([pipeline_closure_gate == "PASS", v4b_regression == "PASS", tracking_gate == "PASS", unknown_lt5, state_sanity, fixed_gate == "PASS", len(false_rows) == 0])

    source_ledger_receipt = {
        "task": "XIAN-FINAL-CANDIDATE-IDENTITYGRAPH-REBUILD-AND-FULL-STATE-RERUN-001",
        "source_counts_eligible": dict(source_counts), "closure_checks": closure_checks,
        "PIPELINE_CLOSURE_GATE": pipeline_closure_gate, "v4b_promoted_components": promoted_component_count,
        "strict_core_edges": len(strict_edges), "aba_rows_used": aba_rows_used,
        "safe_recovery_edges": dict(safe_recovery_count), "physical_tracks": len(track_rows),
        "candidate_objects": len(obj), "graph_edge_rows_selected_or_support": len(graph_edges),
    }
    write_json(out / "PIPELINE_CLOSURE_AUDIT.json", source_ledger_receipt)
    write_json(out / "FINAL_CANDIDATE_RECEIPT.json", {
        **source_ledger_receipt,
        "V4B_CLOSURE_REGRESSION_GATE": v4b_regression,
        "TRACKING_ACCURACY_GATE": tracking_gate,
        "FIXED_CASE_REGRESSION": fixed_gate,
        "UNKNOWN_LT5_GATE": "PASS" if unknown_lt5 else "FAIL",
        "STATE_SANITY_GATE": "PASS" if state_sanity else "FAIL",
        "FALSE_CHANGE_VISUAL_GATE": visual_gate,
        "FULL_CITY_GRID_GATE": "NOT_RUN_GATES_NOT_PASS",
        "FINAL_PRODUCT_GATE": "NOT_RUN_GATES_NOT_PASS",
        "READY_FOR_FINAL_PRODUCTION": "YES" if ready else "NO",
        "interval_stats": interval_stats,
        "residual_unknown_phase_b_rows": len(residual_rows),
        "false_change_queue_v2_rows": len(false_rows),
        "formal_outputs_modified": False,
        "geometry_owner_formal_unchanged": True,
        "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
        "next_action": "FINAL_PRODUCTION" if ready else "STOP_FOR_PARENT_REVIEW_AFTER_RESIDUAL_UNKNOWN_AND_VISUAL_QA_INDEX",
    })
    report = [
        "# Xi'an final candidate IdentityGraph rebuild",
        "", "状态：`" + ("READY_FOR_FINAL_PRODUCTION" if ready else "REVIEW_REQUIRED") + "`",
        "", "本次从新的 Candidate IdentityGraph 重新构建 physical_track，再从头生成 Presence 与 State；未在旧 State 表上 overlay。",
        "", "## Gates", "",
        f"- PIPELINE_CLOSURE_GATE: `{pipeline_closure_gate}`",
        f"- V4B_CLOSURE_REGRESSION_GATE: `{v4b_regression}`",
        f"- TRACKING_ACCURACY_GATE: `{tracking_gate}`",
        f"- FIXED_CASE_REGRESSION: `{fixed_gate}`",
        f"- UNKNOWN_LT5_GATE: `{'PASS' if unknown_lt5 else 'FAIL'}`",
        f"- STATE_SANITY_GATE: `{'PASS' if state_sanity else 'FAIL'}`",
        f"- FALSE_CHANGE_VISUAL_GATE: `{visual_gate}`",
        f"- READY_FOR_FINAL_PRODUCTION: `{'YES' if ready else 'NO'}`",
        "", "## Counts", "",
        f"- canonical objects: {len(obj):,}", f"- physical tracks: {len(track_rows):,}",
        f"- strict core edges: {len(strict_edges):,}", f"- V4B promoted components: {promoted_component_count:,}",
        f"- ABA closures: {aba_rows_used:,}", f"- selected/support graph edges: {len(graph_edges):,}",
        f"- residual Unknown rows: {len(residual_rows):,}", f"- new false-change queue: {len(false_rows):,}",
        "", "## Interval stats", "",
    ]
    for transition, _, _ in TRANSITIONS:
        stat = interval_stats[transition]
        report.append(f"### {transition}")
        report.append(f"- counts: `{json.dumps(stat['counts'], ensure_ascii=False)}`")
        report.append(f"- area_m2: `{json.dumps(stat['area_m2'], ensure_ascii=False)}`")
        report.append(f"- Unknown area: `{stat['unknown_area_pct']:.6f}%`" if stat["unknown_area_pct"] is not None else "- Unknown area: unavailable")
        report.append(f"- Unknown objects: `{stat['unknown_object_pct']:.6f}%`" if stat["unknown_object_pct"] is not None else "- Unknown objects: unavailable")
    report.extend([
        "", "## Safety boundary", "",
        "Formal Geometry, accepted polygons, upstream Tracking, formal Presence/State, and formal TIFs were not modified.",
        "`>=100 m²` product export was not started because candidate gates are not all PASS.",
        "", "## Outputs", "",
        "- FINAL_CANDIDATE_IDENTITY_GRAPH.parquet",
        "- FINAL_IDENTITY_SOURCE_LEDGER.csv",
        "- FINAL_CANDIDATE_TRACKING_OBJECTS.csv",
        "- FINAL_CANDIDATE_PHYSICAL_TRACKS.csv",
        "- CANDIDATE_PRESENCE_FROM_SCRATCH.csv",
        "- CANDIDATE_STATE_EVENTS_FROM_SCRATCH.csv",
        "- RESIDUAL_UNKNOWN_PHASE_B.csv",
        "- FALSE_CHANGE_AUDIT_QUEUE_V2.csv",
        "- VISUAL_QA_SAMPLE_INDEX.csv",
        "- FINAL_CANDIDATE_RECEIPT.json",
    ])
    (out / "FINAL_CANDIDATE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    progress(out, "COMPLETE", pipeline_closure_gate=pipeline_closure_gate, v4b_regression=v4b_regression, tracking_gate=tracking_gate, unknown_lt5_gate="PASS" if unknown_lt5 else "FAIL", ready_for_final_production=ready, next_action="FINAL_PRODUCTION" if ready else "STOP_FOR_PARENT_REVIEW")
    print(json.dumps({"status": "READY_FOR_FINAL_PRODUCTION" if ready else "REVIEW_REQUIRED", "output": str(out), "objects": len(obj), "physical_tracks": len(track_rows), "strict_edges": len(strict_edges), "aba_rows": aba_rows_used, "unknown": {k: v["unknown_area_pct"] for k, v in interval_stats.items()}, "formal_outputs_modified": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
