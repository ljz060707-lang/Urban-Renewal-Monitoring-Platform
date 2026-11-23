"""Read-only identity continuity audit and ABA candidate scan for Xi'an R4.

This helper deliberately writes only to a new candidate output directory.  It
does not modify the formal identity graph, polygons, presence, state, or TIFs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FORMAL = Path("/data/xian_r4_100km2_20260914/output/西安_R4_辅助影像接入前_三期三态候选版_20260914")
OUT = Path("/data/xian_r4_100km2_20260914/output/13_visual_asset_repair_and_obj0008_identity_closure_20260916/identity_candidate")
PAIR_DIR = FORMAL / "_parallel_tracking_work" / "pair_edges"
TRACKING_CSV = FORMAL / "04_identity_graph" / "tracking_buildings.csv"

SCENES = ("xa180429", "xa190101", "xa201231")
PAIRS = {
    "2018_2019": ("xa180429", "xa190101", PAIR_DIR / "xa180429__xa190101.pkl"),
    "2019_2020": ("xa190101", "xa201231", PAIR_DIR / "xa190101__xa201231.pkl"),
    "2018_2020": ("xa180429", "xa201231", PAIR_DIR / "xa180429__xa201231.pkl"),
}

FOCAL = {
    "2018": "xa180429::180396",
    "2019": "xa190101::169127",
    "2020": "xa201231::264190",
}


def f(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def finite(value: float) -> bool:
    return math.isfinite(value)


def edge_key(edge: dict[str, Any]) -> tuple[str, str]:
    return str(edge.get("source_object_id")), str(edge.get("target_object_id"))


def common_fields(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_object_id": edge.get("source_object_id"),
        "target_object_id": edge.get("target_object_id"),
        "from_scene": edge.get("from_scene"),
        "to_scene": edge.get("to_scene"),
        "relation": edge.get("relation"),
        "candidate_relation": edge.get("candidate_relation"),
        "split_merge_type": edge.get("split_merge_type"),
        "score": f(edge.get("score")),
        "identity_confidence": f(edge.get("identity_confidence")),
        "buffered_iou": f(edge.get("buffered_iou")),
        "polygon_iou": f(edge.get("polygon_iou")),
        "bbox_iou": f(edge.get("bbox_iou")),
        "area_ratio": f(edge.get("area_ratio")),
        "alignment_aware_displacement_m": f(edge.get("alignment_aware_displacement_m")),
        "local_neighborhood_consistency": f(edge.get("local_neighborhood_consistency")),
        "identity_conflict": b(edge.get("identity_conflict")),
        "mutual_best": b(edge.get("mutual_best")),
        "accepted_for_identity": b(edge.get("accepted_for_identity")),
        "candidate_count_from_source": edge.get("candidate_count_from_source"),
        "candidate_count_to_target": edge.get("candidate_count_to_target"),
        "reasonable_candidate_count_from_source": edge.get("reasonable_candidate_count_from_source"),
        "reasonable_candidate_count_to_target": edge.get("reasonable_candidate_count_to_target"),
        "score_gap_source": f(edge.get("score_gap_source")),
        "score_gap_target": f(edge.get("score_gap_target")),
        "gap_return": b(edge.get("gap_return")),
        "gap_return_building_id": edge.get("gap_return_building_id"),
    }


def strong_direct(edge: dict[str, Any]) -> bool:
    """Conservative direct 2018->2020 continuity evidence."""
    return (
        b(edge.get("accepted_for_identity"))
        and str(edge.get("relation")) == "1:1"
        and str(edge.get("candidate_relation")) == "1:1"
        and b(edge.get("mutual_best"))
        and not b(edge.get("identity_conflict"))
        and f(edge.get("score"), -1.0) >= 0.60
        and f(edge.get("buffered_iou"), -1.0) >= 0.70
        and f(edge.get("area_ratio"), -1.0) >= 0.55
        and f(edge.get("local_neighborhood_consistency"), -1.0) >= 0.50
        and f(edge.get("alignment_aware_displacement_m"), float("inf")) <= 20.0
    )


def bridge_eligible(edge: dict[str, Any]) -> bool:
    """Evidence filter for a unique middle-epoch bridge; no automatic merge."""
    return (
        b(edge.get("mutual_best"))
        and not b(edge.get("identity_conflict"))
        and f(edge.get("score"), -1.0) >= 0.35
        and f(edge.get("buffered_iou"), -1.0) >= 0.55
        and f(edge.get("area_ratio"), -1.0) >= 0.45
        and f(edge.get("local_neighborhood_consistency"), -1.0) >= 0.40
        and f(edge.get("alignment_aware_displacement_m"), float("inf")) <= 25.0
        and int(f(edge.get("reasonable_candidate_count_from_source"), 99)) <= 1
        and int(f(edge.get("reasonable_candidate_count_to_target"), 99)) <= 1
    )


def load_pair(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    return list(payload["edges"])


def load_track_metadata(uids: set[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    if not uids:
        return result
    with TRACKING_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            uid = str(row.get("object_uid", ""))
            if uid in uids:
                result[uid] = {
                    "formal_date": row.get("formal_date", ""),
                    "building_id": row.get("building_id", ""),
                    "track_id": row.get("track_id", ""),
                    "area_m2": row.get("area_m2", ""),
                    "scientific_status": row.get("scientific_status", ""),
                }
    return result


def strict_eligibility(edge: dict[str, Any]) -> tuple[bool, str]:
    if b(edge.get("identity_conflict")):
        return False, "IDENTITY_CONFLICT"
    if not b(edge.get("mutual_best")):
        return False, "NOT_MUTUAL_BEST"
    if str(edge.get("relation")) != "1:1":
        return False, f"RELATION_{edge.get('relation')}"
    if f(edge.get("score"), -1) < 0.35:
        return False, "SCORE_BELOW_035"
    if not b(edge.get("accepted_for_identity")):
        return False, "NOT_ACCEPTED_BY_CURRENT_RULE"
    return True, "ACCEPTED_CURRENT_RULE"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def local_graph(edges_by_pair: dict[str, list[dict[str, Any]]], metadata: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    focal_ids = set(FOCAL.values())
    selected: list[dict[str, Any]] = []
    nodes: set[str] = set(focal_ids)
    for pair_name, edges in edges_by_pair.items():
        for edge in edges:
            src, dst = edge_key(edge)
            if src in focal_ids or dst in focal_ids:
                ok, reason = strict_eligibility(edge)
                row = common_fields(edge)
                row.update({"pair": pair_name, "strict_eligibility": ok, "strict_reason": reason})
                selected.append(row)
                nodes.update((src, dst))
    # Add all edge records between the local candidate nodes, including
    # competitor and conflict edges that are not incident to the focal node.
    for pair_name, edges in edges_by_pair.items():
        for edge in edges:
            src, dst = edge_key(edge)
            if src in nodes and dst in nodes:
                ok, reason = strict_eligibility(edge)
                row = common_fields(edge)
                row.update({"pair": pair_name, "strict_eligibility": ok, "strict_reason": reason})
                selected.append(row)
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in selected:
        dedup[(str(row["pair"]), str(row["source_object_id"]), str(row["target_object_id"]))] = row
    selected = list(dedup.values())
    node_rows = []
    for uid in sorted(nodes):
        scene = uid.split("::", 1)[0]
        node_rows.append({
            "object_id": uid,
            "scene": scene,
            "epoch": {"xa180429": "2018", "xa190101": "2019", "xa201231": "2020"}.get(scene, ""),
            "is_focal": uid in focal_ids,
            **metadata.get(uid, {}),
        })
    return selected, {"nodes": node_rows, "edges": selected, "focal": FOCAL}


def draw_graph(graph: dict[str, Any], path: Path) -> None:
    # A dependency-light, auditable schematic: columns are actual epochs;
    # colors encode current acceptance/conflict status, not a new geometry.
    from PIL import Image, ImageDraw, ImageFont

    width, height = 2200, max(900, 180 + 115 * max(1, len(graph["nodes"])))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        small = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    columns = {"xa180429": 380, "xa190101": 1100, "xa201231": 1820}
    epoch_name = {"xa180429": "2018", "xa190101": "2019", "xa201231": "2020"}
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in graph["nodes"]:
        by_scene[node["scene"]].append(node)
    pos: dict[str, tuple[int, int]] = {}
    for scene, x in columns.items():
        draw.text((x - 50, 30), epoch_name[scene], fill="black", font=font)
        rows = sorted(by_scene.get(scene, []), key=lambda n: (not n.get("is_focal", False), n["object_id"]))
        for i, node in enumerate(rows):
            y = 115 + i * 110
            uid = node["object_id"]
            pos[uid] = (x, y)
            focal = node.get("is_focal", False)
            fill = "#d7f5d7" if focal else "#f1f1f1"
            outline = "#147a26" if focal else "#555555"
            draw.ellipse((x - 15, y - 15, x + 15, y + 15), fill=fill, outline=outline, width=3)
            label = uid.split("::")[-1]
            track = node.get("track_id") or node.get("building_id") or ""
            draw.text((x - 115, y + 24), f"{label} {track}", fill="black", font=small)
    for edge in graph["edges"]:
        src, dst = edge["source_object_id"], edge["target_object_id"]
        if src not in pos or dst not in pos:
            continue
        accepted = bool(edge.get("accepted_for_identity"))
        conflict = bool(edge.get("identity_conflict"))
        color = "#16823b" if accepted else ("#c1121f" if conflict else "#d97706")
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        draw.line((x1 + 18, y1, x2 - 18, y2), fill=color, width=4 if accepted else 2)
        midx, midy = (x1 + x2) // 2, (y1 + y2) // 2
        score = edge.get("score")
        rel = edge.get("relation")
        draw.text((midx - 45, midy - 20), f"{rel} s={f(score):.3f}", fill=color, font=small)
    draw.text((50, height - 55), "green=accepted current 1:1; orange=rejected/competitor; red=identity conflict; focal nodes highlighted", fill="black", font=small)
    image.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pair_edges: dict[str, list[dict[str, Any]]] = {}
    for pair_name, (_from, _to, path) in PAIRS.items():
        pair_edges[pair_name] = load_pair(path)

    # Local OBJ_0008 graph: all records touching its three observed objects,
    # plus records among their direct competitors.
    local_rows, graph = local_graph(pair_edges, load_track_metadata(set(FOCAL.values())))
    write_csv(OUT / "OBJ0008_LOCAL_IDENTITY_GRAPH.csv", local_rows)
    (OUT / "OBJ0008_LOCAL_GRAPH.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    draw_graph(graph, OUT / "OBJ0008_LOCAL_GRAPH.png")

    # Build a compact bridge index.  Only edge records that could satisfy the
    # general ABA rule are retained in memory; no edge is promoted here.
    ab = pair_edges["2018_2019"]
    bc = pair_edges["2019_2020"]
    ac = pair_edges["2018_2020"]
    ab_by_a: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bc_by_b: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in ab:
        if bridge_eligible(edge):
            ab_by_a[str(edge["source_object_id"])].append(edge)
    for edge in bc:
        if bridge_eligible(edge):
            bc_by_b[str(edge["source_object_id"])].append(edge)

    candidates: list[dict[str, Any]] = []
    direct_count = 0
    for direct in ac:
        if not strong_direct(direct):
            continue
        direct_count += 1
        a, c = edge_key(direct)
        bridges: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for edge_ab in ab_by_a.get(a, []):
            middle = str(edge_ab["target_object_id"])
            for edge_bc in bc_by_b.get(middle, []):
                if str(edge_bc["target_object_id"]) == c:
                    bridges.append((edge_ab, edge_bc))
        unique_middle = sorted({str(pair[0]["target_object_id"]) for pair in bridges})
        for middle in unique_middle or [""]:
            bridge_pair = next(((x, y) for x, y in bridges if str(x["target_object_id"]) == middle), (None, None))
            edge_ab, edge_bc = bridge_pair
            no_competitor = len(unique_middle) == 1
            safe = bool(edge_ab and edge_bc and no_competitor)
            reasons = []
            if not edge_ab or not edge_bc:
                reasons.append("MIDDLE_EPOCH_UNIQUELY_SUPPORTED=False")
            if not no_competitor:
                reasons.append("MULTIPLE_MIDDLE_CANDIDATES")
            if b(direct.get("identity_conflict")):
                reasons.append("DIRECT_IDENTITY_CONFLICT")
            if safe:
                reasons.append("ALL_EDGE_EVIDENCE_FILTERS_PASS; VISUAL_QA_REQUIRED")
            row = {
                "pattern": "ABA_IDENTITY_DISCONTINUITY_CANDIDATE",
                "source_2018": a,
                "middle_2019": middle,
                "target_2020": c,
                "candidate_count_middle": len(unique_middle),
                "direct_2018_2020_accepted": True,
                "direct_score": f(direct.get("score")),
                "direct_buffered_iou": f(direct.get("buffered_iou")),
                "direct_area_ratio": f(direct.get("area_ratio")),
                "direct_displacement_m": f(direct.get("alignment_aware_displacement_m")),
                "direct_neighborhood_consistency": f(direct.get("local_neighborhood_consistency")),
                "direct_relation": direct.get("relation"),
                "direct_identity_conflict": b(direct.get("identity_conflict")),
                "bridge_ab_relation": edge_ab.get("relation") if edge_ab else "",
                "bridge_ab_score": f(edge_ab.get("score")) if edge_ab else float("nan"),
                "bridge_ab_buffered_iou": f(edge_ab.get("buffered_iou")) if edge_ab else float("nan"),
                "bridge_ab_area_ratio": f(edge_ab.get("area_ratio")) if edge_ab else float("nan"),
                "bridge_ab_displacement_m": f(edge_ab.get("alignment_aware_displacement_m")) if edge_ab else float("nan"),
                "bridge_bc_relation": edge_bc.get("relation") if edge_bc else "",
                "bridge_bc_score": f(edge_bc.get("score")) if edge_bc else float("nan"),
                "bridge_bc_buffered_iou": f(edge_bc.get("buffered_iou")) if edge_bc else float("nan"),
                "bridge_bc_area_ratio": f(edge_bc.get("area_ratio")) if edge_bc else float("nan"),
                "bridge_bc_displacement_m": f(edge_bc.get("alignment_aware_displacement_m")) if edge_bc else float("nan"),
                "no_strong_competitor": no_competitor,
                "no_reliable_absence": "EVIDENCE_REQUIRES_PRESENCE_RECOMPUTE",
                "decision": "SAFE_ABA_IDENTITY_CLOSURE_CANDIDATE" if safe else "KEEP_REVIEW",
                "reason": "; ".join(reasons),
            }
            candidates.append(row)

    # Add metadata after the scan, only for candidate endpoints.
    candidate_uids = set()
    for row in candidates:
        candidate_uids.update(x for x in (row["source_2018"], row["middle_2019"], row["target_2020"]) if x)
    meta = load_track_metadata(candidate_uids)
    for row in candidates:
        for prefix, key in (("source_2018", "source_2018"), ("middle_2019", "middle_2019"), ("target_2020", "target_2020")):
            uid = row[key]
            m = meta.get(uid, {})
            row[f"{prefix}_track_id"] = m.get("track_id", "")
            row[f"{prefix}_building_id"] = m.get("building_id", "")
            row[f"{prefix}_area_m2"] = m.get("area_m2", "")

    write_csv(OUT / "ABA_IDENTITY_DISCONTINUITY_CANDIDATES.csv", candidates)
    safe_rows = [r for r in candidates if r["decision"] == "SAFE_ABA_IDENTITY_CLOSURE_CANDIDATE"]
    write_csv(OUT / "ABA_SAFE_CLOSURE_CANDIDATES.csv", safe_rows)

    focal_rows = [r for r in candidates if FOCAL["2018"] == r["source_2018"] and FOCAL["2020"] == r["target_2020"]]
    focal_safe = [r for r in focal_rows if r["decision"] == "SAFE_ABA_IDENTITY_CLOSURE_CANDIDATE"]
    root_cause = f"""# OBJ_0008 root cause and candidate closure

## Evidence

- 2018 object: `{FOCAL['2018']}`; current track: `TRK_R4X_00180326`.
- 2019 object: `{FOCAL['2019']}`; current track: `TRK_R4X_00284557`.
- 2020 object: `{FOCAL['2020']}`; current track: `TRK_R4X_00180326`.
- The three objects are already in compound component `CMP_R4FAST_00077138` in the formal summaries.
- The local edge table above contains accepted, rejected, competitor, and conflict records from the unchanged formal pair-edge artifacts.

## Mechanism found in the current implementation

`run_xian_latest_pre_supplement_cloud.py::pair_edges()` marks an edge as `accepted_for_identity` only when it is a decisive mutual-best `1:1` edge.  `N:1` and `1:N` edges remain diagnostics even when their geometric and neighborhood evidence is strong.  `run_identity()` then unions only the accepted-by-pair edges; it has no general ABA/compound-identity promotion step that can attach a unique middle-epoch node to a strong 2018-to-2020 physical continuity.

For this case, the 2018-to-2019 record is `N:1`/`MERGE` and is not accepted; the 2019-to-2020 record is `1:N`/`SPLIT` and is not accepted; the direct 2018-to-2020 record is a strong accepted `1:1` edge with `gap_return=True`.  Therefore 2018 and 2020 share a component while the 2019 object remains on a separate track.  The resulting `101` versus `U1U` / `UNKNOWN_CONFLICT` is an identity continuity defect, not a geometry edit.

## General candidate rule

The candidate scan is general, not an `OBJ_0008` special case.  A closure candidate requires: a strong accepted direct 2018-to-2020 edge; a uniquely supported 2019 middle object through both adjacent edges; no edge conflict; mutual-best and reasonable-candidate uniqueness; bounded alignment-aware displacement; acceptable overlap/area/neighborhood metrics; and no strong middle competitor.  It is still a candidate until visual QA and a recomputed Presence/State audit establish no reliable absence.  No polygon or formal output is changed by this run.

## Current scan result

- strong direct 2018-to-2020 edges scanned: `{direct_count}`
- ABA candidate rows: `{len(candidates)}`
- safe-closure candidates before visual QA: `{len(safe_rows)}`
- OBJ_0008 safe candidate rows: `{len(focal_safe)}`

The output is therefore a promotion candidate, not a formal Tracking promotion.
"""
    (OUT / "OBJ0008_ROOT_CAUSE.md").write_text(root_cause, encoding="utf-8")

    receipt = {
        "task": "XIAN-VISUAL-ASSET-REPAIR-AND-OBJ0008-IDENTITY-CLOSURE-001",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_outputs_modified": False,
        "pair_edge_source": "formal _parallel_tracking_work/pair_edges/*.pkl",
        "local_graph_edge_count": len(local_rows),
        "local_graph_node_count": len(graph["nodes"]),
        "strong_direct_2018_2020_count": direct_count,
        "aba_candidate_count": len(candidates),
        "aba_safe_candidate_count_before_visual_qa": len(safe_rows),
        "obj0008_focal_candidate_count": len(focal_rows),
        "obj0008_safe_candidate_count_before_visual_qa": len(focal_safe),
        "focal_uids": FOCAL,
        "general_rule_requires_visual_qa": True,
        "promotion_performed": False,
    }
    (OUT / "IDENTITY_CLOSURE_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
