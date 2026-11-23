"""Run one bounded Xi'an R4 Tracking-only development candidate.

This module deliberately consumes the frozen R3 geometry and the read-only R2
scene inputs.  The only scientific-layer change is the ordering bug fix in
the R3 tracking propagation: best-edge dictionaries are built after candidate
edges are grouped.  Non-mutual or low-margin links remain explicit ambiguity
and are never forced into an ID chain.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


def load_r3_module() -> Any:
    path = Path(__file__).with_name("run_xian_r3_vision_guided_repair.py")
    spec = importlib.util.spec_from_file_location("xian_r3_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load R3 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r3 = load_r3_module()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


class R4TrackingRunner(r3.R3Runner):
    """R3 downstream recomputation with one isolated Tracking-layer change."""

    def __init__(self, r2_root: Path, r3_root: Path, out_root: Path) -> None:
        super().__init__(r2_root, out_root, max_instance_rounds=3)
        self.r3_root = r3_root
        self.instance_records = []
        self.semantic = pd.DataFrame(columns=["r3_object_id", "semantic_status"])

    def assert_inputs(self) -> None:
        super().assert_inputs()
        required = [
            self.r3_root / "03_最终Geometry",
            self.r3_root / "11_固定案例前后对照" / "固定案例_R2_vs_R3.csv",
            self.r3_root / "15_最终报告" / "FINAL_REPORT.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("R3 frozen input missing: " + "; ".join(missing))

    def prepare_output(self) -> None:
        for name in [
            "00_任务说明",
            "01_基线引用",
            "03_最终Geometry",
            "04_Tracking",
            "05_Gap_Return",
            "06_Observability",
            "07_Presence",
            "08_State",
            "11_固定案例前后对照",
            "12_独立验收",
            "13_迭代记录",
            "14_R4开发参数",
            "15_运行回执",
        ]:
            self.out_path(name).mkdir(parents=True, exist_ok=True)

    def load_frozen_geometry(self) -> None:
        manifest: list[dict[str, Any]] = []
        for block in r3.BLOCKS:
            for date in r3.DATES:
                source_dir = self.r3_root / "03_最终Geometry" / block / date
                source_gpkg = source_dir / "R3_FINAL_GEOMETRY.gpkg"
                source_csv = source_dir / "R3_FINAL_GEOMETRY.csv"
                if not source_gpkg.exists() or not source_csv.exists():
                    raise FileNotFoundError(f"frozen R3 geometry missing: {block} {date}")
                gdf = gpd.read_file(source_gpkg, layer="buildings")
                if gdf.crs is None:
                    raise ValueError(f"frozen R3 geometry CRS missing: {source_gpkg}")
                gdf = gdf.to_crs(r3.METRIC_CRS)
                if gdf.empty or gdf.geometry.is_empty.any():
                    raise ValueError(f"frozen R3 geometry empty/invalid: {source_gpkg}")
                self.r3_by_scene[(block, date)] = gdf
                target_dir = self.out_path("03_最终Geometry", block, date)
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_gpkg, target_dir / "R3_FINAL_GEOMETRY.gpkg")
                shutil.copy2(source_csv, target_dir / "R3_FINAL_GEOMETRY.csv")
                manifest.append(
                    {
                        "block_id": block,
                        "actual_date": date,
                        "source_gpkg": str(source_gpkg),
                        "source_gpkg_sha256": sha256_file(source_gpkg),
                        "source_csv_sha256": sha256_file(source_csv),
                        "object_count": int(len(gdf)),
                        "crs": str(gdf.crs),
                    }
                )
        json_dump(self.out_path("01_基线引用", "R3_GEOMETRY_MANIFEST.json"), manifest)

    def run_tracking(self) -> None:  # noqa: C901
        all_objects: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for block in r3.BLOCKS:
            date_gdfs = [self.r3_by_scene[(block, date)] for date in r3.DATES]
            local_object_tracks: dict[str, str] = {}
            next_track = 1
            for date, gdf in zip(r3.DATES, date_gdfs):
                for _, row in gdf.iterrows():
                    oid = str(row.r3_object_id)
                    if oid not in local_object_tracks:
                        local_object_tracks[oid] = f"TRK_R4_{block}_{next_track:06d}"
                        next_track += 1
                    all_objects.append(
                        {
                            "block_id": block,
                            "actual_date": date,
                            "r3_object_id": oid,
                            "source_object_id": row.source_object_id,
                            "area_m2": float(row.geometry.area),
                            "perimeter_m": float(row.geometry.length),
                            "geometry_hash": r3.geometry_hash(row.geometry),
                            "centroid_x": float(row.geometry.centroid.x),
                            "centroid_y": float(row.geometry.centroid.y),
                            "instance_status": row.instance_status,
                            "track_id": local_object_tracks[oid],
                            "scientific_status": r3.SCIENTIFIC_STATUS,
                        }
                    )
                if date == r3.DATES[0]:
                    continue
                prev_date = r3.DATES[r3.DATES.index(date) - 1]
                prev = self.r3_by_scene[(block, prev_date)]
                curr = gdf
                candidates = self._edge_candidates(
                    prev,
                    curr,
                    self.scenes[(block, prev_date)],
                    prev_date,
                    date,
                )
                by_source: dict[int, list[dict[str, Any]]] = {}
                by_target: dict[int, list[dict[str, Any]]] = {}
                for edge in candidates:
                    by_source.setdefault(int(edge["source_index"]), []).append(edge)
                    by_target.setdefault(int(edge["target_index"]), []).append(edge)
                # R3 computed these before populating the dictionaries.  That
                # made every propagation lookup empty and silently disabled
                # temporal continuity.  This is the sole R4 change.
                source_best = {
                    index: max(values, key=lambda item: item["score"])
                    for index, values in by_source.items()
                }
                target_best = {
                    index: max(values, key=lambda item: item["score"])
                    for index, values in by_target.items()
                }
                for edge in candidates:
                    source_edges = by_source[edge["source_index"]]
                    target_edges = by_target[edge["target_index"]]
                    source_sorted = sorted(source_edges, key=lambda item: item["score"], reverse=True)
                    target_sorted = sorted(target_edges, key=lambda item: item["score"], reverse=True)
                    gap_source = source_sorted[0]["score"] - (source_sorted[1]["score"] if len(source_sorted) > 1 else 0.0)
                    gap_target = target_sorted[0]["score"] - (target_sorted[1]["score"] if len(target_sorted) > 1 else 0.0)
                    relation = "1_TO_1"
                    if len(source_edges) > 1 and len(target_edges) > 1:
                        relation = "AMBIGUOUS_SPLIT_MERGE"
                    elif len(source_edges) > 1:
                        relation = "1_TO_N"
                    elif len(target_edges) > 1:
                        relation = "N_TO_1"
                    conflict = bool(
                        edge["local_neighborhood_consistency"] < 0.45
                        and (gap_source < 0.08 or gap_target < 0.08)
                    )
                    if conflict:
                        relation = "IDENTITY_CONFLICT"
                    out_edge = dict(edge)
                    out_edge.update(
                        {
                            "block_id": block,
                            "from_date": prev_date,
                            "to_date": date,
                            "relation": relation,
                            "identity_conflict": conflict,
                            "candidate_count_from_source": len(source_edges),
                            "candidate_count_to_target": len(target_edges),
                            "score_gap_source": gap_source,
                            "score_gap_target": gap_target,
                            "mutual_best": bool(
                                source_best.get(edge["source_index"]) is edge
                                and target_best.get(edge["target_index"]) is edge
                            ),
                            "scientific_status": r3.SCIENTIFIC_STATUS,
                        }
                    )
                    edges.append(out_edge)
                # Only a mutual, sufficiently supported link propagates an ID.
                # All other links remain in the explicit edge table.
                for edge in candidates:
                    mutual_best = (
                        source_best.get(edge["source_index"]) is edge
                        and target_best.get(edge["target_index"]) is edge
                    )
                    source_edges = by_source[edge["source_index"]]
                    decisive_geometry = (
                        edge["iou"] >= 0.15
                        and edge["local_neighborhood_consistency"] >= 0.45
                    )
                    decisive_margin = (
                        edge["score"]
                        - (
                            sorted(source_edges, key=lambda item: item["score"], reverse=True)[1]["score"]
                            if len(source_edges) > 1
                            else 0.0
                        )
                    ) >= 0.05
                    if not (mutual_best and edge["score"] >= 0.35 and (decisive_geometry or decisive_margin)):
                        continue
                    source_oid = str(prev.iloc[edge["source_index"]].r3_object_id)
                    target_oid = str(curr.iloc[edge["target_index"]].r3_object_id)
                    source_track = local_object_tracks.get(source_oid)
                    if source_track:
                        local_object_tracks[target_oid] = source_track
            for row in all_objects:
                if row["block_id"] == block:
                    row["track_id"] = local_object_tracks.get(row["r3_object_id"], row["track_id"])
        self.track_objects = pd.DataFrame(all_objects)
        self.track_edges = pd.DataFrame(edges)
        self.track_objects.to_csv(self.out_path("04_Tracking", "R4_TRACKING_OBJECTS.csv"), index=False, encoding="utf-8-sig")
        self.track_edges.to_csv(self.out_path("04_Tracking", "R4_TRACKING_EDGES.csv"), index=False, encoding="utf-8-sig")
        json_dump(
            self.out_path("04_Tracking", "TRACKING_RECEIPT.json"),
            {
                "status": "COMPLETED_R4_TRACKING_ONLY_CANDIDATE",
                "change_scope": "TRACKING_ONLY",
                "r3_propagation_order_bug_fixed": True,
                "mutual_best_required": True,
                "non_mutual_links_remain_explicit": True,
                "forced_nearest_neighbor_ids": False,
                "scientific_status": r3.SCIENTIFIC_STATUS,
            },
        )

    def run_state(self) -> None:
        rows: list[dict[str, Any]] = []
        conflict_tracks: set[str] = set()
        if not self.track_edges.empty:
            object_to_track = dict(zip(self.track_objects.r3_object_id.astype(str), self.track_objects.track_id.astype(str)))
            for _, edge in self.track_edges[self.track_edges.identity_conflict.astype(bool)].iterrows():
                for key in ("source_object_id", "target_object_id"):
                    track = object_to_track.get(str(edge[key]))
                    if track:
                        conflict_tracks.add(track)
        for track_id, group in self.presence.groupby("track_id"):
            values = {str(row.actual_date): str(row.presence) for _, row in group.iterrows()}
            sequence = "-".join(values[date] for date in r3.DATES)
            if str(track_id) in conflict_tracks:
                state, event = "CONFLICT_DIAGNOSTIC", "IDENTITY_CONFLICT"
            elif sequence == "1-1-1":
                state, event = "STABLE", "NONE"
            elif sequence == "1-1-0":
                state, event = "DEMOLITION", "DEMOLITION"
            elif sequence in {"0-0-1", "0-1-1"}:
                state, event = "ADDITION", "ADDITION"
            elif sequence in {"1-U-1", "1-0-U", "1-0-0"}:
                state, event = "UNKNOWN_DIAGNOSTIC", "UNKNOWN_SUPPORT"
            else:
                state, event = "UNKNOWN_DIAGNOSTIC", "UNKNOWN_SUPPORT"
            rows.append(
                {
                    "track_id": track_id,
                    "block_id": str(group.block_id.iloc[0]),
                    "presence_sequence": sequence,
                    "state": state,
                    "event": event,
                    "state_formal_gt": "NOT_EVALUATED",
                    "scientific_status": r3.SCIENTIFIC_STATUS,
                }
            )
        self.state = pd.DataFrame(rows)
        self.state.to_csv(self.out_path("08_State", "R4_STATES.csv"), index=False, encoding="utf-8-sig")
        json_dump(
            self.out_path("08_State", "STATE_RECEIPT.json"),
            {
                "status": "COMPLETED_DOWNSTREAM_OF_TRACKING_CANDIDATE",
                "formal_states": ["STABLE", "ADDITION", "DEMOLITION"],
                "diagnostic_states": ["UNKNOWN_DIAGNOSTIC", "CONFLICT_DIAGNOSTIC"],
                "state_accuracy": "NOT_EVALUATED",
                "scientific_status": r3.SCIENTIFIC_STATUS,
            },
        )

    def write_receipts(self) -> None:
        fixed_path = self.out_path("11_固定案例前后对照", "固定案例_R2_vs_R3.csv")
        if fixed_path.exists():
            shutil.copy2(fixed_path, self.out_path("11_固定案例前后对照", "R4_FIXED_CASES.csv"))
        presence_path = self.out_path("07_Presence", "R3_PRESENCE_1_0_U.csv")
        if presence_path.exists():
            shutil.copy2(presence_path, self.out_path("07_Presence", "R4_PRESENCE_1_0_U.csv"))
        json_dump(
            self.out_path("00_任务说明", "R4_TASK_SPEC.json"),
            {
                "task_id": "XIAN-R4-AUTONOMOUS-CLOUD-OPTIMIZATION-AND-CITYWIDE-PRODUCTION-001",
                "phase": "A_DEVELOPMENT_TRACKING_ONLY",
                "baseline": "R3_FROZEN_GEOMETRY",
                "change_layer": "TRACKING",
                "b0_threshold": 0.50,
                "area_filter_used_scientifically": False,
                "holdout_read": False,
                "raw_and_r2_inputs_modified": False,
                "scientific_status": r3.SCIENTIFIC_STATUS,
            },
        )
        pd.DataFrame(
            [
                {
                    "iteration_id": "R4-TRK-01",
                    "layer": "Tracking",
                    "hypothesis": "R3 best-edge dictionaries were computed before grouping, disabling propagation.",
                    "change": "Compute source/target best after grouping; require mutual-best and margin/geometry support.",
                    "b0_changed": False,
                    "instance_changed": False,
                    "presence_rule_changed": False,
                    "state_downstream_recomputed": True,
                    "decision": "PENDING_INDEPENDENT_ACCEPTANCE",
                    "scientific_status": r3.SCIENTIFIC_STATUS,
                }
            ]
        ).to_csv(self.out_path("13_迭代记录", "R4_ITERATION_LOG.csv"), index=False, encoding="utf-8-sig")
        json_dump(
            self.out_path("14_R4开发参数", "R4_DEVELOPMENT_CANDIDATE.json"),
            {
                "profile_name": "R4_TRACKING_PROPAGATION_ORDER_FIX_CANDIDATE",
                "b0_threshold": 0.50,
                "area_product_threshold_m2": 100.0,
                "instance": "R3_FROZEN_GEOMETRY_UNCHANGED",
                "alignment": "R2_READ_ONLY",
                "tracking": "R4_MUTUAL_BEST_PROPAGATION_ORDER_FIX",
                "presence": "R3_RULES_RECOMPUTED_DOWNSTREAM",
                "state": "R4_FORMAL_NAMES_WITH_DIAGNOSTIC_UNKNOWN_CONFLICT",
                "not_ground_truth": True,
            },
        )
        json_dump(
            self.out_path("15_运行回执", "R4_DEVELOPMENT_RECEIPT.json"),
            {
                "status": "COMPLETED_PHASE_A_TRACKING_CANDIDATE",
                "ready_for_citywide_candidate_run": "PENDING_INDEPENDENT_ACCEPTANCE",
                "output_root": str(self.out),
                "r3_root": str(self.r3_root),
                "r2_root": str(self.r2),
                "scientific_accuracy": "NOT_EVALUATED",
                "raw_and_r2_inputs_modified": False,
            },
        )

    def run(self) -> None:
        self.assert_inputs()
        self.prepare_output()
        self.load_scenes()
        self.load_frozen_geometry()
        self.run_tracking()
        self.run_gap_return()
        self.run_presence()
        self.run_state()
        self.build_fixed_case_comparison()
        self.write_receipts()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-root", type=Path, required=True)
    parser.add_argument("--r3-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    runner = R4TrackingRunner(args.r2_root, args.r3_root, args.output_root)
    runner.run()
    print(json.dumps({"status": "COMPLETED_PHASE_A_TRACKING_CANDIDATE", "output_root": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
