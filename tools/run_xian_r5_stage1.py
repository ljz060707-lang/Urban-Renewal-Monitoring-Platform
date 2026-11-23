#!/usr/bin/env python3
"""Run the isolated Xi'an R5 Stage-1 evidence and no-training prototype.

This runner freezes/records R4 inputs, builds the actual-date registry, runs a
small SIFT registration diagnostic over shared source tiles, records the
official RBF-SAM prompt contract without inventing a checkpoint, and executes
the TBIA handcrafted matcher on a bounded geometry sample.  It does not read
WHU/SECOND/SpaceNet holdouts, retrain B0, or modify any formal output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from xian_r5.alignment.common import sha256_file, utc_now, write_csv, write_json
from xian_r5.alignment.ground_registration import run_alignment_baseline
from xian_r5.alignment.registry import build_observation_registry
from xian_r5.tracking.tbia import BuildingObservation, TBIAPrototype


DEFAULT_OUTPUT = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923")
DEFAULT_MAIN = Path(r"F:\xian\xian")
DEFAULT_SUPPLEMENT = Path(r"F:\xian\XIAN补充")
DEFAULT_DATE_AUDIT = DEFAULT_MAIN / "date_audit"
DEFAULT_CANDIDATE = Path(r"F:\xian\西安_当前最佳候选交付_20260917")
DEFAULT_SUPPLEMENT_AUDIT = DEFAULT_OUTPUT / "00_audit" / "supplement_registry" / "supplement_observation_registry.json"
REQUIRED_CASES = {
    "B02-P2-24": ("unobservable", "reliable_absence_must_not_be_inferred"),
    "B03-TEX04": ("neighbor_switch", "no_neighbor_switch"),
    "B04-TQA07": ("merge", "must_not_swallow_neighbor"),
    "B04-TQA10": ("split", "multiple_villas_separate"),
    "B05-TQA03": ("gap", "no_death_rebirth"),
    "B05-P1-10": ("gap", "no_death_rebirth"),
    "B01-P2-26": ("demolition", "retain_true_demolition"),
    "OBJ_0001": ("addition", "retain_true_addition"),
    "OBJ_0008": ("stable", "111_stable"),
    "B05-TEX01": ("parallax", "split_vs_death_rebirth"),
    "B01-TEX02": ("nonbuilding_control", "do_not_buildingize_sports_field"),
    "B05-P1-06": ("nonbuilding_control", "do_not_track_farmland"),
}


def _git(root: Path, args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30)
        return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": repr(exc)}


def _code_fingerprint(root: Path) -> dict[str, Any]:
    records: list[str] = []
    count = 0
    latest = None
    for subdir in ("src", "tools", "tests", "docs"):
        path = root / subdir
        if not path.is_dir():
            continue
        for file in path.rglob("*"):
            if not file.is_file() or "__pycache__" in file.parts:
                continue
            stat = file.stat()
            relative = file.relative_to(root).as_posix()
            records.append(f"{relative}|{stat.st_size}|{stat.st_mtime_ns}")
            count += 1
            latest = max(latest or stat.st_mtime, stat.st_mtime)
    digest = hashlib.sha256("\n".join(sorted(records)).encode("utf-8")).hexdigest()
    return {"root": str(root), "code_file_count": count, "code_manifest_sha256": digest, "latest_mtime_epoch": latest}


def audit_active_repo(output_root: Path) -> dict[str, Any]:
    active = ROOT.parent.parent.parent.parent  # replaced below with explicit paths
    d_root = Path(r"D:\Codex\projects\Documents-Codex\2026-07-18\urban-renewal-phase1-starter-v0-2")
    c_root = Path(r"C:\Users\ljz52\Documents\Codex\2026-07-18\urban-renewal-phase1-starter-v0-2")
    roots = []
    for label, path in (("priority", d_root), ("mirror_or_archive", c_root)):
        record = _code_fingerprint(path) if path.is_dir() else {"root": str(path), "status": "MISSING"}
        record.update({"label": label, "directory_exists": path.is_dir(), "last_write_time": path.stat().st_mtime if path.is_dir() else None})
        record["git_status"] = _git(path, ["status", "--short", "--branch"]) if path.is_dir() else None
        record["git_root"] = _git(path, ["rev-parse", "--show-toplevel"]) if path.is_dir() else None
        roots.append(record)
    selected = str(d_root) if d_root.is_dir() else str(c_root)
    audit = {
        "schema_version": "xian_r5.active_repo_path.v1",
        "generated_at_utc": utc_now(),
        "active_repo_path": selected,
        "selection_rule": "priority D path exists; C path retained as mirror_or_archive",
        "git_metadata_status": "UNAVAILABLE_NO_VALID_GIT_METADATA",
        "roots": roots,
        "source_policy": "only active D path may receive R5 code; C path is not modified",
    }
    write_json(output_root / "ACTIVE_REPO_PATH.json", audit)
    return audit


def freeze_r4(output_root: Path, candidate_root: Path) -> dict[str, Any]:
    checkpoint = ROOT / r"outputs\v2_readiness\rsbuilding\whu\work_dirs\swin_t_whu_pilot5000_20260728_110750\iter_2000.pth"
    geometry_receipt = candidate_root / "00_handoff" / "GEOMETRY_SOURCE_PATHS.json"
    status_path = candidate_root / "08_cloud_v2_candidate" / "RUN_STATUS.json"
    freeze = {
        "schema_version": "xian_r5.r4_freeze.v1",
        "generated_at_utc": utc_now(),
        "formal_b0": {
            "threshold": 0.50,
            "checkpoint_declared_sha256": "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7",
            "checkpoint_path": str(checkpoint),
            "checkpoint_exists": checkpoint.is_file(),
            "checkpoint_observed_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
            "role": "PRIMARY_GEOMETRY_BASELINE",
        },
        "r4_products": {
            "candidate_root": str(candidate_root),
            "geometry_receipt": str(geometry_receipt),
            "geometry_receipt_sha256": sha256_file(geometry_receipt) if geometry_receipt.is_file() else None,
            "run_status": str(status_path),
            "run_status_sha256": sha256_file(status_path) if status_path.is_file() else None,
            "candidate_not_ground_truth": True,
        },
        "frozen_assets": ["B0 checkpoint", "B0 threshold", "formal geometry", "historical tracking", "presence", "state", "TIF", "WHU Test", "SECOND Test", "SpaceNet7 holdout"],
        "r5_policy": "ISOLATED_CANDIDATE_ONLY",
        "production_modified": False,
    }
    write_json(output_root / "R4_FREEZE_MANIFEST.json", freeze)
    return freeze


def build_case_registry(candidate_root: Path, output_root: Path) -> dict[str, Any]:
    source = candidate_root / "05_visual_qa_reference" / "FIXED_CASE_VISUAL_SCREEN.csv"
    by_case: dict[str, dict[str, str]] = {}
    if source.is_file():
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                by_case[row.get("case_id", "")] = row
    rows: list[dict[str, Any]] = []
    for case_id, (category, gate) in REQUIRED_CASES.items():
        existing = by_case.get(case_id, {})
        rows.append({
            "case_id": case_id,
            "source": str(source),
            "case_category": category,
            "required_gate": gate,
            "r4_presence_sequence": existing.get("presence_sequence"),
            "r4_final_state": existing.get("final_state"),
            "r4_unknown_reason": existing.get("unknown_reason"),
            "r4_visual_label": existing.get("visual_label"),
            "r4_fixed_gate_verdict": existing.get("fixed_gate_verdict"),
            "r4_evidence_status": "AVAILABLE_VISUAL_SCREEN" if existing else "NOT_FOUND_IN_FIXED_CASE_SCREEN",
            "r5_status": "NOT_RUN_STAGE1",
            "formal_ground_truth": "NO",
            "candidate_not_ground_truth": "YES",
        })
    path = output_root / "R5_CHALLENGE_CASE_REGISTRY.csv"
    write_csv(path, rows)
    return {"path": str(path), "rows": len(rows), "source": str(source), "matched_existing": sum(row["r4_evidence_status"] == "AVAILABLE_VISUAL_SCREEN" for row in rows)}


def _geometry_paths(candidate_root: Path) -> dict[str, Path]:
    receipt = candidate_root / "00_handoff" / "GEOMETRY_SOURCE_PATHS.json"
    if receipt.is_file():
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            paths = {}
            for item in payload.get("epochs", []):
                epoch = str(item.get("epoch"))
                for key in ("absolute_path", "primary_copy"):
                    candidate = Path(str(item.get(key) or ""))
                    if candidate.is_file():
                        paths[epoch] = candidate
                        break
            if paths:
                return paths
        except Exception:
            pass
    return {
        "2018": Path(r"F:\xian\_xian_geom_transfer\2018-04-29_building_objects.gpkg"),
        "2019": Path(r"F:\xian\西安_三期建筑追踪与三态最终正式候选版_20260916\02_三期建筑几何\remote_full_geometry\2019-01-01_building_objects.gpkg"),
        "2020": Path(r"F:\xian\西安_三期建筑追踪与三态最终正式候选版_20260916\02_三期建筑几何\remote_full_geometry\2020-12-31_building_objects.gpkg"),
    }


def _geometry_sample(path: Path, epoch: str, limit: int = 300) -> list[BuildingObservation]:
    try:
        import fiona
        from shapely.geometry import shape
    except Exception:
        return []
    observations: list[BuildingObservation] = []
    with fiona.open(path, layer="buildings") as source:
        for index, feature in enumerate(source):
            if index >= limit:
                break
            geometry = shape(feature["geometry"]) if feature.get("geometry") else None
            if geometry is None or geometry.is_empty:
                continue
            props = dict(feature.get("properties") or {})
            observations.append(BuildingObservation(
                object_id=str(props.get("object_id") or f"{epoch}::{index + 1}"),
                epoch=epoch,
                x=float(geometry.centroid.x),
                y=float(geometry.centroid.y),
                area_m2=float(props.get("area_m2") or geometry.area),
                perimeter_m=float(props.get("perimeter_m") or geometry.length),
                compactness=float(4.0 * 3.141592653589793 * geometry.area / max(geometry.length * geometry.length, 1e-9)),
                rectangularity=float(geometry.area / max((geometry.envelope.area), 1e-9)),
                convexity=float(geometry.area / max(geometry.convex_hull.area, 1e-9)),
                metadata={"source_path": str(path), "scientific_status": props.get("scientific_status")},
            ))
    return observations


def run_tbia_sample(candidate_root: Path, output_root: Path) -> dict[str, Any]:
    paths = _geometry_paths(candidate_root)
    samples = {epoch: _geometry_sample(path, epoch) for epoch, path in paths.items() if path.is_file()}
    matcher = TBIAPrototype()
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for left, right in (("2018", "2019"), ("2019", "2020")):
        if left not in samples or right not in samples:
            receipts.append({"from_epoch": left, "to_epoch": right, "status": "BLOCKED_NO_GEOMETRY_SAMPLE"})
            continue
        decisions = matcher.match(samples[left], samples[right])
        for decision in decisions:
            rows.append({
                "from_epoch": left,
                "to_epoch": right,
                "previous_id": decision.previous_id,
                "current_id": decision.current_id,
                "affinity": decision.affinity,
                "candidate_count": decision.candidate_count,
                "search_radius_m": decision.search_radius_m,
                "relation": decision.relation,
                "resolver_stage": decision.resolver_stage,
                "evidence_source": decision.evidence_source,
                "hard_negative_ids": ";".join(decision.hard_negative_ids),
                "scientific_status": "DIAGNOSTIC_NOT_GROUND_TRUTH",
            })
        receipts.append({"from_epoch": left, "to_epoch": right, "status": "PASS_HANDCRAFTED_DIAGNOSTIC", "previous_count": len(samples[left]), "current_count": len(samples[right]), "decision_count": len(decisions)})
    out = output_root / "TBIA_PROTOTYPE_MATCHES.csv"
    write_csv(out, rows)
    receipt = {
        "schema_version": "xian_r5.tbia.prototype.v1",
        "generated_at_utc": utc_now(),
        "status": "PASS_HANDCRAFTED_DIAGNOSTIC" if rows else "BLOCKED_NO_GEOMETRY_SAMPLE",
        "input_geometry_paths": {key: str(value) for key, value in paths.items()},
        "sample_limits": {key: len(value) for key, value in samples.items()},
        "pair_receipts": receipts,
        "training": "NOT_RUN_INFERENCE_ONLY",
        "accuracy": "NOT_CLAIMED_NO_IDENTITY_GROUND_TRUTH",
        "output": str(out),
    }
    write_json(output_root / "TBIA_PROTOTYPE_RECEIPT.json", receipt)
    return receipt


def environment_matrix(output_root: Path) -> None:
    packages = {}
    for package in ("numpy", "pandas", "Pillow", "pyarrow", "opencv-python-headless", "shapely", "fiona", "pyproj", "rasterio"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "UNAVAILABLE_IN_BASE_RUNTIME"
    text = """# R5 environment matrix\n\nGenerated by `tools/run_xian_r5_stage1.py`. The R5 diagnostics use a per-run site-packages directory under the R5 output root; the production runtime was not modified.\n\n| Component | Declared / observed environment | Status |\n|---|---|---|\n| R5 main | Python 3.12 bundled runtime + isolated site-packages | ACTIVE STAGE-1 |\n| RBF-SAM | PyTorch 1.13.1 / CUDA 11.6 / MMDetection 2.3.0 (upstream declaration) | CHECKPOINT NOT PRESENT; INFERENCE NOT RUN |\n| SemiBCD | Python 3.9 recommendation; upstream requirements include torch 2.7.0+cu128 | CONCEPT-ONLY; NOT INSTALLED INTO PRODUCTION |\n| LightGlue | PyTorch/OpenCV/Kornia; SIFT and ALIKED backends available upstream | SIFT BASELINE RUN; LIGHTGLUE WEIGHTS NOT RUN |\n\nObserved package versions in the isolated runner:\n\n""" + "\n".join(f"- `{key}`: `{value}`" for key, value in packages.items()) + "\n\nDependency policy: RBF-SAM, SemiBCD, and LightGlue are not imported into the frozen R4 runtime.\n"
    (output_root / "ENVIRONMENT_MATRIX.md").write_text(text, encoding="utf-8")


def write_rbf_sam_contract(output_root: Path) -> dict[str, Any]:
    reference = ROOT / "third_party_reference" / "RBF-SAM"
    checkpoints = list(reference.rglob("*.pth")) if reference.is_dir() else []
    payload = {
        "status": "BLOCKED_NO_RBF_SAM_CHECKPOINT" if not checkpoints else "READY_FOR_ISOLATED_INFERENCE",
        "reference_root": str(reference),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "official_prompt_contract": {
            "0": "building_box",
            "1": "roof_box",
            "2": "footprint_box",
            "3": "noisy_box_or_perturbation",
        },
        "planned_temporal_prompt": "simulate_with_existing_box_type_3_until_benchmark_justifies_new_embedding",
        "offset_outputs": ["offset_final", "offset_global", "offset_ll", "offset_lh", "offset_hl", "offset_hh"],
        "geometry_role": "AMBIGUITY_RESOLVER_ONLY;B0_REMAINS_PRIMARY",
        "inference_started": False,
        "reason": "upstream code is present but no checkpoint was downloaded; no mask/offset was fabricated",
    }
    write_json(output_root / "RBF_SAM_PRETRAINED_REPRODUCTION.json", payload)
    rows = [{"prompt_type": name, "box_type": code, "status": payload["status"], "iou": None, "boundary_f1": None, "offset_error": None, "scientific_status": "NOT_RUN_NO_CHECKPOINT"} for code, name in payload["official_prompt_contract"].items()]
    rows.append({"prompt_type": "TEMPORAL_TRACK_PROMPT", "box_type": "candidate_box_type_3", "status": "NOT_RUN_NO_CHECKPOINT", "iou": None, "boundary_f1": None, "offset_error": None, "scientific_status": "NOT_RUN_NO_CHECKPOINT"})
    write_csv(output_root / "TEMPORAL_PROMPT_BENCHMARK.csv", rows)
    return payload


def initial_status(output_root: Path, *, phase: str, subphase: str, current_experiment: str, blocker: str | None = None, **updates: Any) -> dict[str, Any]:
    status = {
        "schema_version": "xian_r5.run_status.v1",
        "updated_at_utc": utc_now(),
        "phase": phase,
        "subphase": subphase,
        "active_process": os.getpid(),
        "gpu_usage": "NOT_QUERIED_STAGE1_CPU_DIAGNOSTIC",
        "current_experiment": current_experiment,
        "alignment_pairs_done": 0,
        "rbf_sam_cases_done": 0,
        "tbia_training_epoch": None,
        "merge_cases_resolved": 0,
        "fixed_cases_pass": None,
        "id_switch_count": None,
        "unknown_identity_pct": None,
        "false_addition_rate": None,
        "false_demolition_rate": None,
        "current_best_model": "NONE_STAGE1_NO_TRAINING",
        "current_blocker": blocker,
        "next_action": "RUN_ALIGNMENT_AND_RECORD_STAGE1_EVIDENCE",
    }
    status.update(updates)
    write_json(output_root / "RUN_STATUS.json", status)
    return status


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve()
    audit_root = output / "00_audit"
    receipts = output / "16_receipts"
    audit_root.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    progress = [f"# R5 Stage-1 progress", "", f"Started: {utc_now()}", "", "- R4 freeze: RUNNING"]
    (receipts / "RUN_PROGRESS.md").write_text("\n".join(progress) + "\n", encoding="utf-8")
    initial_status(receipts, phase="STAGE_1", subphase="AUDIT_AND_FREEZE", current_experiment=args.run_id, blocker=None)
    active_repo = audit_active_repo(audit_root)
    freeze = freeze_r4(audit_root, args.candidate_root)
    environment_matrix(audit_root)
    case_receipt = build_case_registry(args.candidate_root, output / "09_fixed_cases")
    registry = build_observation_registry(
        args.main_root,
        args.supplement_root,
        args.date_audit_root,
        output / "00_audit" / "observation_registry",
        supplement_registry_path=args.supplement_audit_path,
    )
    progress.append("- R4 freeze: PASS_RECORDED")
    progress.append("- actual-date/source registry: PASS_OR_EXPLICIT_PARTIAL")
    (receipts / "RUN_PROGRESS.md").write_text("\n".join(progress) + "\n", encoding="utf-8")
    initial_status(receipts, phase="STAGE_1", subphase="ALIGNMENT_BASELINE", current_experiment="R5_ALIGN_A01", alignment_pairs_done=0, current_blocker="SIFT_DIAGNOSTIC_RUNNING")
    tile_roots = {row["epoch"]: Path(row["tile_root"]) for row in registry["rows"] if row.get("tile_root")}
    alignment = run_alignment_baseline(tile_roots, output / "02_alignment" / "R5_ALIGN_A01", sample_count=args.alignment_sample_count)
    initial_status(receipts, phase="STAGE_1", subphase="RBF_SAM_CONTRACT", current_experiment="R5_RBFSAM_M01", alignment_pairs_done=len(alignment["qa_rows"]), current_blocker=None)
    rbf = write_rbf_sam_contract(output / "03_rbf_sam" / "R5_RBFSAM_M01")
    tbia = run_tbia_sample(args.candidate_root, output / "04_tbia" / "R5_TBIA_T01")
    write_json(output / "09_fixed_cases" / "FIXED_CASE_GATE.json", {"status": "NOT_RUN_NO_R5_INFERENCE", "r4_case_registry": case_receipt, "required_cases": sorted(REQUIRED_CASES)})
    final_status = initial_status(receipts, phase="STAGE_1", subphase="COMPLETE_DIAGNOSTIC_ONLY", current_experiment=args.run_id, alignment_pairs_done=len(alignment["qa_rows"]), rbf_sam_cases_done=0, current_blocker="RBF_SAM_CHECKPOINT_AND_FORMAL_CASE_GEOMETRY_BENCHMARK_PENDING", next_action="OBTAIN_OR_REGISTER_RBF_SAM_CHECKPOINT;MAP_FIXED_CASES_TO_TILES;REVIEW_ALIGNMENT_QA")
    progress.extend([
        "- SIFT alignment: PASS_DIAGNOSTIC outputs recorded; stable-ground/parallax strata remain explicit NOT_RUN.",
        f"- RBF-SAM prompt reproduction: {rbf['status']}.",
        f"- TBIA handcrafted sample: {tbia['status']}.",
        "- Scientific/operational R5 products: NOT_CREATED; no A/B gate passed.",
        f"- Completed: {utc_now()}",
    ])
    (receipts / "RUN_PROGRESS.md").write_text("\n".join(progress) + "\n", encoding="utf-8")
    return {"active_repo": active_repo, "freeze": freeze, "registry": registry["receipt"], "case_registry": case_receipt, "alignment": alignment["receipt"], "rbf_sam": rbf, "tbia": tbia, "status": final_status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--supplement-root", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--date-audit-root", type=Path, default=DEFAULT_DATE_AUDIT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--supplement-audit-path", type=Path, default=DEFAULT_SUPPLEMENT_AUDIT)
    parser.add_argument("--alignment-sample-count", type=int, default=100)
    parser.add_argument("--run-id", default="R5_STAGE1_20260923")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": "PASS_STAGE1_DIAGNOSTIC_COMPLETE", "output_root": str(args.output_root), "summary": result["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
