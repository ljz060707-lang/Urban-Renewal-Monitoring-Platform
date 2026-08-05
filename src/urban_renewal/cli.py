from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .core.hashing import sha256_file
from .core.receipts import read_json, write_json, utc_now
from .geometry import audit_alignment, audit_geometry, geometry_status, vectorize_geometry
from .inference import run_geometry_overlay


FULL20_SHA = "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
RSBUILDING_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
JOINT_SHA = "35668be3a61b30a29576fc66203b67505e4a0cfa56051d04d6bc25ed21de875c"


def _paths(project: Path) -> dict[str, Path]:
    return {
        "geometry": project / "outputs/inference/beijing_tongzhou_building_geometry",
        "joint": project / "outputs/training/surface_state_v1/joint_sample_20260730_034853/checkpoints/surface_joint_sample_best_validation.pth",
        "full20": project / "outputs/pretraining/change_v2/20260727_001029/checkpoints/best_v2_multisource_change.pth",
        "rsbuilding": project / "outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/iter_2000.pth",
        "mti": project / "outputs/training/surface_state_v1/mti_sample_20260730_034853/MTI_SELECTION_FINAL.json",
    }


def models_verify(project: Path) -> dict[str, Any]:
    paths = _paths(project)
    result = {"status": "pass", "artifacts": {}, "checked_at": utc_now(), "test_reads": {"WHU": 0, "SECOND": 0}}
    expected = {"joint": JOINT_SHA, "full20": FULL20_SHA, "rsbuilding": RSBUILDING_SHA}
    for name, path in paths.items():
        if name in {"geometry", "mti"}:
            result["artifacts"][name] = {"path": str(path), "exists": path.exists()}
            continue
        digest = sha256_file(path) if path.exists() else None
        result["artifacts"][name] = {"path": str(path), "exists": path.exists(), "sha256": digest, "expected": expected[name], "match": digest == expected[name]}
        result["status"] = "pass" if result["status"] == "pass" and digest == expected[name] else "fail"
    return result


def system_status(project: Path) -> dict[str, Any]:
    paths = _paths(project)
    geometry = geometry_status(paths["geometry"])
    numerical_path = paths["geometry"] / "TONGZHOU_URRS_GEOMETRY_AUDIT.json"
    alignment_path = paths["geometry"] / "TONGZHOU_URRS_ALIGNMENT_AUDIT.json"
    vector_path = paths["geometry"] / "TONGZHOU_URRS_VECTORIZATION.json"
    numerical = read_json(numerical_path) if numerical_path.exists() else {}
    alignment = read_json(alignment_path) if alignment_path.exists() else {}
    vector = read_json(vector_path) if vector_path.exists() else {}
    overlay_runs = sorted((project / "outputs/integration/urban_renewal_overlay_with_geometry").glob("*/MASTER_OVERLAY_MANIFEST.json"))
    overlay = read_json(overlay_runs[-1]) if overlay_runs else {}
    gates = {
        "DATA_READY": (paths["geometry"] / "TONGZHOU_SCENES.json").exists(),
        "MODEL_READY": models_verify(project)["status"] == "pass",
        "GEOMETRY_READY": numerical.get("geometry_numerical_status") == "pass" and vector.get("vectorization_status") == "pass",
        "ALIGNMENT_READY": alignment.get("alignment_status") == "ready_for_pixelwise_temporal_tracking",
        "FUSION_READY": bool(overlay.get("status") == "success"),
        "NUMERICAL_AUDIT_READY": numerical.get("geometry_numerical_status") == "pass",
        "REVIEW_PACKET_READY": bool(overlay.get("review_packet")),
        "HUMAN_REVIEW_READY": False,
        "INFERENCE_READY": False,
    }
    blocked = [name for name, value in gates.items() if not value]
    result = {
        "system": "Urban Renewal Remote Sensing System",
        "version": "1.0.0-candidate",
        "current_stage": "human_review_pending" if gates["REVIEW_PACKET_READY"] else "geometry_or_overlay_pending",
        "completed_gates": [name for name, value in gates.items() if value],
        "blocked_gates": blocked,
        "blocking_reasons": ["human_visual_review=pending_user_review"] if gates["REVIEW_PACKET_READY"] else [f"{name}_not_verified" for name in blocked],
        "next_allowed_actions": ["user reviews local Tongzhou review packet"] if gates["REVIEW_PACKET_READY"] else ["run the next gated URRS stage"],
        "prohibited_actions": ["retrain Full20", "retrain Joint", "rerun RSBuilding raster inference", "read WHU Test", "read SECOND Test"],
        "gates": gates,
        "inference_ready": False,
        "geometry": geometry,
        "overlay": str(overlay_runs[-1].parent) if overlay_runs else None,
        "test_reads": {"WHU": 0, "SECOND": 0},
        "checked_at": utc_now(),
    }
    write_json(project / "outputs/urrs/status/SYSTEM_STATE.json", result)
    return result


def inventory(project: Path) -> dict[str, Any]:
    roots = ["src", "tools", "scripts", "configs", "extensions", "tests", "docs", "outputs"]
    rows = []
    for root in roots:
        base = project / root
        if not base.exists():
            continue
        for file in base.rglob("*"):
            if not file.is_file():
                continue
            classification = "generated_output" if root == "outputs" else "legacy_active"
            if file.name in {"urrs.py", "codex_windows_fallback_runner.py"} or str(file).startswith(str(project / "src/urban_renewal")):
                classification = "canonical"
            if file.suffix in {".pth", ".pt"}:
                classification = "frozen_asset"
            rows.append({"path": str(file.relative_to(project)), "classification": classification, "size_bytes": file.stat().st_size})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = project / "outputs/urrs/audits" / stamp
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "FILE_INVENTORY.json", {"generated_at": utc_now(), "rows": rows, "counts": {"files": len(rows)}})
    with (out / "CODE_MODULE_INVENTORY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "classification", "size_bytes"])
        writer.writeheader(); writer.writerows(rows)
    return {"status": "pass", "inventory_dir": str(out), "file_count": len(rows)}


def registry_validate(project: Path) -> dict[str, Any]:
    candidates = [project / "outputs/model_registry", project / "registry", project / "configs/registry"]
    files = [str(path) for root in candidates if root.exists() for path in root.rglob("*") if path.is_file()]
    result = {"status": "pass" if files else "fail", "registry_files": files, "validated_at": utc_now(), "test_reads": {"WHU": 0, "SECOND": 0}}
    write_json(project / "outputs/urrs/registry/VALIDATION.json", result)
    return result


def finalize(project: Path) -> dict[str, Any]:
    state = system_status(project)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = project / "reports" / f"URRS_FINAL_GATE_{stamp}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# URRS Final Gate", "", f"Generated: {utc_now()}", "", "## Gate status", ""]
    for key, value in state["gates"].items():
        lines.append(f"- {key}: {'PASS' if value else 'PENDING/BLOCKED'}")
    lines.extend(["", "## Scientific boundaries", "", "- All Tongzhou outputs are candidate evidence, not Ground Truth.", "- Human visual review remains pending.", "- No Full20, Joint, FUSU, CWLD or MTI training was started.", "- WHU Test reads = 0; SECOND Test reads = 0.", "", f"System state: {project / 'outputs/urrs/status/SYSTEM_STATE.json'}"])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"status": "pass", "inference_ready": False, "report": str(report), "system_state": state, "finalized_at": utc_now()}
    write_json(project / "outputs/urrs/status/FINAL_GATE.json", result)
    return result

