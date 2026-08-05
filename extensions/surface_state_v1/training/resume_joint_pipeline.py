"""Resume an interrupted formal Joint run, then finish downstream validation stages."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from ..integration.overlay_final import final_overlay_infer
from ..integration.registration_final import register_final_candidate
from ..integration.tongzhou_acceptance import audit_tongzhou
from .checkpointing import sha256
from .final_pipeline import FULL20_SHA, RSBUILDING_SHA, _best
from .mti_final import recalibrate_mti
from .runtime_compat import train


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def resume_joint_and_finish(*, repo: str | Path, joint_run: str | Path, pipeline_dir: str | Path) -> dict:
    repo = Path(repo)
    joint_run = Path(joint_run)
    if not joint_run.is_absolute():
        joint_run = repo / joint_run
    pipeline = Path(pipeline_dir)
    if not pipeline.is_absolute():
        pipeline = repo / pipeline
    receipt_path = pipeline / "POST_TEMPORAL_PIPELINE_RECEIPT.json"
    state = _read(receipt_path)
    stamp = joint_run.name.removeprefix("joint_sample_")
    try:
        state["status"] = "running"
        state["current_stage"] = "joint_resume"
        state["recovery"] = {
            "resumed_at": datetime.now(timezone.utc).isoformat(),
            "joint_run": str(joint_run.resolve()),
            "reason": "external_process_exit_without_python_exception",
        }
        _write(receipt_path, state)

        joint = train(joint_run / "joint_config.yaml", repo, run_dir=str(joint_run))
        if joint.get("status") != "success" or int(joint.get("test_read_count", 0)) != 0:
            raise RuntimeError(f"Joint resume did not finish safely: {joint}")
        joint_checkpoint = _best(joint_run)
        state["stages"]["joint"] = joint
        state["current_stage"] = "mti_recalibration"
        _write(receipt_path, state)

        mti_run = repo / "outputs/training/surface_state_v1" / f"mti_sample_{stamp}"
        if mti_run.exists():
            raise FileExistsError(f"Refusing to overwrite MTI run: {mti_run}")
        mti = recalibrate_mti(
            repo / "extensions/surface_state_v1/configs/mti_sample_calibration.yaml",
            joint_checkpoint,
            mti_run,
            repo=repo,
            provisional_selection=repo / "outputs/training/surface_state_v1/mti_sample_20260729_165729/MTI_SELECTION.json",
        )
        state["stages"]["mti"] = mti
        state["current_stage"] = "registration"
        _write(receipt_path, state)

        source = state["source_runs"]
        selection = _read(Path(source["selection"]) / "FUSU_TEMPORAL_BRANCH_SELECTION.json")
        registrations = {
            "cwld": register_final_candidate(Path(source["cwld"]), "cwld_validation_candidate", repo=repo),
            "fusu": register_final_candidate(
                Path(source["tfr"] if selection["selected_temporal_branch"] == "tfr" else source["baseline"]),
                "fusu_sample_validation_candidate",
                repo=repo,
                extra={"selected_temporal_branch": selection["selected_temporal_branch"]},
            ),
            "tfr": register_final_candidate(Path(source["tfr"]), selection["tfr_status"], repo=repo),
            "joint": register_final_candidate(joint_run, "surface_joint_sample_validation_candidate", repo=repo),
        }
        mti_registry = repo / "outputs/model_registry/surface_state_v1" / f"{mti_run.name}_final.json"
        mti_registry.parent.mkdir(parents=True, exist_ok=True)
        _write(mti_registry, mti)
        registrations["mti"] = str(mti_registry.resolve())
        state["stages"]["registration"] = registrations
        state["current_stage"] = "final_overlay"
        _write(receipt_path, state)

        manifest = repo / "outputs/data_audit/fusu_sample/FUSU_SAMPLE_VALIDATION_MANIFEST.csv"
        sample = next(csv.DictReader(manifest.open(encoding="utf-8-sig")))
        overlay_output = repo / "outputs/integration/surface_state_v1_overlay_final" / stamp
        if overlay_output.exists():
            raise FileExistsError(f"Refusing to overwrite overlay run: {overlay_output}")
        overlay = final_overlay_infer(
            {
                "surface_checkpoint": str(joint_checkpoint),
                "mti_selection": str(mti_run / "MTI_SELECTION.json"),
                "input_t1": sample["image_t1"],
                "input_t2": sample["image_t2"],
                "output": str(overlay_output),
            },
            repo=repo,
        )
        state["stages"]["overlay"] = overlay
        state["current_stage"] = "tongzhou"
        _write(receipt_path, state)

        tongzhou = audit_tongzhou(repo)
        state["stages"]["tongzhou"] = tongzhou
        full20 = repo / "outputs/pretraining/change_v2/20260727_001029/checkpoints/best_v2_multisource_change.pth"
        rsbuilding = repo / "outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/iter_2000.pth"
        hashes = {"full20": sha256(full20), "rsbuilding": sha256(rsbuilding)}
        if hashes != {"full20": FULL20_SHA, "rsbuilding": RSBUILDING_SHA}:
            raise RuntimeError(f"Frozen checkpoint hash changed: {hashes}")
        project_state = {
            "CWLD candidate": "cwld_validation_candidate",
            "FUSU baseline completion": _read(Path(source["baseline"]) / "TRAINING_RECEIPT.json")["status"],
            "FUSU TFR completion": _read(Path(source["tfr"]) / "TRAINING_RECEIPT.json")["status"],
            "FUSU selected temporal branch": selection["selected_temporal_branch"],
            "FUSU full dataset validation": False,
            "Joint training completion": joint["status"],
            "Joint candidate": "surface_joint_sample_validation_candidate",
            "MTI candidate": "mti_sample_validation_calibrated_candidate",
            "Final overlay": overlay.get("status"),
            "Tongzhou geometry readiness": tongzhou.get("tongzhou_overlay_status"),
            "Tongzhou alignment": tongzhou.get("alignment_status"),
            "Tongzhou human review": tongzhou.get("human_visual_review"),
            "Full20 hash unchanged": True,
            "RSBuilding hash unchanged": True,
            "WHU Test reads": 0,
            "SECOND Test reads": 0,
            "Inference ready": False,
        }
        status_dir = repo / "outputs/status"
        status_dir.mkdir(parents=True, exist_ok=True)
        _write(status_dir / "SURFACE_STATE_V1_PROJECT_STATE.json", project_state)
        report = repo / "reports" / f"SURFACE_STATE_V1_FINAL_GATE_{stamp}.md"
        report.write_text(
            "# Surface State V1 Final Gate\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in project_state.items()) + "\n",
            encoding="utf-8",
        )
        state.update(
            status="success",
            current_stage="complete",
            completed_at=datetime.now(timezone.utc).isoformat(),
            project_state=project_state,
            frozen_hashes=hashes,
            final_report=str(report.resolve()),
        )
        _write(receipt_path, state)
        return state
    except Exception as error:
        state.update(status="failed", error=repr(error), failed_at=datetime.now(timezone.utc).isoformat())
        _write(receipt_path, state)
        raise