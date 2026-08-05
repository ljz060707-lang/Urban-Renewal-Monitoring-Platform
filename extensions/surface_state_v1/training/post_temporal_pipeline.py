"""Resume the formal pipeline after completed FUSU baseline/TFR validation."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from ..integration.overlay_final import final_overlay_infer
from ..integration.registration_final import register_final_candidate
from ..integration.tongzhou_acceptance import audit_tongzhou
from .checkpointing import sha256
from .experiment_control import run_joint_from_selection, run_joint_probes
from .final_pipeline import FULL20_SHA, RSBUILDING_SHA, _best
from .mti_final import recalibrate_mti


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_post_temporal_pipeline(
    *,
    repo: str | Path,
    cwld_run: str | Path,
    baseline_run: str | Path,
    tfr_run: str | Path,
    selection_dir: str | Path,
    pipeline_dir: str | Path,
) -> dict:
    repo = Path(repo)
    paths = {
        key: (Path(value) if Path(value).is_absolute() else repo / value)
        for key, value in {
            "cwld": cwld_run,
            "baseline": baseline_run,
            "tfr": tfr_run,
            "selection": selection_dir,
            "pipeline": pipeline_dir,
        }.items()
    }
    pipeline = paths["pipeline"]
    pipeline.mkdir(parents=True, exist_ok=False)
    receipt_path = pipeline / "POST_TEMPORAL_PIPELINE_RECEIPT.json"
    state = {
        "status": "running",
        "current_stage": "preflight",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_runs": {k: str(v.resolve()) for k, v in paths.items() if k != "pipeline"},
        "stages": {},
        "whu_test_reads": 0,
        "second_test_reads": 0,
    }
    _write(receipt_path, state)
    try:
        finalization = _read(paths["cwld"] / "CWLD_FINALIZATION_RECEIPT.json")
        baseline = _read(paths["baseline"] / "TRAINING_RECEIPT.json")
        tfr = _read(paths["tfr"] / "TRAINING_RECEIPT.json")
        selection = _read(paths["selection"] / "FUSU_TEMPORAL_BRANCH_SELECTION.json")
        if finalization["candidate_status"] != "cwld_validation_candidate":
            raise RuntimeError("CWLD candidate gate did not pass")
        if baseline["status"] != "success" or tfr["status"] != "success":
            raise RuntimeError("FUSU baseline/TFR source run is not successful")
        if any(int(x.get("test_read_count", 0)) for x in (baseline, tfr)):
            raise RuntimeError("A source run reports Test reads")
        cwld_checkpoint = Path(finalization["checkpoint_path"])
        fusu_checkpoint = Path(selection["selected_checkpoint"])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        state["current_stage"] = "joint_probes"
        probes = repo / "outputs/training/surface_state_v1" / f"joint_probes_{stamp}"
        probe_result = run_joint_probes(
            repo / "extensions/surface_state_v1/configs/surface_joint_sample_train.yaml",
            cwld_checkpoint,
            fusu_checkpoint,
            selection["selected_temporal_branch"] == "tfr",
            probes,
            repo=repo,
        )
        state["stages"]["joint_probes"] = probe_result
        _write(receipt_path, state)

        state["current_stage"] = "joint_training"
        joint_run = repo / "outputs/training/surface_state_v1" / f"joint_sample_{stamp}"
        joint = run_joint_from_selection(
            repo / "extensions/surface_state_v1/configs/surface_joint_sample_train.yaml",
            probes / "JOINT_WARMSTART_SELECTION.json",
            joint_run,
            repo=repo,
        )
        joint_checkpoint = _best(joint_run)
        state["stages"]["joint"] = joint
        _write(receipt_path, state)

        state["current_stage"] = "mti_recalibration"
        mti_run = repo / "outputs/training/surface_state_v1" / f"mti_sample_{stamp}"
        mti = recalibrate_mti(
            repo / "extensions/surface_state_v1/configs/mti_sample_calibration.yaml",
            joint_checkpoint,
            mti_run,
            repo=repo,
            provisional_selection=repo
            / "outputs/training/surface_state_v1/mti_sample_20260729_165729/MTI_SELECTION.json",
        )
        state["stages"]["mti"] = mti
        _write(receipt_path, state)

        state["current_stage"] = "registration"
        registrations = {
            "cwld": register_final_candidate(paths["cwld"], "cwld_validation_candidate", repo=repo),
            "fusu": register_final_candidate(
                paths["tfr"] if selection["selected_temporal_branch"] == "tfr" else paths["baseline"],
                "fusu_sample_validation_candidate",
                repo=repo,
                extra={"selected_temporal_branch": selection["selected_temporal_branch"]},
            ),
            "tfr": register_final_candidate(paths["tfr"], selection["tfr_status"], repo=repo),
            "joint": register_final_candidate(
                joint_run, "surface_joint_sample_validation_candidate", repo=repo
            ),
        }
        mti_registry = (
            repo / "outputs/model_registry/surface_state_v1" / f"{mti_run.name}_final.json"
        )
        mti_registry.parent.mkdir(parents=True, exist_ok=True)
        _write(mti_registry, mti)
        registrations["mti"] = str(mti_registry.resolve())
        state["stages"]["registration"] = registrations
        _write(receipt_path, state)

        state["current_stage"] = "final_overlay"
        manifest = repo / "outputs/data_audit/fusu_sample/FUSU_SAMPLE_VALIDATION_MANIFEST.csv"
        sample = next(csv.DictReader(manifest.open(encoding="utf-8-sig")))
        overlay = final_overlay_infer(
            {
                "surface_checkpoint": str(joint_checkpoint),
                "mti_selection": str(mti_run / "MTI_SELECTION.json"),
                "input_t1": sample["image_t1"],
                "input_t2": sample["image_t2"],
                "output": str(repo / "outputs/integration/surface_state_v1_overlay_final" / stamp),
            },
            repo=repo,
        )
        state["stages"]["overlay"] = overlay
        _write(receipt_path, state)

        state["current_stage"] = "tongzhou"
        tongzhou = audit_tongzhou(repo)
        state["stages"]["tongzhou"] = tongzhou

        full20 = repo / "outputs/pretraining/change_v2/20260727_001029/checkpoints/best_v2_multisource_change.pth"
        rsbuilding = repo / "outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/iter_2000.pth"
        hashes = {"full20": sha256(full20), "rsbuilding": sha256(rsbuilding)}
        if hashes != {"full20": FULL20_SHA, "rsbuilding": RSBUILDING_SHA}:
            raise RuntimeError(f"Frozen checkpoint hash changed: {hashes}")
        project_state = {
            "CWLD candidate": finalization["candidate_status"],
            "FUSU baseline completion": baseline["status"],
            "FUSU TFR completion": tfr["status"],
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
            "# Surface State V1 Final Gate\n\n"
            + "\n".join(f"- {key}: `{value}`" for key, value in project_state.items())
            + "\n",
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
        state.update(
            status="failed",
            error=repr(error),
            failed_at=datetime.now(timezone.utc).isoformat(),
        )
        _write(receipt_path, state)
        raise

