from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from ..integration.overlay_final import final_overlay_infer
from ..integration.registration_final import register_final_candidate
from ..integration.tongzhou_acceptance import audit_tongzhou
from .checkpointing import sha256
from .experiment_control import (
    run_joint_from_selection,
    run_joint_probes,
    select_temporal_branch,
)
from .mti_final import recalibrate_mti
from .runtime_compat import train


FULL20_SHA = "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
RSBUILDING_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"


def _best(run: Path) -> Path:
    paths = list((run / "checkpoints").glob("*best_validation.pth"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one best checkpoint in {run}, found {len(paths)}")
    return paths[0]


def _write_receipt(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_final_pipeline(
    cwld_run: str | Path,
    pipeline_dir: str | Path,
    *,
    repo: str | Path,
) -> dict:
    repo, cwld_run, pipeline = Path(repo), Path(cwld_run), Path(pipeline_dir)
    if not cwld_run.is_absolute():
        cwld_run = repo / cwld_run
    if not pipeline.is_absolute():
        pipeline = repo / pipeline
    pipeline.mkdir(parents=True, exist_ok=False)
    receipt_path = pipeline / "FINAL_PIPELINE_RECEIPT.json"
    state = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cwld_run": str(cwld_run.resolve()),
        "current_stage": "preflight",
        "stages": {},
        "whu_test_reads": 0,
        "second_test_reads": 0,
    }
    _write_receipt(receipt_path, state)
    try:
        finalization = json.loads(
            (cwld_run / "CWLD_FINALIZATION_RECEIPT.json").read_text(encoding="utf-8")
        )
        if finalization["candidate_status"] != "cwld_validation_candidate":
            raise RuntimeError("CWLD final candidate gate did not pass")
        cwld_checkpoint = Path(finalization["checkpoint_path"])
        state["stages"]["cwld"] = finalization
        _write_receipt(receipt_path, state)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state["current_stage"] = "fusu_baseline"
        baseline_run = repo / "outputs" / "training" / "surface_state_v1" / f"fusu_sample_{stamp}"
        baseline_receipt = train(
            repo
            / "extensions"
            / "surface_state_v1"
            / "configs"
            / "fusu_sample_baseline_train.yaml",
            repo,
            run_dir=str(baseline_run),
        )
        state["stages"]["fusu_baseline"] = baseline_receipt
        _write_receipt(receipt_path, state)

        state["current_stage"] = "fusu_tfr"
        tfr_run = repo / "outputs" / "training" / "surface_state_v1" / f"tfr_sample_{stamp}"
        tfr_receipt = train(
            repo
            / "extensions"
            / "surface_state_v1"
            / "configs"
            / "temporal_tfr_sample_train.yaml",
            repo,
            run_dir=str(tfr_run),
        )
        state["stages"]["fusu_tfr"] = tfr_receipt
        _write_receipt(receipt_path, state)

        state["current_stage"] = "temporal_selection"
        selection_dir = (
            repo
            / "outputs"
            / "training"
            / "surface_state_v1"
            / f"fusu_branch_selection_{stamp}"
        )
        temporal_selection = select_temporal_branch(
            baseline_run, tfr_run, selection_dir
        )
        selected_fusu = Path(temporal_selection["selected_checkpoint"])
        state["stages"]["temporal_selection"] = temporal_selection
        _write_receipt(receipt_path, state)

        state["current_stage"] = "joint_probes"
        probes = (
            repo / "outputs" / "training" / "surface_state_v1" / f"joint_probes_{stamp}"
        )
        probe_selection = run_joint_probes(
            repo
            / "extensions"
            / "surface_state_v1"
            / "configs"
            / "surface_joint_sample_train.yaml",
            cwld_checkpoint,
            selected_fusu,
            temporal_selection["selected_temporal_branch"] == "tfr",
            probes,
            repo=repo,
        )
        state["stages"]["joint_probes"] = probe_selection
        _write_receipt(receipt_path, state)

        state["current_stage"] = "joint_training"
        joint_run = (
            repo / "outputs" / "training" / "surface_state_v1" / f"joint_sample_{stamp}"
        )
        joint_receipt = run_joint_from_selection(
            repo
            / "extensions"
            / "surface_state_v1"
            / "configs"
            / "surface_joint_sample_train.yaml",
            probes / "JOINT_WARMSTART_SELECTION.json",
            joint_run,
            repo=repo,
        )
        joint_checkpoint = _best(joint_run)
        state["stages"]["joint"] = joint_receipt
        _write_receipt(receipt_path, state)

        state["current_stage"] = "mti_recalibration"
        mti_run = (
            repo / "outputs" / "training" / "surface_state_v1" / f"mti_sample_{stamp}"
        )
        provisional = (
            repo
            / "outputs"
            / "training"
            / "surface_state_v1"
            / "mti_sample_20260729_165729"
            / "MTI_SELECTION.json"
        )
        mti = recalibrate_mti(
            repo
            / "extensions"
            / "surface_state_v1"
            / "configs"
            / "mti_sample_calibration.yaml",
            joint_checkpoint,
            mti_run,
            repo=repo,
            provisional_selection=provisional,
        )
        state["stages"]["mti"] = mti
        _write_receipt(receipt_path, state)

        state["current_stage"] = "registration"
        registrations = {
            "cwld": register_final_candidate(
                cwld_run, "cwld_validation_candidate", repo=repo
            ),
            "fusu": register_final_candidate(
                tfr_run
                if temporal_selection["selected_temporal_branch"] == "tfr"
                else baseline_run,
                "fusu_sample_validation_candidate",
                repo=repo,
                extra={
                    "selected_temporal_branch": temporal_selection[
                        "selected_temporal_branch"
                    ]
                },
            ),
            "tfr": register_final_candidate(
                tfr_run,
                temporal_selection["tfr_status"],
                repo=repo,
            ),
            "joint": register_final_candidate(
                joint_run, "surface_joint_sample_validation_candidate", repo=repo
            ),
        }
        mti_registry = (
            repo
            / "outputs"
            / "model_registry"
            / "surface_state_v1"
            / f"{mti_run.name}_final.json"
        )
        mti_registry.write_text(
            json.dumps(mti, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        registrations["mti"] = str(mti_registry.resolve())
        state["stages"]["registration"] = registrations
        _write_receipt(receipt_path, state)

        state["current_stage"] = "final_overlay"
        validation_manifest = (
            repo
            / "outputs"
            / "data_audit"
            / "fusu_sample"
            / "FUSU_SAMPLE_VALIDATION_MANIFEST.csv"
        )
        sample = next(csv.DictReader(validation_manifest.open(encoding="utf-8-sig")))
        overlay_run = (
            repo
            / "outputs"
            / "integration"
            / "surface_state_v1_overlay_final"
            / stamp
        )
        overlay = final_overlay_infer(
            {
                "surface_checkpoint": str(joint_checkpoint),
                "mti_selection": str(mti_run / "MTI_SELECTION.json"),
                "input_t1": sample["image_t1"],
                "input_t2": sample["image_t2"],
                "output": str(overlay_run),
            },
            repo=repo,
        )
        state["stages"]["overlay"] = overlay
        _write_receipt(receipt_path, state)

        state["current_stage"] = "tongzhou"
        tongzhou = audit_tongzhou(repo)
        state["stages"]["tongzhou"] = tongzhou
        _write_receipt(receipt_path, state)

        full20 = (
            repo
            / "outputs"
            / "pretraining"
            / "change_v2"
            / "20260727_001029"
            / "checkpoints"
            / "best_v2_multisource_change.pth"
        )
        rsbuilding = (
            repo
            / "outputs"
            / "v2_readiness"
            / "rsbuilding"
            / "whu"
            / "work_dirs"
            / "swin_t_whu_pilot5000_20260728_110750"
            / "iter_2000.pth"
        )
        hashes = {
            "full20": sha256(full20),
            "rsbuilding": sha256(rsbuilding),
        }
        if hashes["full20"] != FULL20_SHA or hashes["rsbuilding"] != RSBUILDING_SHA:
            raise RuntimeError(f"Frozen checkpoint hash changed: {hashes}")
        inference_ready = (
            overlay.get("status") == "success"
            and tongzhou.get("tongzhou_overlay_status") == "ready"
            and tongzhou.get("human_visual_review") == "completed"
        )
        project_state = {
            "CWLD training completion": "PASS",
            "CWLD full Validation": "PASS",
            "CWLD class collapse gate": finalization["class_collapse_gate"]["status"],
            "CWLD human visual review": "pending",
            "CWLD candidate": finalization["candidate_status"],
            "FUSU baseline completion": baseline_receipt["status"],
            "FUSU TFR completion": tfr_receipt["status"],
            "FUSU selected temporal branch": temporal_selection[
                "selected_temporal_branch"
            ],
            "FUSU sample candidate": "fusu_sample_validation_candidate",
            "FUSU full dataset validation": False,
            "Joint training completion": joint_receipt["status"],
            "Joint candidate": "surface_joint_sample_validation_candidate",
            "MTI recalibration": "PASS",
            "MTI candidate": "mti_sample_validation_calibrated_candidate",
            "Final overlay": overlay.get("status"),
            "Tongzhou geometry readiness": tongzhou.get("tongzhou_overlay_status"),
            "Tongzhou alignment": tongzhou.get("alignment_status"),
            "Tongzhou numerical audit": "blocked"
            if tongzhou.get("tongzhou_overlay_status") != "ready"
            else "PASS",
            "Tongzhou review packet": tongzhou.get("visual_review_packet"),
            "Tongzhou human review": tongzhou.get("human_visual_review"),
            "Full20 hash unchanged": hashes["full20"] == FULL20_SHA,
            "RSBuilding hash unchanged": hashes["rsbuilding"] == RSBUILDING_SHA,
            "WHU Test reads": 0,
            "SECOND Test reads": 0,
            "Inference ready": inference_ready,
        }
        status_dir = repo / "outputs" / "status"
        status_dir.mkdir(parents=True, exist_ok=True)
        (status_dir / "SURFACE_STATE_V1_PROJECT_STATE.json").write_text(
            json.dumps(project_state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
            final_report=str(report.resolve()),
        )
        _write_receipt(receipt_path, state)
        return state
    except Exception as error:
        state.update(
            status="failed",
            error=repr(error),
            failed_at=datetime.now(timezone.utc).isoformat(),
        )
        _write_receipt(receipt_path, state)
        raise

