"""Complete registration, final overlay, and Tongzhou audit after Joint/MTI."""
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
from .mti_selection_guard import finalize_mti_selection


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _register_or_reuse(repo: Path, run: Path, status: str, *, key: str | None = None, extra=None) -> dict:
    target = repo / "outputs/model_registry/surface_state_v1" / f"{key or run.name}_final.json"
    checkpoints = list((run / "checkpoints").glob("*best_validation.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected exactly one best checkpoint in {run}")
    expected_sha = sha256(checkpoints[0])
    if target.exists():
        record = _read(target)
        if record.get("candidate_status") != status or record.get("checkpoint_sha256") != expected_sha:
            raise RuntimeError(f"Existing registry record conflicts with requested role: {target}")
        return record
    return register_final_candidate(run, status, repo=repo, extra=extra, registry_key=key)


def complete_post_joint(*, repo: str | Path, joint_run: str | Path, pipeline_dir: str | Path) -> dict:
    repo = Path(repo)
    joint_run = repo / joint_run if not Path(joint_run).is_absolute() else Path(joint_run)
    pipeline = repo / pipeline_dir if not Path(pipeline_dir).is_absolute() else Path(pipeline_dir)
    receipt_path = pipeline / "POST_TEMPORAL_PIPELINE_RECEIPT.json"
    state = _read(receipt_path)
    stamp = joint_run.name.removeprefix("joint_sample_")
    state.update(status="running", current_stage="registration_recovery", error=None)
    _write(receipt_path, state)
    try:
        joint = _read(joint_run / "TRAINING_RECEIPT.json")
        if joint.get("status") != "success" or int(joint.get("test_read_count", 0)) != 0:
            raise RuntimeError("Joint run is not a successful Test-free formal run")
        joint_checkpoint = _best(joint_run)
        mti_run = repo / "outputs/training/surface_state_v1" / f"mti_sample_{stamp}"
        source_mti = mti_run / "MTI_SELECTION.json"
        if not source_mti.exists():
            raise FileNotFoundError(source_mti)
        final_mti_path = mti_run / "MTI_SELECTION_FINAL.json"
        final_mti = finalize_mti_selection(source_mti, final_mti_path)
        state["stages"]["mti_final_selection"] = final_mti

        source = state["source_runs"]
        selection = _read(Path(source["selection"]) / "FUSU_TEMPORAL_BRANCH_SELECTION.json")
        selected_fusu_run = Path(source["tfr"] if selection["selected_temporal_branch"] == "tfr" else source["baseline"])
        registrations = {
            "cwld": _register_or_reuse(repo, Path(source["cwld"]), "cwld_validation_candidate"),
            "fusu": _register_or_reuse(
                repo, selected_fusu_run, "fusu_sample_validation_candidate",
                extra={"selected_temporal_branch": selection["selected_temporal_branch"]},
            ),
            "tfr": _register_or_reuse(
                repo, Path(source["tfr"]), selection["tfr_status"],
                key=f"{Path(source['tfr']).name}_tfr",
            ),
            "joint": _register_or_reuse(repo, joint_run, "surface_joint_sample_validation_candidate"),
        }
        mti_registry = repo / "outputs/model_registry/surface_state_v1" / f"{mti_run.name}_final.json"
        if mti_registry.exists():
            existing = _read(mti_registry)
            if existing != final_mti:
                raise RuntimeError(f"Existing MTI registry conflicts: {mti_registry}")
        else:
            _write(mti_registry, final_mti)
        registrations["mti"] = str(mti_registry.resolve())
        state["stages"]["registration"] = registrations
        state["current_stage"] = "final_overlay"
        _write(receipt_path, state)

        manifest = repo / "outputs/data_audit/fusu_sample/FUSU_SAMPLE_VALIDATION_MANIFEST.csv"
        sample = next(csv.DictReader(manifest.open(encoding="utf-8-sig")))
        overlay_output = repo / "outputs/integration/surface_state_v1_overlay_final" / stamp
        if overlay_output.exists():
            overlay = _read(overlay_output / "MASTER_OVERLAY_MANIFEST.json")
            audit = _read(overlay_output / "CHECKPOINT_LOAD_AUDIT.json")
            if audit.get("checkpoint_sha256") != sha256(joint_checkpoint):
                raise RuntimeError("Existing overlay loaded a different Joint checkpoint")
        else:
            overlay = final_overlay_infer(
                {
                    "surface_checkpoint": str(joint_checkpoint),
                    "mti_selection": str(final_mti_path),
                    "input_t1": sample["image_t1"],
                    "input_t2": sample["image_t2"],
                    "output": str(overlay_output),
                }, repo=repo,
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
            "MTI candidate": final_mti["mti_status"],
            "MTI integration mode": final_mti["integration_mode"],
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
        report.write_text("# Surface State V1 Final Gate\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in project_state.items()) + "\n", encoding="utf-8")
        state.update(status="success", current_stage="complete", completed_at=datetime.now(timezone.utc).isoformat(), project_state=project_state, frozen_hashes=hashes, final_report=str(report.resolve()), whu_test_reads=0, second_test_reads=0)
        _write(receipt_path, state)
        return state
    except Exception as error:
        state.update(status="failed", error=repr(error), failed_at=datetime.now(timezone.utc).isoformat())
        _write(receipt_path, state)
        raise