from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..training.checkpointing import sha256


def register_final_candidate(
    run_dir: str | Path,
    candidate_status: str,
    *,
    repo: str | Path,
    extra: dict | None = None,
    registry_key: str | None = None,
) -> dict:
    repo, run = Path(repo), Path(run_dir)
    if not run.is_absolute():
        run = repo / run
    receipt = json.loads((run / "TRAINING_RECEIPT.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "success":
        raise RuntimeError(f"Cannot register incomplete run: {run}")
    checkpoints = list((run / "checkpoints").glob("*best_validation.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one formal best checkpoint, found {len(checkpoints)}")
    checkpoint = checkpoints[0]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config_snapshot"]
    if int(config.get("epochs", 0)) < 10:
        raise RuntimeError("Smoke/mini checkpoint cannot be registered as a formal candidate")
    record = {
        "model_id": checkpoint.stem,
        "candidate_status": candidate_status,
        "run_dir": str(run.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "config": config,
        "manifest_hashes": state.get("manifest_hashes", {}),
        "split_scope": state.get("split_scope"),
        "validation_metrics": state.get("validation_metrics"),
        "cwld_class_names": state.get("cwld_class_names"),
        "fusu_class_names": state.get("fusu_class_names"),
        "ontology_mapping": state.get("ontology_mapping"),
        "software_versions": state.get(
            "software_versions",
            {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()},
        ),
        "git_status": state.get("git_status"),
        "scientific_status": "candidate_not_ground_truth",
        "full_fusu_validated": False,
        "national_generalization_verified": False,
        "inference_ready": False,
        "whu_test_reads": 0,
        "second_test_reads": 0,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    target = repo / "outputs" / "model_registry" / "surface_state_v1"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{registry_key or run.name}_final.json"
    if path.exists():
        raise FileExistsError(f"Final registration already exists: {path}")
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record

