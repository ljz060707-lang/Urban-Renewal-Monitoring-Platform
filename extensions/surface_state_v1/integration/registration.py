from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..training.checkpointing import sha256


def register_candidate(run_dir, *, repo):
    run_dir, repo = Path(run_dir), Path(repo)
    receipt = json.loads((run_dir / "TRAINING_RECEIPT.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "success":
        raise RuntimeError("Only a successful run can be registered")
    selection = json.loads((run_dir / "CHECKPOINT_SELECTION.json").read_text(encoding="utf-8"))
    checkpoints = sorted((run_dir / "checkpoints").glob("*best_validation.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one best validation checkpoint, found {len(checkpoints)}")
    checkpoint = checkpoints[0]
    record = {
        "model_id": checkpoint.stem,
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "candidate_status": selection.get("candidate_status"),
        "full_fusu_validated": False,
        "national_generalization_verified": False,
        "inference_ready": False,
        "scientific_status": "candidate_not_ground_truth",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "whu_test_reads": 0,
        "second_test_reads": 0,
    }
    target = repo / "outputs" / "model_registry" / "surface_state_v1"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{run_dir.name}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
