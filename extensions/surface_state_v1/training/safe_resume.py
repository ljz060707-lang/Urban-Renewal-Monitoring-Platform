from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from . import full_trainer
from .checkpointing import select_better, sha256


def resume_cwld(config_path: str | Path, run_dir: str | Path, *, repo: str | Path) -> dict:
    """Resume an interrupted CWLD run without allowing a worse first epoch to replace best."""
    repo, run = Path(repo), Path(run_dir)
    if not run.is_absolute():
        run = repo / run
    last = run / "checkpoints" / "last.pth"
    best = run / "checkpoints" / "cwld_best_validation.pth"
    if not last.exists() or not best.exists():
        raise FileNotFoundError("Safe resume requires both last.pth and cwld_best_validation.pth")
    prior = torch.load(best, map_location="cpu", weights_only=False)
    prior_candidate = {
        "selection_score": float(prior["validation_metrics"]["cwld"]["macro_mIoU"]),
        "macro_F1": float(prior["validation_metrics"]["cwld"]["macro_F1"]),
        "epoch": int(prior["epoch"]),
    }
    guarded = run / "checkpoints" / f"resume_guard_best_epoch_{prior['epoch']:03d}.pth"
    if not guarded.exists():
        shutil.copy2(best, guarded)
    original_select = full_trainer.select_better

    def compare_with_history(candidate, current, mode):
        return select_better(candidate, prior_candidate if current is None else current, mode)

    full_trainer.select_better = compare_with_history
    try:
        receipt = full_trainer.train(config_path, repo, run_dir=str(run))
    finally:
        full_trainer.select_better = original_select

    current = torch.load(best, map_location="cpu", weights_only=False)
    current_candidate = {
        "selection_score": float(current["validation_metrics"]["cwld"]["macro_mIoU"]),
        "macro_F1": float(current["validation_metrics"]["cwld"]["macro_F1"]),
        "epoch": int(current["epoch"]),
    }
    restored = False
    if not select_better(current_candidate, prior_candidate, "cwld"):
        shutil.copy2(guarded, best)
        restored = True
        selected = prior_candidate
    else:
        selected = current_candidate
    receipt_path = run / "TRAINING_RECEIPT.json"
    final_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    final_receipt["best"] = selected
    final_receipt["resume_guard"] = {
        "prior_best": prior_candidate,
        "guarded_checkpoint": str(guarded.resolve()),
        "guarded_checkpoint_sha256": sha256(guarded),
        "historical_best_restored": restored,
        "selected_best": selected,
    }
    receipt_path.write_text(
        json.dumps(final_receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return final_receipt

