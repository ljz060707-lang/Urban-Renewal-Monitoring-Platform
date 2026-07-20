from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest


def _save(path: Path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def _index(tmp_path: Path) -> Path:
    rows = []
    for split in ("train", "val"):
        for number in range(1):
            image = np.full((64, 64, 3), 40 + number, dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8); mask[16:40, 16:40] = 1
            base = tmp_path / split / str(number)
            paths = {}
            for name, value in (("t1", image), ("t2", image + 10), ("cd", mask), ("new", mask), ("dem", np.zeros_like(mask)), ("gt1", np.zeros_like(mask)), ("gt2", mask)):
                path = base.with_name(base.name + f"_{name}.png"); _save(path, value); paths[name] = path
            rows.append({"sample_id": f"second:{split}:{number}", "dataset_name": "second", "split": split,
                         "t1_path": paths["t1"], "t2_path": paths["t2"], "gt_cd_path": paths["cd"],
                         "gt_new_path": paths["new"], "gt_dem_path": paths["dem"], "gt_t1_path": paths["gt1"], "gt_t2_path": paths["gt2"],
                         "has_binary": True, "has_directional": True, "has_semantic": True})
    path = tmp_path / "index.csv"; pd.DataFrame(rows).to_csv(path, index=False); return path


def test_formal_training_rejects_cpu():
    pytest.importorskip("torch")
    from urban_monitor.change.training import resolve_device
    with pytest.raises(RuntimeError, match="CPU is allowed only"):
        resolve_device("cpu", allow_cpu=False)


def test_stage_selection_matches_required_public_datasets(tmp_path: Path):
    from urban_monitor.change.training import stage_frame
    frame = pd.DataFrame([
        {"dataset_name": "s2looking", "has_binary": True, "has_directional": True, "has_semantic": False},
        {"dataset_name": "second", "has_binary": True, "has_directional": True, "has_semantic": True},
        {"dataset_name": "dsifn", "has_binary": True, "has_directional": False, "has_semantic": False},
    ])
    assert set(stage_frame(frame, "binary")["dataset_name"]) == {"s2looking", "second", "dsifn"}
    assert stage_frame(frame, "semantic")["dataset_name"].tolist() == ["second"]
    assert set(stage_frame(frame, "directional")["dataset_name"]) == {"s2looking", "second"}


def test_local_finetune_schedule_freezes_then_unfreezes_encoder():
    torch = pytest.importorskip("torch"); pytest.importorskip("torchvision")
    from urban_monitor.change.models import build_change_model
    from urban_monitor.change.training import apply_local_finetune_schedule
    model = build_change_model({"pretrained": False, "semantic_classes": 2})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = {"local_finetune": {"freeze_encoder_epochs": 1, "partial_unfreeze_epochs": 1,
                                  "learning_rate_head": 2e-4, "learning_rate_partial": 5e-5, "learning_rate_full": 1e-5}}
    assert apply_local_finetune_schedule(model, optimizer, 0, config) == "head_only"
    assert not any(parameter.requires_grad for parameter in model.layer4.parameters())
    assert apply_local_finetune_schedule(model, optimizer, 1, config) == "partial_encoder"
    assert all(parameter.requires_grad for parameter in model.layer4.parameters())
    assert not any(parameter.requires_grad for parameter in model.layer1.parameters())
    assert apply_local_finetune_schedule(model, optimizer, 2, config) == "full_model"
    assert all(parameter.requires_grad for parameter in model.layer1.parameters())


@pytest.mark.train_smoke
def test_tiny_cpu_epoch_writes_best_last_and_loadable_checkpoint(tmp_path: Path):
    torch = pytest.importorskip("torch"); pytest.importorskip("torchvision")
    from urban_monitor.change.models import build_change_model
    from urban_monitor.change.training import train_change_stage
    config = {
        "model": {"architecture": "multitask_siamese_unet", "semantic_classes": 2, "pretrained": False, "dropout": 0},
        "training": {"epochs": 1, "batch_size": 1, "num_workers": 0, "patch_size": 64, "amp": False,
                     "learning_rate": 1e-4, "samples_per_epoch": 1, "early_stopping_patience": 2, "seed": 3},
    }
    run = tmp_path / "run"
    best = train_change_stage(config, _index(tmp_path), run, "binary", device_name="cpu", allow_cpu=True)
    assert best.name == "best_binary_change.pth" and best.is_file()
    assert (run / "checkpoints" / "last_binary.pth").is_file()
    payload = torch.load(best, map_location="cpu", weights_only=False)
    model = build_change_model(config); model.load_state_dict(payload["model"]); model.eval()
    with torch.no_grad():
        prediction = model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))["binary_change"]
    assert prediction.shape == (1, 1, 64, 64)
