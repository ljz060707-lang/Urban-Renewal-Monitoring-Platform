from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest


def _gpu_index(root: Path) -> Path:
    rows = []
    for split in ("train", "val"):
        image = np.full((64, 64, 3), 45, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8); mask[18:42, 18:42] = 1
        paths = {}
        for name, value in (("t1", image), ("t2", image + 20), ("cd", mask), ("new", mask), ("dem", np.zeros_like(mask)), ("gt1", np.zeros_like(mask)), ("gt2", mask)):
            path = root / split / f"{name}.png"; path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(value).save(path); paths[name] = path
        rows.append({"sample_id":f"second:{split}:gpu","dataset_name":"second","split":split,"t1_path":paths["t1"],"t2_path":paths["t2"],
                     "gt_cd_path":paths["cd"],"gt_new_path":paths["new"],"gt_dem_path":paths["dem"],"gt_t1_path":paths["gt1"],"gt_t2_path":paths["gt2"],
                     "has_binary":True,"has_directional":True,"has_semantic":True})
    path = root / "index.csv"; pd.DataFrame(rows).to_csv(path, index=False); return path


@pytest.mark.gpu
def test_real_cuda_three_stage_smoke(tmp_path: Path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip(f"real CUDA unavailable; torch={torch.__version__}, runtime={torch.version.cuda}")
    pytest.importorskip("torchvision")
    from urban_monitor.change.training import train_change_pipeline
    config = {
        "model": {"architecture": "multitask_siamese_unet", "semantic_classes": 2, "pretrained": False, "dropout": 0},
        "training": {"epochs": 1, "batch_size": 1, "num_workers": 0, "patch_size": 64, "amp": True,
                     "learning_rate": 1e-4, "samples_per_epoch": 1, "early_stopping_patience": 2, "seed": 9},
    }
    results = train_change_pipeline(config, _gpu_index(tmp_path), tmp_path / "gpu_run", device_name="cuda:0")
    assert set(results) == {"binary", "semantic", "directional"}
    assert all(path.is_file() for path in results.values())
