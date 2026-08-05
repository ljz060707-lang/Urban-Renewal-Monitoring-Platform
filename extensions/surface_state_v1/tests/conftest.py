from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

EXTENSIONS = Path(__file__).resolve().parents[2]
if str(EXTENSIONS) not in sys.path:
    sys.path.insert(0, str(EXTENSIONS))


def _write_pair(root: Path, stem: str, color: tuple[int, int, int], mask_colors):
    image = root / f"{stem}.png"
    mask = root / f"{stem}_mask.png"
    Image.fromarray(np.full((32, 32, 3), color, dtype=np.uint8)).save(image)
    array = np.zeros((32, 32, 4), dtype=np.uint8)
    array[..., :3] = mask_colors[0]
    array[..., 3] = 255
    array[8:24, 8:24, :3] = mask_colors[1]
    Image.fromarray(array, "RGBA").save(mask)
    return image, mask


@pytest.fixture(scope="session")
def synthetic_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("surface_v1")
    rows = []
    colors = [(0, 0, 0), (255, 255, 255), (0, 0, 255), (255, 0, 0)]
    for index in range(8):
        image, mask = _write_pair(root, f"cwld_{index}", (30 + index, 60, 90), (colors[0], colors[(index % 3) + 1]))
        rows.append({"dataset": "cwld", "sample_id": f"cwld_{index}", "scene_id": f"s{index}", "split": "train" if index < 6 else "validation", "image_t1": str(image), "image_t2": "", "mask_t1": str(mask), "mask_t2": ""})
    manifest = root / "cwld.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "kind": "cwld",
        "experiment_name": "synthetic_cwld",
        "cwld_manifest": str(manifest),
        "seed": 20260729,
        "device": "cpu",
        "image_size": 32,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "pin_memory": False,
        "amp": False,
        "epochs": 1,
        "early_stopping_patience": 2,
        "gradient_clip_norm": 1.0,
        "resume": "auto",
        "candidate_status": "cwld_validation_candidate",
        "class_weight_min": 0.25,
        "class_weight_max": 4.0,
        "codec": {"max_anchor_distance": 32.0, "min_distance_margin": 16.0, "ignore_index": 255},
        "optimizer": {"decoder_lr": 0.0003, "weight_decay": 0.01},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return root, manifest, config_path, config


@pytest.fixture(scope="session")
def mini_run(synthetic_project):
    from surface_state_v1.training.full_trainer import train
    root, _, config_path, _ = synthetic_project
    receipt = train(config_path, root)
    return Path(receipt["run_dir"])
