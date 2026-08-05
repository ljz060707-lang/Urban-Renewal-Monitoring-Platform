import json
from pathlib import Path

import yaml

from surface_state_v1.training.full_trainer import train


def test_resume_continues_global_step(synthetic_project, mini_run):
    root, _, _, config = synthetic_project
    first = json.loads((mini_run / "TRAINING_RECEIPT.json").read_text())["global_step"]
    config = {**config, "epochs": 2}
    config_path = root / "resume.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    receipt = train(config_path, root, run_dir=str(mini_run))
    assert receipt["global_step"] > first
