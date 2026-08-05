from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.training import full_trainer
from surface_state_v1.training.formal_runtime_guards import formal_training_runtime


def test_requested_run_directory_gets_all_training_subdirectories(tmp_path):
    requested = tmp_path / "formal-run"
    original = full_trainer._run_directory
    with formal_training_runtime():
        run = full_trainer._run_directory(tmp_path, "fusu_sample", str(requested))
        assert run == requested
        for name in ("checkpoints", "logs", "metrics", "visualizations"):
            assert (run / name).is_dir()
    assert full_trainer._run_directory is original

