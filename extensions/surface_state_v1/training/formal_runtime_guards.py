"""Process-local guards for formal Surface State V1 training runs."""
from __future__ import annotations

from contextlib import contextmanager

from . import full_trainer
from .amp_safe_runtime import amp_safe_probability_bce


@contextmanager
def formal_training_runtime():
    """Install narrow runtime fixes without mutating historical trainer files."""

    original_run_directory = full_trainer._run_directory

    def complete_run_directory(repo, kind, requested=None):
        run = original_run_directory(repo, kind, requested)
        for name in ("checkpoints", "logs", "metrics", "visualizations"):
            (run / name).mkdir(parents=True, exist_ok=True)
        return run

    full_trainer._run_directory = complete_run_directory
    try:
        with amp_safe_probability_bce():
            yield
    finally:
        full_trainer._run_directory = original_run_directory

