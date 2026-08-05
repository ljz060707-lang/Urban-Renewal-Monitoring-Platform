"""Training entrypoint guard for future verified surface-state datasets."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..datasets.onboarding import audit_dataset


def train_surface_state(config_path: str | Path, *, smoke: bool = False) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    audit = audit_dataset(config_path)
    if audit["status"] != "ready":
        raise RuntimeError("Surface-state training blocked: dataset onboarding and verified label mapping must pass first")
    mapping_path = config.get("label_mapping_file")
    mapping = yaml.safe_load(Path(mapping_path).read_text(encoding="utf-8")) if mapping_path else {}
    if not mapping.get("verified", False):
        raise RuntimeError("Surface-state training blocked: label mapping verified=false")
    # The dataset-specific loader is intentionally the extension seam. A real
    # dataset must supply its tensor reader before optimization can begin.
    raise NotImplementedError(
        "Surface-state optimizer is scaffolded but no verified tensor reader is registered for this dataset; "
        "implement the adapter reader after onboarding, then run with --smoke"
    )
