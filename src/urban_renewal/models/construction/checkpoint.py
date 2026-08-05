from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def save_construction_checkpoint(path, model, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "module_id": "construction_activity_v1", **metadata}
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def load_construction_checkpoint(path, model, map_location="cpu"):
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload
