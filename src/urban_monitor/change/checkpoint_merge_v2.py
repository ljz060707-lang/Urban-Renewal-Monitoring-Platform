from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .models.siamese_unet import build_change_model


@dataclass
class MergeRecord:
    target_key: str
    source: str
    source_key: str | None
    status: str
    reason: str = ""


def _state(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a mapping")
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_v2_checkpoints(
    config: dict,
    dsifn_checkpoint: str | Path,
    s2looking_checkpoint: str | Path,
    output_checkpoint: str | Path,
    report_path: str | Path,
) -> dict:
    """Merge compatible V1 heads into a V2 initial state without guessing keys."""
    import torch

    ds_path, s2_path = Path(dsifn_checkpoint), Path(s2looking_checkpoint)
    for path in (ds_path, s2_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    model = build_change_model(config)
    target = model.state_dict()
    sources = {
        "dsifn": _state(torch.load(ds_path, map_location="cpu", weights_only=False)),
        "s2looking": _state(torch.load(s2_path, map_location="cpu", weights_only=False)),
    }
    records: list[MergeRecord] = []
    merged = {}
    directional = ("new_building_head.", "demolition_head.")
    semantic = ("semantic_t1_head.", "semantic_t2_head.", "uncertainty_head.")
    for key, tensor in target.items():
        if key.startswith(semantic):
            merged[key] = tensor
            reason = "derived uncertainty is not imported" if key.startswith("uncertainty_head.") else "no verified semantic checkpoint"
            records.append(MergeRecord(key, "random_init", None, "kept", reason))
            continue
        source_name = "s2looking" if key.startswith(directional) else "dsifn"
        candidate = sources[source_name].get(key)
        if candidate is None:
            raise KeyError(f"{source_name} checkpoint missing required tensor: {key}")
        if tuple(candidate.shape) != tuple(tensor.shape) or candidate.dtype != tensor.dtype:
            raise ValueError(
                f"Tensor mismatch for {key}: source={tuple(candidate.shape)}/{candidate.dtype}, "
                f"target={tuple(tensor.shape)}/{tensor.dtype}"
            )
        merged[key] = candidate
        records.append(MergeRecord(key, source_name, key, "copied"))
    model.load_state_dict(merged, strict=True)
    output = Path(output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "v2_merge": {
                "dsifn_checkpoint": str(ds_path.resolve()),
                "dsifn_sha256": _sha(ds_path),
                "s2looking_checkpoint": str(s2_path.resolve()),
                "s2looking_sha256": _sha(s2_path),
            },
        },
        output,
    )
    summary = {"dsifn_loaded_count": 0, "s2looking_loaded_count": 0, "random_init_count": 0,
               "shape_mismatch_count": 0, "missing_count": 0, "unexpected_count": 0}
    for record in records:
        if record.source in {"dsifn", "s2looking"} and record.status == "copied": summary[f"{record.source}_loaded_count"] += 1
        elif record.source == "random_init": summary["random_init_count"] += 1
    report = {
        "status": "success",
        "output_checkpoint": str(output.resolve()),
        "output_sha256": _sha(output),
        "records": [asdict(item) for item in records],
        "summary": summary,
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
