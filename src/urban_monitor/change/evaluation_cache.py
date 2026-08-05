from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .augmentations import ChangePairTransform
from .datasets import PublicChangeDataset
from .models import build_change_model
from .splits import resolve_train_validation_test_splits


HEADS = {"binary": ("binary_change", "binary"), "new": ("new_building", "new"), "demolition": ("demolition", "demolition")}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_probabilities(checkpoint: str | Path, index_path: str | Path, cache_root: str | Path, *,
                             device_name: str = "cuda:0", split: str = "validation") -> tuple[dict, dict]:
    """Forward a split once and cache raw probabilities/targets for all heads."""
    import torch
    if split != "validation":
        raise ValueError("Probability cache for model selection is validation-only; test is isolated")
    checkpoint, index_path, cache_root = Path(checkpoint), Path(index_path), Path(cache_root)
    checkpoint_hash, full_index_hash = sha256_file(checkpoint), sha256_file(index_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    frame = pd.read_csv(index_path, dtype=str).fillna("")
    _, validation, test, selection = resolve_train_validation_test_splits(frame, config)
    if validation.empty: raise ValueError("No validation samples available")
    identity_columns = [column for column in ("sample_id", "split", "t1_path", "t2_path", "gt_cd_path", "gt_new_path", "gt_dem_path", "has_binary", "has_directional") if column in validation]
    index_hash = hashlib.sha256(validation[identity_columns].sort_values("sample_id").to_csv(index=False).encode("utf-8")).hexdigest()
    model_hash = hashlib.sha256(json.dumps(config.get("model", {}), sort_keys=True).encode()).hexdigest()
    preprocessing = {"patch_size": int(config.get("training", {}).get("patch_size", 256)), "training": False}
    preprocessing_hash = hashlib.sha256(json.dumps(preprocessing, sort_keys=True).encode()).hexdigest()
    key_payload = {"checkpoint_sha256": checkpoint_hash, "index_sha256": index_hash, "model_config_hash": model_hash,
                   "preprocessing_hash": preprocessing_hash, "split": split}
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
    destination = cache_root / key; data_path = destination / "probabilities.npz"; meta_path = destination / "metadata.json"
    if data_path.is_file() and meta_path.is_file():
        loaded = np.load(data_path)
        values = {head: (loaded[f"{head}_probability"], loaded[f"{head}_target"]) for head in HEADS}
        metadata = json.loads(meta_path.read_text(encoding="utf-8")); metadata["cache_hit"] = True
        return values, metadata
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    model = build_change_model(config).to(device).eval(); model.load_state_dict(payload["model"])
    dataset = PublicChangeDataset(validation, transform=ChangePairTransform(preprocessing["patch_size"], training=False))
    try:
        configured = int(config.get("training", {}).get("batch_size", 8))
    except (TypeError, ValueError):
        configured = 8
    batch_size = min(16, max(1, configured))
    collected = {head: [[], []] for head in HEADS}; started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = [dataset[index] for index in range(start, min(len(dataset), start + batch_size))]
            output = model(torch.stack([sample["t1"] for sample in batch]).to(device),
                           torch.stack([sample["t2"] for sample in batch]).to(device))
            for head, (output_name, target_name) in HEADS.items():
                for offset, sample in enumerate(batch):
                    target = sample[target_name]
                    if target is None: continue
                    collected[head][0].append(torch.sigmoid(output[output_name][offset, 0]).detach().cpu().numpy().astype(np.float32).ravel())
                    collected[head][1].append(target.numpy().astype(np.uint8).ravel())
    # DSIFN binary-only validation has no directional targets.  Preserve an
    # explicit empty payload for those heads instead of concatenating nothing.
    values = {
        head: (np.concatenate(parts[0]), np.concatenate(parts[1])) if parts[0] and parts[1]
        else (np.empty(0, dtype=np.float32), np.empty(0, dtype=np.uint8))
        for head, parts in collected.items()
    }
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path, **{f"{head}_{kind}": array for head, pair in values.items() for kind, array in zip(("probability", "target"), pair)})
    metadata = {**key_payload, "full_index_sha256": full_index_hash, "cache_key": key, "cache_hit": False, "inference_seconds": time.perf_counter() - started,
                "validation_samples": len(validation), "test_samples_held_out": len(test), "test_images_read": 0,
                "selection": selection, "data_path": str(data_path.resolve())}
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return values, metadata
