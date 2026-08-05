"""One-time, fixed-threshold evaluation of the held-out change-detection test set.

This module deliberately does not reuse the validation probability cache or the
threshold scanner.  Test evaluation is an explicit terminal action: it forwards
the test split once, evaluates the caller-provided thresholds, then records the
exact checkpoint and configuration used.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .augmentations import ChangePairTransform
from .datasets import PublicChangeDataset
from .models import build_change_model
from .runtime_config import resolve_evaluation_runtime_config


HEADS = {
    "binary": ("binary_change", "binary"),
    "new": ("new_building", "new"),
    "demolition": ("demolition", "demolition"),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _pr_auc(probability: np.ndarray, target: np.ndarray) -> float | None:
    truth = target.astype(bool).ravel()
    if not truth.any():
        return None
    order = np.argsort(-probability.ravel(), kind="stable")
    ordered_truth = truth[order]
    hits = np.cumsum(ordered_truth)
    precision = hits / np.arange(1, len(ordered_truth) + 1)
    return float(precision[ordered_truth].sum() / ordered_truth.sum())


def _fixed_metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    predicted, truth = probability >= threshold, target.astype(bool)
    true_positive = int((predicted & truth).sum())
    false_positive = int((predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative),
        "iou": true_positive / max(1, true_positive + false_positive + false_negative),
        "pr_auc": _pr_auc(probability, target),
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "true_positive_pixels": true_positive,
        "predicted_positive_pixel_ratio": float(predicted.mean()),
        "actual_positive_pixel_ratio": float(truth.mean()),
    }


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.casefold().isin({"true", "1", "yes"})


def _supervision_mode(test: pd.DataFrame) -> tuple[str, list[str]]:
    """Derive evaluable heads from index metadata, before any image is decoded."""
    if "has_binary" not in test or not _truthy(test["has_binary"]).any():
        raise ValueError("Final test requires binary supervision in split=test.")
    directional = (
        "has_directional" in test
        and _truthy(test["has_directional"]).any()
    )
    return ("multitask", ["binary", "new", "demolition"]) if directional else ("binary_only", ["binary"])


def _validate_thresholds(thresholds: dict[str, float | None], active_heads: list[str]) -> None:
    for head in active_heads:
        value = thresholds.get(head)
        if value is None:
            raise ValueError(f"Final test requires an explicit {head} threshold for supervised head '{head}'.")
        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {head} threshold: {value!r}") from exc
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Invalid {head} threshold: {value!r}; expected 0 < threshold < 1.")


def run_final_test(
    checkpoint: str | Path,
    index_path: str | Path,
    run: str | Path,
    *,
    device_name: str,
    thresholds: dict[str, float | None],
    confirm_final_test: bool,
    cli_config: dict | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> pd.DataFrame:
    """Evaluate exactly the held-out ``test`` split once at fixed thresholds.

    ``confirm_final_test`` is intentionally required at this lower level too,
    so future callers cannot accidentally bypass the CLI acknowledgement.
    """
    if not confirm_final_test:
        raise PermissionError("Final test is blocked: pass --confirm-final-test after fixing all thresholds.")

    import torch

    checkpoint, index_path, run = Path(checkpoint).resolve(), Path(index_path).resolve(), Path(run).resolve()
    manifest_path = run / "FINAL_TEST_MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError(f"Final test already recorded at {manifest_path}; refusing to read test again.")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    # Hash and optional manifest expectation are checked before loading the
    # index or constructing a dataset, so a bad checkpoint cannot consume a
    # held-out test read.
    checkpoint_hash = _sha256_file(checkpoint)
    if expected_checkpoint_sha256 and checkpoint_hash != expected_checkpoint_sha256:
        raise ValueError(
            "Checkpoint SHA256 mismatch before test read: "
            f"expected {expected_checkpoint_sha256}, got {checkpoint_hash}."
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Normalize before reading the index or constructing a dataset.  In
    # particular, old checkpoints legitimately retain batch_size: auto.
    checkpoint_config = resolve_evaluation_runtime_config(payload["config"], run)
    # Only construct a dataset from test rows.  Train/validation images are
    # never decoded, transformed, or forwarded by this command.
    frame = pd.read_csv(index_path, dtype=str).fillna("")
    test = frame[frame["split"].astype(str).str.casefold() == "test"].copy()
    if test.empty:
        raise ValueError("Final test requires a non-empty split=test index; no fallback split is allowed.")
    supervision_mode, active_heads = _supervision_mode(test)
    _validate_thresholds(thresholds, active_heads)
    validation_sample_count = int((frame["split"].astype(str).str.casefold() == "validation").sum())
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    model = build_change_model(checkpoint_config).to(device).eval()
    model.load_state_dict(payload["model"])
    runtime_training = checkpoint_config["training"]
    patch_size = runtime_training["patch_size"]
    dataset = PublicChangeDataset(test, transform=ChangePairTransform(patch_size, training=False))
    batch_size = min(16, runtime_training["resolved_batch_size"])

    collected = {head: [[], []] for head in active_heads}
    with torch.no_grad():
        # This is the only model-forward loop in final-test.  Thresholds are
        # evaluated afterwards on collected probabilities, never by re-forwarding.
        for start in range(0, len(dataset), batch_size):
            batch = [dataset[offset] for offset in range(start, min(len(dataset), start + batch_size))]
            output = model(
                torch.stack([sample["t1"] for sample in batch]).to(device),
                torch.stack([sample["t2"] for sample in batch]).to(device),
            )
            for head in active_heads:
                output_name, target_name = HEADS[head]
                for offset, sample in enumerate(batch):
                    target = sample[target_name]
                    if target is None:
                        continue
                    collected[head][0].append(torch.sigmoid(output[output_name][offset, 0]).detach().cpu().numpy().astype(np.float32).ravel())
                    collected[head][1].append(target.numpy().astype(np.uint8).ravel())

    rows = []
    for head in HEADS:
        if head not in active_heads:
            rows.append({
                "head": head, "status": "skipped_no_supervision", "threshold": None,
                "precision": None, "recall": None, "f1": None, "iou": None, "pr_auc": None,
                "false_positive_pixels": None, "false_negative_pixels": None, "true_positive_pixels": None,
                "predicted_positive_pixel_ratio": None, "actual_positive_pixel_ratio": None,
                "sample_count": len(test), "checkpoint_sha256": checkpoint_hash,
            })
            continue
        probabilities, targets = collected[head]
        if not probabilities:
            raise ValueError(f"Test split has no {head} targets; final-test metrics would be misleading.")
        row = {
            "head": head, "status": "available",
            **_fixed_metrics(np.concatenate(probabilities), np.concatenate(targets), float(thresholds[head])),
            "sample_count": len(test),
            "checkpoint_sha256": checkpoint_hash,
        }
        rows.append(row)
    output = pd.DataFrame(rows)
    metrics_path = run / "metrics" / "final_test_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    manifest = {
        "run_id": run.name,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "config_hash": _config_hash(cli_config if cli_config is not None else checkpoint_config),
        "checkpoint_config_hash": _config_hash(checkpoint_config),
        "resolved_runtime": {key: runtime_training[key] for key in ("resolved_batch_size", "resolved_num_workers", "patch_size", "cache_max_gb", "cache_mode")},
        "supervision_mode": supervision_mode,
        "active_heads": active_heads,
        "thresholds": {head: (float(thresholds[head]) if head in active_heads else None) for head in HEADS},
        "binary_threshold": float(thresholds["binary"]),
        "new_threshold": float(thresholds["new"]) if "new" in active_heads else None,
        "demolition_threshold": float(thresholds["demolition"]) if "demolition" in active_heads else None,
        "threshold_source": "validation",
        "validation_sample_count": validation_sample_count,
        "test_used": True,
        "test_read": True,
        "test_used_for_training": False,
        "test_used_for_threshold_selection": False,
        "test_sample_count": len(test),
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "metrics_path": str(metrics_path.resolve()),
        "single_forward_pass": True,
    }
    # Manifest is last: a failed partial attempt must not consume the one-time gate.
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
