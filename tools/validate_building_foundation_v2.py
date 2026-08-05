"""Validate a Building Foundation v2 checkpoint on WHU validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


class GpuRuntimeLock:
    """Fail-closed lock shared with the Building v2 training entry point."""

    def __init__(self, path: Path, run_dir: Path) -> None:
        self.path, self.run_dir = path, run_dir
        self.payload = {"pid": os.getpid(), "run_dir": str(run_dir.resolve()), "task": "building_foundation_v2_validation"}
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"GPU lock is held: {self.path}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, ensure_ascii=False, indent=2); handle.write("\n")
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                current = {}
            if current.get("pid") == self.payload["pid"] and current.get("run_dir") == self.payload["run_dir"]:
                self.path.unlink(missing_ok=True)
            self.acquired = False


def write_failure(run_dir: Path, error: BaseException) -> None:
    payload = {"status": "failed", "stage": "validation", "error_type": type(error).__name__, "error": str(error), "test_used": False, "test_pixel_reads": 0}
    write_json(run_dir / "07_validation" / "FAILURE_RECEIPT.json", payload)
    write_json(run_dir / "RUN_MANIFEST.validation_failed.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Data Set\WHU_Building"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    receipt = json.loads((args.run_dir / "TRAINING_RECEIPT.json").read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or Path(receipt["best_checkpoint"])
    if not checkpoint.exists(): raise SystemExit(f"checkpoint missing: {checkpoint}")
    from urban_monitor.building_foundation_v2 import BuildingFoundationNet, WHUBuildingDataset
    from urban_monitor.building_foundation_v2.data import build_whu_manifest
    from urban_monitor.building_foundation_v2.metrics import binary_metrics, summarize_metrics
    resolved = {}
    resolved_path = args.run_dir / "resolved_config.yaml"
    if resolved_path.exists():
        import yaml  # type: ignore
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    manifest = args.run_dir / "05_data" / "validation_only_manifest.csv"
    rows = [r for r in build_whu_manifest(args.data_root, manifest, train_limit=0, validation_limit=None) if r["split"] == "validation"]
    if any("test" in str(r["image_path"]).lower() for r in rows): raise RuntimeError("test path detected")
    dataset = WHUBuildingDataset(rows, image_size=int(resolved.get("validation_image_size", resolved.get("image_size", 256))), train=False)
    loader = DataLoader(dataset, batch_size=int(resolved.get("batch_size", 2)), shuffle=False, num_workers=0)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    lock = GpuRuntimeLock(Path(__file__).resolve().parents[1] / "outputs" / "runtime_locks" / "urrs_gpu.lock", args.run_dir) if device.type == "cuda" else None
    lock_audit = {"required": lock is not None, "lock_path": str((Path(__file__).resolve().parents[1] / "outputs" / "runtime_locks" / "urrs_gpu.lock").resolve()), "acquired": False, "released": False, "test_pixel_reads": 0}
    write_json(args.run_dir / "00_governance" / "GPU_LOCK_AUDIT.validation.json", lock_audit)
    try:
        if lock is not None:
            lock.acquire(); lock_audit["acquired"] = True; write_json(args.run_dir / "00_governance" / "GPU_LOCK_AUDIT.validation.json", lock_audit)
        model = BuildingFoundationNet(base_channels=16, height_enabled=bool(resolved.get("height_enabled", False))).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"]); model.eval()
        metrics = []
        with torch.no_grad():
            for batch in loader:
                outputs = model(batch["image"].to(device)); probabilities = outputs["building_probability"].cpu().numpy(); targets = batch["mask"].numpy()
                for probability, target in zip(probabilities, targets): metrics.append(binary_metrics(probability[0], target[0]))
    except BaseException as exc:
        write_failure(args.run_dir, exc)
        raise
    finally:
        if lock is not None:
            lock.release(); lock_audit["released"] = True; write_json(args.run_dir / "00_governance" / "GPU_LOCK_AUDIT.validation.json", lock_audit)
    summary = summarize_metrics(metrics)
    result = {"status": "available", "split": "validation", "sample_count": len(rows), "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint), "metrics": summary, "test_used": False, "test_pixel_reads": 0}
    write_json(args.run_dir / "07_validation" / "VALIDATION_REPORT.json", result)
    (args.run_dir / "07_validation" / "VALIDATION_REPORT.md").write_text("# Building Foundation v2 Validation\n\n- split: validation only\n- test_used: false\n- metrics: `" + json.dumps(summary, ensure_ascii=False) + "`\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
