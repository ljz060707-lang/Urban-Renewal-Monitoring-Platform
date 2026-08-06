from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from urban_monitor.compat import autocast_context, create_grad_scaler


def _latest_audit(project_root: Path):
    paths = sorted((project_root / "outputs" / "urrs" / "audits").glob("construction_activity_*"))
    if not paths:
        raise FileNotFoundError("construction audit not found; run `python tools/urrs.py construction audit` first")
    return paths[-1]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_dataloader_smoke(project_root: Path, *, image_size=128, samples=4):
    from urban_renewal.construction.datasets import UCSManifestDataset

    audit_dir = _latest_audit(project_root)
    gate = _json(audit_dir / "ONBOARDING_GATE.json")
    if not gate.get("development_smoke_allowed"):
        raise RuntimeError("development smoke blocked by ONBOARDING_GATE")
    dataset = UCSManifestDataset(audit_dir / "UCS_TRAIN_MANIFEST.csv", image_size=image_size)
    records = []
    for index in range(min(samples, len(dataset))):
        item = dataset[index]
        records.append({
            "sample_id": item["sample_id"], "image_shape": list(item["image"].shape),
            "target_shape": list(item["target"].shape), "target_values": sorted(int(value) for value in item["target"].unique()),
            "image_min": float(item["image"].min()), "image_max": float(item["image"].max()),
        })
    receipt = {
        "status": "success", "scope": "development_dataloader_smoke_not_validation",
        "records": records, "test_pixel_reads": 0, "dataset_pixel_reads": dataset.pixel_reads,
        "current_urrs_release_unchanged": True, "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = audit_dir / "DATALOADER_SMOKE_RECEIPT.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**receipt, "receipt": str(path.resolve())}


def run_20_step_smoke(project_root: Path, *, device="auto", steps=20, image_size=128):
    import torch
    from torch.utils.data import DataLoader

    from urban_renewal.construction.datasets import UCSManifestDataset
    from urban_renewal.construction.partial_labels import PartialLabelLoss
    from urban_renewal.models.construction import ConstructionElementNet
    from urban_renewal.models.construction.checkpoint import save_construction_checkpoint

    if int(steps) != 20:
        raise ValueError("construction first-run smoke is fixed at exactly 20 optimizer steps")
    audit_dir = _latest_audit(project_root)
    gate = _json(audit_dir / "ONBOARDING_GATE.json")
    if not gate.get("development_smoke_allowed"):
        raise RuntimeError("development smoke blocked by ONBOARDING_GATE")
    seed = 20260729
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    selected_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    dataset = UCSManifestDataset(audit_dir / "UCS_TRAIN_MANIFEST.csv", image_size=image_size)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, generator=generator)
    model = ConstructionElementNet(width=8).to(selected_device)
    criterion = PartialLabelLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = create_grad_scaler(selected_device, selected_device.type == "cuda")
    iterator = iter(loader); logs = []; probability_min = 1.0; probability_max = 0.0
    model.train()
    for step in range(20):
        try: batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        image = batch["image"].to(selected_device); texture = batch["texture"].to(selected_device); target = batch["target"].to(selected_device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(selected_device, selected_device.type == "cuda"):
            output = model(image, "ucs", texture)
            losses = criterion(output, target, "ucs")
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite loss at step {step}")
        scaler.scale(losses["total"]).backward(); scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        probabilities = output["ucs_probabilities"].detach()
        probability_min = min(probability_min, float(probabilities.min()))
        probability_max = max(probability_max, float(probabilities.max()))
        logs.append({"step": step + 1, **{name: float(value.detach()) for name, value in losses.items()}, "grad_norm": float(grad_norm)})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "outputs" / "training" / "construction_activity_v1" / f"smoke_{stamp}"
    checkpoint = save_construction_checkpoint(
        run_dir / "checkpoints" / "smoke_checkpoint.pth", model,
        optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
        steps=20, smoke=True, scientific_status="development_smoke_not_validation",
    )
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    with (run_dir / "logs" / "train_steps.jsonl").open("w", encoding="utf-8") as stream:
        for row in logs: stream.write(json.dumps(row) + "\n")
    receipt = {
        "status": "success", "module_id": "construction_activity_v1",
        "scope": "UCS_available_pairs_development_smoke_only", "optimizer_steps": 20,
        "device": str(selected_device), "loss_first": logs[0]["total"], "loss_last": logs[-1]["total"],
        "all_losses_finite": all(np.isfinite(row["total"]) for row in logs),
        "probability_range": [probability_min, probability_max],
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(selected_device)) if selected_device.type == "cuda" else 0,
        "checkpoint": checkpoint, "checkpoint_registerable": False,
        "formal_validation_run": False, "formal_training_allowed": gate["formal_ucs_training_allowed"],
        "test_pixel_reads": {"UCS": 0, "CWLD": 0, "WHU": 0, "SECOND": 0},
        "current_urrs_release_unchanged": True, "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "SMOKE_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**receipt, "run_dir": str(run_dir.resolve())}


def formal_train_guard(project_root: Path, stage: str):
    audit_dir = _latest_audit(project_root)
    gate = _json(audit_dir / "ONBOARDING_GATE.json")
    if not gate.get("formal_ucs_training_allowed"):
        return {"status": "blocked", "stage": stage, "reason": "ONBOARDING_GATE forbids formal UCS training", "blocking_reasons": gate.get("blocking_reasons", []), "test_pixel_reads": gate["test_pixel_reads"]}
    raise NotImplementedError("formal training is gated but no unblocked dataset has been accepted")
