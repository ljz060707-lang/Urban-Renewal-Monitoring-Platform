from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CocoBuildingDataset, collate_fn
from .model import build_model


def train_teacher(
    coco_root: str | Path,
    output_dir: str | Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    num_classes: int = 2,
    checkpoint: str | Path | None = None,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CocoBuildingDataset(coco_root)
    if not dataset:
        raise ValueError("COCO dataset is empty")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    model = build_model(num_classes=num_classes, checkpoint=checkpoint).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs}")
        for images, targets in progress:
            images = [x.to(device) for x in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "mean_loss": running / max(len(loader), 1),
        }
        torch.save(state, output / f"epoch_{epoch:03d}.pth")
        torch.save(state, output / "latest.pth")
    return output / "latest.pth"
