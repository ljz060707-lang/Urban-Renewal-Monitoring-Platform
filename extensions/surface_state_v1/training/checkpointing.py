from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path, payload) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return sha256(path)


def select_better(candidate: dict, current: dict | None, mode: str) -> bool:
    if current is None:
        return True
    candidate_key = (candidate["selection_score"], candidate.get("macro_F1", 0), -candidate["epoch"])
    current_key = (current["selection_score"], current.get("macro_F1", 0), -current["epoch"])
    return candidate_key > current_key


def write_hashes(run_dir: Path, paths: list[Path]) -> None:
    lines = [f"{sha256(path)}  {path.relative_to(run_dir)}" for path in paths if path.exists()]
    (run_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
