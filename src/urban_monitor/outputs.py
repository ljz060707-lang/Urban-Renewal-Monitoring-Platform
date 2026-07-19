from __future__ import annotations

from pathlib import Path


def prepare_output(path: str | Path, *, force: bool) -> Path:
    """Refuse silent replacement; remove one explicit output when --force is set."""

    target = Path(path)
    if target.exists():
        if not force:
            raise FileExistsError(f"Output exists; rerun with --force to replace it: {target}")
        if target.is_dir():
            raise IsADirectoryError(f"prepare_output only accepts a file path: {target}")
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
