from __future__ import annotations

"""BOM-tolerant and crash-safe JSON I/O for URRS construction artifacts."""

import json
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping


_WRITE_LOCK = threading.RLock()


def load_json(path: str | Path) -> Any:
    """Read UTF-8 JSON with or without a UTF-8 BOM."""

    path = Path(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(f"{exc.msg}; failed_json_path={path}", exc.doc, exc.pos) from exc


def atomic_write_text(path: str | Path, content: str) -> None:
    """Atomically write UTF-8 text without a BOM."""

    _atomic_write_text(Path(path), content)


def _atomic_write_text(path: Path, content: str) -> None:
    with _WRITE_LOCK:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except Exception:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def dump_json(path: str | Path, payload: Mapping[str, Any], *, indent: int = 2) -> None:
    """Atomically write UTF-8 JSON without a BOM."""

    content = json.dumps(_json_safe(payload), ensure_ascii=False, indent=indent, allow_nan=False, default=str) + "\n"
    _atomic_write_text(Path(path), content)
