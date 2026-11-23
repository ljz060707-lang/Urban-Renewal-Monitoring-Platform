"""Small dependency-light helpers used by the isolated R5 pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: json_default(value) for key, value in row.items()})
    return len(materialized)


def write_parquet(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Write a real parquet table when the isolated pyarrow runtime exists.

    Falling back to a CSV is explicit and never creates a fake/empty parquet
    placeholder.  The caller can therefore distinguish a measured registry
    from a blocked serialization step.
    """

    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - depends on runtime
        return {"status": "BLOCKED_NO_PARQUET_ENGINE", "error": str(exc), "rows": len(materialized)}
    try:
        table = pa.Table.from_pylist([dict(row) for row in materialized])
        pq.write_table(table, path, compression="zstd")
        return {"status": "PASS", "rows": len(materialized), "path": str(path)}
    except Exception as exc:  # pragma: no cover - depends on data types
        return {"status": "BLOCKED_PARQUET_WRITE", "error": str(exc), "rows": len(materialized)}
