from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(cwd: str | Path | None = None) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_run_manifest(
    path: str | Path,
    *,
    run_id: str,
    aoi_id: str,
    command: str,
    config_path: str | Path,
    status: str,
    input_files: Iterable[str | Path] = (),
    output_files: Iterable[str | Path] = (),
    warnings: Iterable[str] = (),
    start_time: str = "",
    model_version: str = "",
    schema_version: str = "0.3.0",
    source_crs: str = "",
    analysis_crs: str = "",
    exchange_crs: str = "EPSG:4326",
) -> Path:
    """Write the stable v0.3 run-provenance contract."""
    config = Path(config_path)
    inputs = [Path(value) for value in input_files]
    outputs = [Path(value) for value in output_files]
    payload: dict[str, Any] = {
        "run_id": run_id,
        "aoi_id": aoi_id,
        "command": command,
        "config_path": str(config),
        "config_hash": file_sha256(config) if config.exists() else "",
        "git_commit": git_commit(config.parent if config.exists() else None),
        "model_version": model_version,
        "schema_version": schema_version,
        "start_time": start_time or utc_now(),
        "end_time": utc_now(),
        "status": status,
        "input_files": [str(item) for item in inputs],
        "input_hashes": {str(item): file_sha256(item) for item in inputs if item.is_file()},
        "output_files": [str(item) for item in outputs],
        "warnings": list(warnings),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "source_crs": source_crs,
        "analysis_crs": analysis_crs,
        "exchange_crs": exchange_crs,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
