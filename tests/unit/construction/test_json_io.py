from __future__ import annotations

import json
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from urban_renewal.construction.json_io import dump_json, load_json


def test_load_json_accepts_bom_and_non_bom(tmp_path: Path) -> None:
    payload = {"status": "ok", "中文": "保留"}
    plain = tmp_path / "plain.json"
    bom = tmp_path / "bom.json"
    plain.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    bom.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    assert load_json(plain) == payload
    assert load_json(bom) == payload


def test_dump_json_is_utf8_without_bom_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    dump_json(path, {"value": 1, "finite": 2.5})
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert load_json(path)["value"] == 1
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_json_reads_are_never_partial_under_concurrent_writes(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"

    def write(index: int) -> None:
        dump_json(path, {"index": index, "status": "running", "test_pixel_reads": 0})

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write, range(20)))
    result = load_json(path)
    assert result["status"] == "running"
    assert result["test_pixel_reads"] == 0


def test_corrupt_json_reports_path(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_bytes(b"{not-json")
    with pytest.raises(json.JSONDecodeError, match="failed_json_path"):
        load_json(path)


def test_worker_status_reads_bom_historical_receipts(tmp_path: Path) -> None:
    worker_path = Path(__file__).resolve().parents[3] / "tools" / "construction_training_worker.py"
    spec = importlib.util.spec_from_file_location("construction_training_worker_for_test", worker_path)
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    state = {"status": "FAILED", "test_pixel_reads": 0}
    receipt = {"pid": 54708, "detached": True}
    (tmp_path / "TRAINING_STATE.json").write_bytes(b"\xef\xbb\xbf" + json.dumps(state).encode("utf-8"))
    (tmp_path / "TRAINING_LAUNCH_RECEIPT.json").write_bytes(b"\xef\xbb\xbf" + json.dumps(receipt).encode("utf-8"))
    result = worker.status_run(tmp_path)
    assert result["status"] == "FAILED"
    assert result["launch_receipt"]["pid"] == 54708
    assert result["test_pixel_reads"] == 0


def test_powershell_writer_contract_is_no_bom_and_atomic() -> None:
    script = Path(__file__).resolve().parents[3] / "scripts" / "construction_task_training.ps1"
    text = script.read_text(encoding="utf-8-sig")
    assert "function Write-Utf8NoBomAtomic" in text
    assert "UTF8Encoding($false)" in text
    assert "Flush($true)" in text
    assert "File]::Replace" in text
