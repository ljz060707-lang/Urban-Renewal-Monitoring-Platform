"""Tests for the non-invasive post-inference watcher."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_PATH = Path(__file__).resolve().parents[1] / "tools" / "watch_xian_additive_post_inference.py"
_SPEC = spec_from_file_location("watch_xian_additive_post_inference", _PATH)
assert _SPEC and _SPEC.loader
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_watcher_waits_without_canonical_final_receipt(tmp_path):
    result = _MODULE.inspect_readiness(tmp_path / "canonical")
    assert result["status"] == "WAITING_CANONICAL_FINAL_RECEIPT"
    assert result["canonical_status"] is None


def test_watcher_accepts_only_pass_canonical_receipt(tmp_path):
    canonical = tmp_path / "canonical"
    receipt = canonical / "receipts" / "FINAL_TEMPORAL_PACKAGE_RECEIPT.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"status":"partial"}\n', encoding="utf-8")
    assert _MODULE.inspect_readiness(canonical)["status"] == "WAITING_CANONICAL_FINAL_RECEIPT"
    receipt.write_text('{"status":"pass"}\n', encoding="utf-8")
    ready = _MODULE.inspect_readiness(canonical)
    assert ready["status"] == "READY"
    assert ready["canonical_status"] == "pass"


def test_watcher_lock_blocks_duplicate_processes(tmp_path):
    first = _MODULE._acquire_watcher_lock(tmp_path)
    assert first is not None
    try:
        assert _MODULE._acquire_watcher_lock(tmp_path) is None
    finally:
        _MODULE._release_watcher_lock(first)
    second = _MODULE._acquire_watcher_lock(tmp_path)
    assert second is not None
    _MODULE._release_watcher_lock(second)
