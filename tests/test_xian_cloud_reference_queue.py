"""Regression tests for the resume-safe Xi'an cloud queue."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "watch_xian_cloud_reference_queue.py"
_SPEC = spec_from_file_location("xian_cloud_reference_queue", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_atomic_json_retries_transient_windows_permission_error(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    real_replace = _MODULE.os.replace
    calls = {"count": 0}

    def flaky_replace(source, destination):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError(5, "sharing violation")
        return real_replace(source, destination)

    monkeypatch.setattr(_MODULE.os, "replace", flaky_replace)
    _MODULE.atomic_json(target, {"status": "ok"})

    assert calls["count"] == 2
    assert target.read_text(encoding="utf-8") == '{\n  "status": "ok"\n}\n'


def test_sync_output_downloads_remote_geometry_into_local_run(tmp_path, monkeypatch):
    queue = _MODULE.CloudQueue(
        tmp_path,
        tmp_path / "key.pem",
        tmp_path / "sources",
        tmp_path / "metadata",
        tmp_path / "python.exe",
    )
    calls = []

    def fake_download(remote, local, recursive=False, timeout=14400):
        calls.append((remote, Path(local), recursive))
        manifest = Path(local) / "04_geometry" / "GEOMETRY_MANIFEST.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"status":"pass"}\n', encoding="utf-8")

    def fail_upload(*args, **kwargs):
        raise AssertionError("sync_output must not upload the local target")

    monkeypatch.setattr(queue, "scp_from_remote", fake_download, raising=False)
    monkeypatch.setattr(queue, "scp", fail_upload)

    state = {"date": "2021-01-29", "sync_status": "PENDING"}
    assert queue.sync_output("2021-01-29", state) is True
    assert calls == [
        (
            "/data/xian_urrs/reference_outputs/reference_2021-01-29/.",
            tmp_path / "06_geometry_raw" / "reference_2021-01-29",
            True,
        )
    ]
    assert state["sync_status"] == "VERIFIED"


def test_launch_remote_recovers_only_stale_failure_lock(tmp_path, monkeypatch):
    queue = _MODULE.CloudQueue(
        tmp_path,
        tmp_path / "key.pem",
        tmp_path / "sources",
        tmp_path / "metadata",
        tmp_path / "python.exe",
    )
    captured = []

    def fake_run(args, timeout):
        captured.append((args, timeout))
        return "71450"

    monkeypatch.setattr(_MODULE, "run_process", fake_run)
    pid = queue.launch_remote("2017-07-14", "1", "/data/xian_urrs/reference_input/2017-07-14/2017-07-14.tif")

    assert pid == 71450
    command = captured[0][0][-1]
    assert "FAILURE_RECEIPT_*.json" in command
    assert "rmdir '/data/xian_urrs/reference_outputs/reference_2017-07-14/.launch_lock'" in command
    assert "run_cloud_single_aliyun.sh" in command
    assert "CUDA_VISIBLE_DEVICES=1" in command
    assert "cuda:0" in command


def test_repairs_legacy_materialized_geotransform_without_changing_pixels(tmp_path):
    import numpy as np
    import rasterio
    from affine import Affine

    tif = tmp_path / "legacy.tif"
    pixels = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    # This is the historical bug: the GDAL tuple was passed directly to
    # Affine, so origin_y occupies d and the y scale is zero in e.
    bad = Affine(108.75, 2.5e-6, 0.0, 34.43, 0.0, -2.5e-6)
    with rasterio.open(tif, "w", driver="GTiff", width=2, height=2, count=3, dtype="uint8", crs="EPSG:4326", transform=bad) as dst:
        dst.write(pixels)

    material = {"output_sha256": "before-recorded"}
    sha = _MODULE.repair_materialized_geotransform("2017-07-14", tif, material, tmp_path / "receipts")

    with rasterio.open(tif) as src:
        assert list(src.transform)[:6] == [2.5e-6, 0.0, 108.75, 0.0, -2.5e-6, 34.43]
        np.testing.assert_array_equal(src.read(), pixels)
    assert material["output_sha256"] == sha
    assert (tmp_path / "receipts" / "GEOTRANSFORM_REPAIR_2017-07-14.json").is_file()


def test_cloud_wrapper_normalizes_bound_single_gpu_device():
    wrapper = Path(__file__).resolve().parents[1] / "outputs" / "urban-renewal-phase1-starter" / "tools" / "run_cloud_single_aliyun.sh"
    text = wrapper.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "DEVICE=cuda:0" in text
    assert "URRS_PHYSICAL_GPU" in text
    assert "single-device environment" in text


def test_cloud_queue_does_not_upload_when_all_slots_are_busy():
    assert _MODULE.cloud_slot_available([], 2) is True
    assert _MODULE.cloud_slot_available([{"gpu": "0"}], 2) is True
    assert _MODULE.cloud_slot_available([{"gpu": "0"}, {"gpu": "1"}], 2) is False


def test_queue_completion_requires_verified_sync_for_every_date():
    records = [
        {"date": date, "geometry_status": "COMPLETE", "sync_status": "VERIFIED"}
        for date in _MODULE.DATES
    ]
    records[0]["sync_status"] = "PENDING"

    assert _MODULE.all_outputs_synced(records) is False
    records[0]["sync_status"] = "VERIFIED"
    assert _MODULE.all_outputs_synced(records) is True
