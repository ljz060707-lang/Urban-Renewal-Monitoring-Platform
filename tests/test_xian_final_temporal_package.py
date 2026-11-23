"""Public-seam tests for the Xi'an actual-date temporal package."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_xian_final_temporal_package.py"
_SPEC = spec_from_file_location("xian_final_temporal_package", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_equal_actual_acquisitions_are_folded_not_stable():
    result = _MODULE.fold_actual_epochs(["2015-06-12", "2015-06-12", "2015-06-12"])
    assert result["status"] == "DUPLICATE_EPOCH"
    assert result["genuine_interval"] is False
    assert _MODULE.classify_pattern((1, 1, 1), ["2015-06-12"] * 3) == "DUPLICATE_EPOCH"


def test_distinct_off_date_observations_are_valid_interval_evidence():
    result = _MODULE.fold_actual_epochs(["2015-03-01", "2015-07-10", "2015-11-20"])
    assert result["status"] == "OFF_DATE_VALID"
    assert result["genuine_interval"] is True
    assert _MODULE.classify_pattern((1, 0, 1), ["2015-03-01", "2015-07-10", "2015-11-20"]) == "REDEVELOPMENT_CANDIDATE"


def test_missing_date_is_temporally_insufficient():
    result = _MODULE.fold_actual_epochs([None, "2019-01-01", None])
    assert result["status"] == "TEMPORAL_INSUFFICIENT"
    assert result["genuine_interval"] is False


def test_discover_geometry_normalizes_windows_manifest_paths(tmp_path):
    manifest_root = tmp_path / "local_xar190101" / "04_geometry"
    manifest_root.mkdir(parents=True)
    (manifest_root / "BUILDINGS.gpkg").write_bytes(b"gpkg")
    (manifest_root / "GEOMETRY_MANIFEST.json").write_text(
        '{"status":"pass","scene":{"selected_source":"D:\\\\Data Set\\\\xiannew\\\\xar190101_combined\\\\xar190101_zoom_20.tif","scene_date":"2017-09-12","source_sha256":"abc"},"checkpoint_sha256":"b0"}\n',
        encoding="utf-8",
    )
    found = _MODULE.discover_geometry([tmp_path], {"xar190101": {}})
    assert [row["stem"] for row in found] == ["xar190101"]
