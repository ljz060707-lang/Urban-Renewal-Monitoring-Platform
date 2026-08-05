from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import shutil

import numpy as np
import pytest

import tools.nanjing_jiangbei_modular_rebuild as modular
from urban_renewal.modular.contracts import SceneRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_NANJING_SOURCE = Path(r"D:\Data Set\Satellite\nanjing_jiangbei")


def _write_scene(path: Path, value: int) -> None:
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((3, 8, 10), value, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(118.0, 32.0, 0.001, 0.001),
    ) as dataset:
        dataset.write(data)


def test_frozen_contract_cannot_be_mutated() -> None:
    scene = SceneRecord(
        date="2022-12-02",
        path="scene.tif",
        filename="scene.tif",
        sha256="a" * 64,
        size_bytes=1,
        width=10,
        height=8,
        bands=3,
        dtype=("uint8", "uint8", "uint8"),
        crs="EPSG:4326",
        transform=(0.001, 0.0, 118.0, 0.0, -0.001, 32.0, 0.0, 0.0, 1.0),
        bounds=(118.0, 31.992, 118.01, 32.0),
        resolution=(0.001, 0.001),
    )
    with pytest.raises(FrozenInstanceError):
        scene.date = "2024-01-01"  # type: ignore[misc]


def test_module_matrix_is_write_isolated() -> None:
    modules = {item["module_id"]: item for item in modular._module_matrix()}
    assert all(item["can_modify_upstream"] is False for item in modules.values())
    assert "06_construction_single_date" in modules["M6"]["allowed_write_paths"]
    assert all("02_baseline" not in path and "04_baseline" not in path for path in modules["M6"]["allowed_write_paths"])
    assert all("06_construction" not in path and "07_construction" not in path for path in modules["M4"]["allowed_write_paths"])
    assert "M6 -> M4" in modular._data_contracts()["forbidden_edges"]


def test_architecture_audit_writes_complete_read_only_audit_set(tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    modular._audit_current_architecture(output_root)
    required = {
        "CURRENT_DATAFLOW.md",
        "CURRENT_IMPORT_GRAPH.json",
        "CURRENT_CROSS_MODULE_WRITES.json",
        "CURRENT_PREPROCESSING_COMPARISON.json",
        "REFACTOR_PLAN.md",
    }
    assert {path.name for path in (output_root / "00_architecture_audit").iterdir()} >= required
    assert (output_root / "MODULE_RESPONSIBILITY_MATRIX.md").is_file()
    assert (output_root / "MODULE_DATA_CONTRACTS.json").is_file()


def test_report_metric_separation_allows_provenance_flags() -> None:
    baseline_row = {
        "pair_name": "2015-01-04_to_2022-12-02",
        "construction_inputs_used": False,
        "surface_inputs_used": False,
        "new_building_area_m2": 12.0,
    }
    construction_row = {"construction_candidate_count": 3}
    assert modular._baseline_metrics_are_separate([baseline_row]) is True
    assert modular._baseline_metrics_are_separate([baseline_row, construction_row]) is False


def test_duplicate_exclusion_realistic_fixture_does_not_modify_input(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_scene(source / "scene_20150104.tif", 10)
    _write_scene(source / "scene_20221202.tif", 20)
    shutil.copyfile(source / "scene_20221202.tif", source / "scene_20241020.tif")
    before = modular._snapshot_source_files(source)
    audit = modular.audit_inputs(source, tmp_path / "output")
    after = modular._snapshot_source_files(source)
    assert audit["status"] == "pass"
    assert audit["final_date"] == "2022-12-02"
    assert [item["date"] for item in audit["valid_records"]] == ["2015-01-04", "2022-12-02"]
    assert audit["duplicate_audit"]
    duplicate = modular.read_json(tmp_path / "output" / "INPUT_DUPLICATE_AUDIT.json")
    assert duplicate["excluded_dates"] == ["2024-10-20"]
    assert modular._compare_source_snapshot(before, after)["original_data_modified"] is False


@pytest.mark.skipif(not REAL_NANJING_SOURCE.exists(), reason="real Nanjing source is not mounted")
def test_real_nanjing_duplicate_date_regression(tmp_path: Path) -> None:
    before = modular._snapshot_source_files(REAL_NANJING_SOURCE)
    audit = modular.audit_inputs(REAL_NANJING_SOURCE, tmp_path / "real_nanjing_audit")
    after = modular._snapshot_source_files(REAL_NANJING_SOURCE)
    assert audit["status"] == "pass"
    assert len(audit["records"]) == 7
    assert len(audit["valid_records"]) == 6
    assert audit["final_date"] == "2022-12-02"
    assert "2024-10-20" in modular.read_json(tmp_path / "real_nanjing_audit" / "INPUT_DUPLICATE_AUDIT.json")["excluded_dates"]
    assert modular._compare_source_snapshot(before, after)["original_data_modified"] is False
