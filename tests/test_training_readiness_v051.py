from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from urban_monitor.change.splits import quick_subset, resolve_train_validation_test_splits
from urban_monitor.ground_truth import read_building_labels
from urban_monitor.preflight import run_personal_preflight
from urban_monitor.project_discovery import discover_project
from urban_monitor.change.adapters.base import read_binary, read_s2looking_binary
from PIL import Image


def _frame():
    return pd.DataFrame([
        {"sample_id": "a", "dataset_name": "s2looking", "split": "train", "has_binary": True, "has_semantic": False, "has_directional": True},
        {"sample_id": "b", "dataset_name": "s2looking", "split": "val", "has_binary": True, "has_semantic": False, "has_directional": True},
        {"sample_id": "c", "dataset_name": "s2looking", "split": "test", "has_binary": True, "has_semantic": False, "has_directional": True},
    ])


def test_validation_isolated_from_test_and_test_fallback_is_opt_in():
    train, val, test, summary = resolve_train_validation_test_splits(_frame(), {})
    assert train.sample_id.tolist() == ["a"] and val.sample_id.tolist() == ["b"] and test.sample_id.tolist() == ["c"]
    assert not summary["test_used_for_validation"]
    only_test = _frame().query("split != 'val'")
    with pytest.raises(ValueError, match="No validation split"):
        resolve_train_validation_test_splits(only_test, {})
    _, val, _, summary = resolve_train_validation_test_splits(only_test, {"use_test_as_validation": True})
    assert val.sample_id.tolist() == ["c"] and summary["test_used_for_validation"]


def test_quick_subset_is_per_dataset_and_excludes_test():
    frame = pd.concat([_frame(), _frame().assign(dataset_name="dsifn", sample_id=lambda x: "d_" + x.sample_id)], ignore_index=True)
    quick = quick_subset(frame, {"quick": {"train_samples_per_dataset": 1, "validation_samples_per_dataset": 1}})
    assert set(quick.split) == {"train", "val"}
    assert quick.groupby(["split", "dataset_name"]).size().max() == 1


def test_s2looking_verified_rgb_binary_is_allowed_but_generic_binary_rejects_it(tmp_path: Path):
    path = tmp_path / "s2_rgb.png"
    label = np.zeros((4, 4, 3), dtype=np.uint8); label[1:3, 1:3] = (0, 0, 255); Image.fromarray(label).save(path)
    assert read_s2looking_binary(path).sum() == 4
    with pytest.raises(ValueError, match="single-channel"):
        read_binary(path)


def test_pipeline_skips_unsupervised_stages_and_keeps_binary_as_final(tmp_path: Path, monkeypatch):
    from urban_monitor.change import training
    frame = _frame().assign(has_directional=False, has_semantic=False)
    index = tmp_path / "index.csv"; frame.to_csv(index, index=False)
    def fake_stage(config, index_path, run, stage, **kwargs):
        path = Path(run) / "checkpoints" / f"best_{stage}_change.pth"; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"ok"); return path
    monkeypatch.setattr(training, "train_change_stage", fake_stage)
    result = training.train_change_pipeline({}, index, tmp_path / "run", device_name="cpu", allow_cpu=True)
    status = json.loads((tmp_path / "run" / "training_stage_status.json").read_text(encoding="utf-8"))
    assert list(result) == ["binary"]
    assert status["semantic"]["status"] == "skipped_no_supervision"
    assert status["directional"]["status"] == "skipped_no_supervision"


def test_binary_only_validation_metrics_do_not_concatenate_empty_directional_lists():
    from urban_monitor.change.training import _validation_head_metrics
    import torch
    binary = _validation_head_metrics(torch.tensor([3.0, 1.0, 2.0]), [np.array([.9, .1])], [np.array([1, 0])])
    absent = _validation_head_metrics(torch.zeros(3), [], [])
    assert binary["status"] == "available" and binary["f1"] is not None and binary["iou"] is not None and binary["pr_auc"] is not None
    assert absent == {"precision": None, "recall": None, "f1": None, "iou": None, "pr_auc": None, "status": "skipped_no_supervision"}


def test_binary_only_freezes_unsupported_heads_and_keeps_shared_binary_trainable():
    pytest.importorskip("torch")
    from urban_monitor.change.models import build_change_model
    from urban_monitor.change.training import configure_binary_only
    model = build_change_model({"pretrained": False, "semantic_classes": 2})
    trainable = {name for name, _ in configure_binary_only(model)}
    assert any(name.startswith("stem.") for name in trainable)
    assert any(name.startswith("dec1.") for name in trainable)
    assert any(name.startswith("binary_change_head.") for name in trainable)
    assert not any(name.startswith(("new_building_head.", "demolition_head.", "semantic_t1_head.", "semantic_t2_head.", "uncertainty_head.")) for name in trainable)


def test_gpkg_buildings_layer_is_explicit_and_single_layer_is_compatible(tmp_path: Path):
    path = tmp_path / "labels.gpkg"
    frame = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:32650")
    frame.to_file(path, layer="buildings", driver="GPKG")
    frame.to_file(path, layer="notes", driver="GPKG")
    assert len(read_building_labels(path)) == 1
    other = tmp_path / "other.gpkg"; frame.to_file(other, layer="one_layer", driver="GPKG")
    assert len(read_building_labels(other)) == 1
    with pytest.raises(ValueError, match="multiple layers"):
        read_building_labels(path, building_layer="missing")


def test_external_data_root_and_personal_preflight(tmp_path: Path):
    root, aoi = tmp_path / "Data Set" / "Personal", "pilot"
    project = root / aoi; (project / "images").mkdir(parents=True); (project / "labels").mkdir()
    for date in ("2022-01-01", "2023-01-01"):
        image = project / "images" / f"{date}.tif"
        with rasterio.open(image, "w", driver="GTiff", width=8, height=8, count=3, dtype="uint8", crs="EPSG:32650", transform=from_origin(0, 8, 1, 1)) as dst:
            dst.write(np.zeros((3, 8, 8), dtype="uint8"))
        gpd.GeoDataFrame(geometry=[box(1, 1, 3, 3)], crs="EPSG:32650").to_file(project / "labels" / f"{date}.gpkg", layer="buildings", driver="GPKG")
    assert discover_project(aoi, root).project_dir == project.resolve()
    report = run_personal_preflight(aoi, root, tmp_path / "report")
    assert report["READY_TO_PREPARE"] and Path(report["report_path"]).is_file()
