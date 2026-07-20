from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from urban_monitor.change.adapters.dsifn import DSIFNAdapter
from urban_monitor.change.adapters.s2looking import S2LookingAdapter
from urban_monitor.change.adapters.second import SECONDAdapter, semantic_ids
from urban_monitor.change.datasets import PublicChangeDataset, change_collate


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _rgb(value: int = 10) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def test_s2looking_normalizes_binary_and_directional_labels(tmp_path: Path):
    root = tmp_path / "S2Looking" / "train"
    for name in ("Image1", "Image2"):
        _save(root / name / "x.png", _rgb())
    new = np.zeros((8, 8), dtype=np.uint8)
    new[1:3, 1:3] = 255
    demolition = np.zeros_like(new)
    demolition[5:7, 5:7] = 255
    _save(root / "label1" / "x.png", new)
    _save(root / "label2" / "x.png", demolition)
    _save(root / "label" / "x.png", np.maximum(new, demolition))

    adapter = S2LookingAdapter(root.parent)
    records = adapter.discover()
    normalized = adapter.normalize(records[0], tmp_path / "normalized")

    assert normalized.has_binary and normalized.has_directional
    assert normalized.gt_cd_path.is_file()
    assert "mismatch=0.00000000" in normalized.notes


def test_second_derives_change_without_fabricating_input_path(tmp_path: Path):
    root = tmp_path / "SECOND" / "train"
    for name in ("T1", "T2"):
        _save(root / name / "x.png", _rgb())
    t1 = np.zeros((8, 8), dtype=np.uint8)
    t2 = t1.copy()
    t2[2:5, 2:5] = 1
    _save(root / "GT_T1" / "x.png", t1)
    _save(root / "GT_T2" / "x.png", t2)

    adapter = SECONDAdapter(root.parent, {"building_class_id": 1})
    discovered = adapter.discover()[0]
    assert discovered.gt_cd_path is None
    normalized = adapter.normalize(discovered, tmp_path / "normalized")

    assert normalized.gt_cd_path.is_file()
    assert np.asarray(Image.open(normalized.gt_new_path)).sum() == 9
    assert np.asarray(Image.open(normalized.gt_dem_path)).sum() == 0


def test_second_rgb_mapping_must_be_explicit(tmp_path: Path):
    path = tmp_path / "rgb.png"
    _save(path, _rgb(128))
    with pytest.raises(ValueError, match="explicit verified class_mapping_file"):
        semantic_ids(path)

    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"128,128,128": 3}), encoding="utf-8")
    adapter = SECONDAdapter(tmp_path, {"building_class_id": 3, "class_mapping_file": str(mapping)})
    assert np.all(semantic_ids(path, adapter.color_mapping) == 3)


def test_dsifn_keeps_directional_supervision_missing(tmp_path: Path):
    root = tmp_path / "DSIFN" / "train"
    for name in ("T1", "T2"):
        _save(root / name / "x.png", _rgb())
    _save(root / "GT" / "x.png", np.eye(8, dtype=np.uint8) * 255)
    adapter = DSIFNAdapter(root.parent)
    normalized = adapter.normalize(adapter.discover()[0], tmp_path / "normalized")
    assert normalized.has_binary
    assert not normalized.has_directional
    assert normalized.gt_new_path is None and normalized.gt_dem_path is None


def test_public_dataset_and_collate_preserve_missing_tasks(tmp_path: Path):
    torch = pytest.importorskip("torch")
    image = tmp_path / "image.png"
    label = tmp_path / "label.png"
    _save(image, _rgb())
    _save(label, np.ones((8, 8), dtype=np.uint8))
    frame = [{
        "sample_id": "dsifn:train:x", "dataset_name": "dsifn", "split": "train",
        "t1_path": str(image), "t2_path": str(image), "gt_cd_path": str(label),
        "gt_new_path": "", "gt_dem_path": "", "gt_t1_path": "", "gt_t2_path": "",
        "has_binary": "True", "has_directional": "False", "has_semantic": "False",
    }]
    dataset = PublicChangeDataset(__import__("pandas").DataFrame(frame))
    item = dataset[0]
    assert item["new"] is None
    batch = change_collate([item])
    assert batch["new"] == [None]
    assert torch.equal(batch["has_directional"], torch.tensor([False]))
