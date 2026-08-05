from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from urban_monitor.change.adapters.second import SECONDAdapter
from urban_monitor.change.preparation import _quick_records
from urban_monitor.change.second_dev_split import build_or_load_second_development_split, second_transition_stratum


def _write(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def _adapter(tmp_path: Path) -> SECONDAdapter:
    root = tmp_path / "SECOND"
    patterns = [(0, 5), (5, 0), (5, 5), (0, 0), (5, 5), (0, 5), (5, 0), (0, 0)]
    for index, (before, after) in enumerate(patterns):
        stem = f"{index:05d}"
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        gt1 = np.zeros((4, 4), dtype=np.uint8); gt2 = np.zeros((4, 4), dtype=np.uint8)
        gt1[1:3, 1:3] = before; gt2[1:3, 1:3] = after
        if index == 7:
            gt1[0, 0] = gt2[0, 0] = 255
        for folder, array in (("T1", image), ("T2", image), ("GT_T1", gt1), ("GT_T2", gt2)):
            _write(root / "train" / folder / f"{stem}.png", array)
    return SECONDAdapter(root, {
        "allowed_splits": ["train"], "mapping_verified": True, "building_class_id": 5,
        "semantic_ignore_index": 255,
        "development_validation": {"enabled": True, "count": 3, "seed": 20260726,
                               "manifest": str(tmp_path / "SECOND_DEV_VALIDATION_IDS.csv"), "source_split": "train"},
    })


def test_frozen_second_train_split_is_reused_and_quick_is_a_subset(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    first = build_or_load_second_development_split(adapter.discover(), adapter)
    original_bytes = first.manifest_path.read_bytes()
    second = build_or_load_second_development_split(adapter.discover(), adapter)
    assert second.manifest_path.read_bytes() == original_bytes
    assert first.manifest_sha256 == second.manifest_sha256
    rows = list(csv.DictReader(first.manifest_path.open(encoding="utf-8")))
    assert sum(row["development_split"] == "validation" for row in rows) == 3
    assert {row["source_split"] for row in rows} == {"train"}
    assert not any(":test:" in row["sample_id"] for row in rows)
    val = {row["sample_id"] for row in rows if row["development_split"] == "validation"}
    train = {row["sample_id"] for row in rows if row["development_split"] == "train"}
    assert val.isdisjoint(train)
    quick = _quick_records(first.records, {"quick": {"train_samples_per_dataset": 2, "validation_samples_per_dataset": 2}})
    assert {item.sample_id for item in quick if item.split == "val"} <= val
    assert not ({item.sample_id for item in quick if item.split == "train"} & val)
    assert first.audit["test_images_read"] == 0


def test_second_directional_strata_ignore_void_pixels() -> None:
    gt1 = np.array([[255, 0], [5, 5]], dtype=np.uint8)
    gt2 = np.array([[255, 5], [0, 5]], dtype=np.uint8)
    assert second_transition_stratum(gt1, gt2, building_class_id=5, ignore_index=255) == ("both", True, True)
    assert second_transition_stratum(np.full((2, 2), 255), np.full((2, 2), 255), building_class_id=5) == ("background", False, False)


def test_second_manifest_audit_hash_mismatch_is_not_silently_accepted(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    frozen = build_or_load_second_development_split(adapter.discover(), adapter)
    audit = json.loads(frozen.audit_path.read_text(encoding="utf-8"))
    audit["manifest_sha256"] = "0" * 64
    frozen.audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        build_or_load_second_development_split(adapter.discover(), adapter)
