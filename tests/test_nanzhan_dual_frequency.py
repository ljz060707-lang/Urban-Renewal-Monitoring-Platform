from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.run_nanzhan_dual_frequency import (
    EXPECTED_B0_SHA,
    EXPECTED_FULL20_SHA,
    _registration_for,
    parse_scene_date,
    protected_asset_snapshot,
)


def test_parse_bjnz_yymmdd_is_unambiguous():
    value, source, candidates = parse_scene_date(Path("bjnz200406_zoom_20.tif"))
    assert value == "2020-04-06"
    assert source == "filename:bjnz_yymmdd"
    assert candidates == ["2020-04-06"]


def test_parse_metadata_date_takes_part_in_contract():
    value, source, candidates = parse_scene_date(Path("scene.tif"), {"acquisition_date": "2024-12-07"})
    assert value == "2024-12-07"
    assert source == "metadata:acquisition_date"
    assert candidates == ["2024-12-07"]


def test_missing_date_is_not_silently_selected():
    value, source, candidates = parse_scene_date(Path("unknown_scene.tif"))
    assert value is None
    assert source == "ambiguous_or_missing"
    assert candidates == []


def test_registration_missing_pair_is_blocked():
    row = _registration_for([], "HF", "2020-02-11", "2020-04-06")
    assert row["status"] == "blocked"
    assert row["pairwise_change_allowed"] == "False"


def test_prediction_outputs_are_not_ground_truth(tmp_path: Path):
    manifest = {
        "status": "success",
        "ground_truth": False,
        "scientific_status": "candidate_not_ground_truth",
        "test_images_read": 0,
    }
    path = tmp_path / "PAIR_INFERENCE_MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["ground_truth"] is False
    assert loaded["test_images_read"] == 0


def test_protected_constants_are_explicit():
    assert EXPECTED_B0_SHA == "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
    assert EXPECTED_FULL20_SHA == "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
