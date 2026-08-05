from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.audit_ucs_readiness import seed_progress


def _write_inventory(root: Path, payload: object) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "SEED_LABEL_INVENTORY.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("payload", "source", "total"),
    [
        ([{"sample_id": "a"}, {"sample_id": "b"}], "$", 2),
        ({"items": [1, 2]}, "items", 2),
        ({"samples": [1, 2, 3]}, "samples", 3),
        ({"inventory": [1]}, "inventory", 1),
        ({"records": [1, 2, 3, 4]}, "records", 4),
        ({"seed_inventory": [1, 2]}, "seed_inventory", 2),
        ({"total": 5}, "total", 5),
        ({"count": 6}, "count", 6),
        ({"inventory_total": 7}, "inventory_total", 7),
    ],
)
def test_seed_progress_supports_declared_inventory_shapes(tmp_path: Path, payload: object, source: str, total: int) -> None:
    _write_inventory(tmp_path, payload)

    result = seed_progress(tmp_path)

    assert result["parse_status"] == "SEED_INVENTORY_PARSED"
    assert result["inventory_container_key"] == source
    assert result["inventory_total"] == total
    assert result["seed"] == f"0/{total}"
    assert result["seed_status"] == "PENDING_HUMAN_SEED_LABELS"


def test_seed_progress_unknown_structure_is_explicit_error(tmp_path: Path) -> None:
    _write_inventory(tmp_path, {"status": "success", "metadata": {"version": 1}})

    result = seed_progress(tmp_path)

    assert result["status"] == "SEED_INVENTORY_PARSE_ERROR"
    assert result["seed_status"] == "SEED_INVENTORY_PARSE_ERROR"
    assert result["top_level_keys"] == ["status", "metadata"]
    assert result["seed"] is None
    assert result["inventory_total"] is None


def test_seed_progress_parses_real_seed_inventory_without_modifying_it() -> None:
    project_root = Path(__file__).resolve().parents[3]
    seed_root = project_root / "outputs" / "urrs" / "construction"
    inventory_path = seed_root / "human_seed_labels_20260802_134145" / "SEED_LABEL_INVENTORY.json"
    assert inventory_path.is_file(), inventory_path
    before = hashlib.sha256(inventory_path.read_bytes()).hexdigest()

    result = seed_progress(seed_root)

    after = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    assert result["top_level_type"] == "object"
    assert result["top_level_keys"] == [
        "status", "scope", "seed_inventory_version", "source_transition_inventory",
        "source_transition_inventory_hash", "records", "seed_count", "inventory_hash",
        "coverage_transition_requirements", "coverage_scene_requirements", "coverage_status",
        "label_source", "formal_ground_truth", "model_independent", "model_prediction_display",
        "current_urrs_release_unchanged", "test_pixel_reads", "created_at",
    ]
    assert result["inventory_container_key"] == "records"
    assert result["inventory_container_length"] == 96
    assert result["inventory_total"] == 96
    assert result["reviewed_count"] == 0
    assert result["remaining_count"] == 96
    assert result["seed"] == "0/96"
    assert result["seed_status"] == "PENDING_HUMAN_SEED_LABELS"
    assert result["status"] == "PENDING_HUMAN_SEED_LABELS"
    assert before == after
