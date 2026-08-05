from __future__ import annotations

import json
from pathlib import Path

from urban_renewal.review.seed_temporal import (
    SeedTemporalLabelStore,
    create_seed_temporal_inventory,
)


def _make_review_tree(tmp_path: Path) -> Path:
    review_root = tmp_path / "review"
    review_dir = review_root / "tongzhou_20260801"
    review_dir.mkdir(parents=True)
    preview_dir = review_root / "review_migration_20260802_122123" / "transition_previews"
    preview_dir.mkdir(parents=True)
    rows = [
        {
            "sample_id": "transition_20180506_to_20200723_00",
            "track_id": "construction_review_track_00",
            "before_date": "2018-05-06",
            "after_date": "2020-07-23",
            "transition": "20180506_to_20200723",
            "preview_quality": {"preview_status": "valid"},
            "pair_preview_path": str(preview_dir / "previews" / "pair.png"),
            "raw_before_path": str(preview_dir / "raw" / "before.png"),
            "raw_after_path": str(preview_dir / "raw" / "after.png"),
            "source_metadata": {"before": {"source_crs": "EPSG:4326"}, "after": {"source_crs": "EPSG:4326"}},
            "evidence": {"building_geometry": {"status": "available", "paths": []}},
            "evidence_missing": ["construction_elements"],
            "full20_supported": False,
        }
    ]
    (preview_dir / "TRANSITION_REVIEW_INVENTORY.json").write_text(
        json.dumps({"records": rows}, ensure_ascii=False), encoding="utf-8"
    )
    return review_dir


def test_seed_inventory_preserves_source_and_is_model_independent(tmp_path):
    review_dir = _make_review_tree(tmp_path)
    output_dir = tmp_path / "human_seed_labels_20260802_140000"
    inventory = create_seed_temporal_inventory(review_dir, output_dir)

    assert inventory["seed_count"] == 1
    row = inventory["records"][0]
    assert row["sample_id"] == "seed_temporal_000"
    assert row["supersedes_sample_ids"] == ["transition_20180506_to_20200723_00"]
    assert row["label_source"] == "human_seed_label"
    assert row["formal_ground_truth"] is False
    assert row["model_independent"] is True
    assert inventory["model_prediction_display"] is False
    assert (output_dir / "SEED_LABEL_INVENTORY.json").is_file()
    assert (output_dir / "SEED_LABEL_DECISIONS.csv").is_file()
    assert (output_dir / "SEED_LABEL_EVENTS.jsonl").is_file()
    assert (output_dir / "SEED_LABEL_RESUME_RECEIPT.json").is_file()


def test_seed_store_requires_vocab_and_keeps_verified_seed_non_ground_truth(tmp_path):
    review_dir = _make_review_tree(tmp_path)
    output_dir = tmp_path / "human_seed_labels_20260802_140001"
    inventory = create_seed_temporal_inventory(review_dir, output_dir)
    store = SeedTemporalLabelStore(review_dir, inventory)
    payload = {
        "sample_id": "seed_temporal_000",
        "inventory_hash": inventory["inventory_hash"],
        "before_surface_state": "ordinary_bare_land",
        "after_surface_state": "possible_active_construction",
        "transition_type": "CONSTRUCTION_START",
        "construction_activity_visible": True,
        "confidence": 0.8,
        "notes": "independent seed review",
        "reviewer": "reviewer",
        "decision": "ACCEPT",
        "human_confirmed": False,
    }
    first = store.save(payload)
    assert first["label_source"] == "human_seed_label"
    assert first["formal_ground_truth"] is False

    second = store.save({**payload, "human_confirmed": True})
    assert second["label_source"] == "human_verified_seed"
    assert second["formal_ground_truth"] is False
    assert store.summary()["labeled_count"] == 1
    assert store.summary()["verified_seed_count"] == 1
    assert json.loads((output_dir / "SEED_LABEL_STATE.json").read_text(encoding="utf-8"))["summary"]["formal_ground_truth_count"] == 0

