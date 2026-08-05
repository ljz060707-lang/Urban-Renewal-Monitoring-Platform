from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_b1_pilot_has_hard_label_and_split_guards():
    text = (ROOT / "tools" / "run_building_v2_b1_native_pilot.py").read_text(encoding="utf-8")
    assert "only train and validation are admitted" in text
    assert 'test_pixel_reads": 0' in text
    assert 'choices=("sanity", "b1", "all")' in text


def test_native_worker_preserves_pack_transform_and_disables_only_single_sample_mosaic():
    text = (ROOT / "tools" / "native_b1_worker.py").read_text(encoding="utf-8")
    assert "MultiImgPackSegInputs" in text
    assert "skip_type_keys" in text
    assert "RandomMosaic_Modified" in text
    assert "init_weights = lambda" in text


def test_finalizer_writes_selection_and_protection_receipts():
    text = (ROOT / "tools" / "finalize_building_v2_b1_native_pilot.py").read_text(encoding="utf-8")
    for artifact in ("MODEL_SELECTION_RECEIPT.json", "CHECKPOINT_SELECTION_FINAL.json", "B0_B1_COMPARISON.csv", "PROTECTED_ASSET_INVARIANCE.json"):
        assert artifact in text
    assert "native_baseline_retained_training_pipeline_validated" in text
    assert "test_pixel_reads" in text
