from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_b2_uses_b0_not_b1_and_fixed_registry():
    text = (ROOT / "tools" / "native_b2_worker.py").read_text(encoding="utf-8")
    assert "load_b0_state" in text
    assert "decode_head.boundary_head." in text
    assert "B2BoundaryDecoder" in text
    init = (ROOT / "tools" / "init_building_v2_b2_boundary_ablation.py").read_text(encoding="utf-8")
    assert "B1_TRAIN_SAMPLE_REGISTRY.csv" in init
    assert "random_reselection" in init


def test_b2_smoke_is_gated_before_pilot_and_never_uses_test():
    text = (ROOT / "tools" / "run_building_v2_b2_boundary_ablation.py").read_text(encoding="utf-8")
    assert "all smoke and pilot candidates" not in text  # finalizer owns this wording
    assert "smoke_pass" in text
    assert "test_pixel_reads" in text
    assert "B2_L005" in text and "B2_L010" in text and "B2_L020" in text


def test_b2_finalizer_protection_and_statuses():
    text = (ROOT / "tools" / "finalize_building_v2_b2_boundary_ablation.py").read_text(encoding="utf-8")
    for phrase in ("pixel_iou_drop_max", "object_f1_0.5_no_drop", "native_baseline_retained_boundary_ablation_completed", "b2_boundary_candidate_ready_for_final_static_comparison", "test_pixel_reads"):
        assert phrase in text

