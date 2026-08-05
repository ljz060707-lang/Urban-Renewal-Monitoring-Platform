from urban_monitor.building_foundation_v2.checkpoint import CandidateResult, candidate_eligibility, select_best_checkpoint


def test_class_collapse_is_ineligible():
    result = candidate_eligibility(metrics={"f1": 0.9, "iou": 0.8, "predicted_positive_fraction": 1.0}, leakage_count=0, test_pixel_reads=0, source_migration_ok=True)
    assert not result.eligible
    assert "class_collapse" in result.reason


def test_baseline_retained_when_candidate_does_not_improve():
    baseline = CandidateResult("iter_2000", True, 0.9, "baseline")
    candidate = CandidateResult("v2", True, 0.89, "candidate")
    assert select_best_checkpoint([candidate], baseline=baseline).candidate_id == "iter_2000"
