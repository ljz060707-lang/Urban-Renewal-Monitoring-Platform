from urban_monitor.building_foundation_v2.checkpoint import candidate_eligibility


def test_test_reads_block_candidate():
    result = candidate_eligibility(metrics={"f1": 0.5, "iou": 0.3, "predicted_positive_fraction": 0.5}, leakage_count=0, test_pixel_reads=1, source_migration_ok=True)
    assert not result.eligible
    assert "test_read" in result.reason
