import numpy as np
import pytest

from urban_monitor.building_foundation_v2.data import WHUBuildingDataset, load_label_contract
from urban_monitor.building_foundation_v2.metrics import binary_metrics, prediction_distribution_gate, summarize_metrics


def test_label_contract_is_explicit_and_supports_zero_polarity():
    assert load_label_contract(None)["building_values"] == [1]
    contract = load_label_contract({"label_polarity": "zero_is_building"})
    assert contract["building_values"] == [0]
    assert contract["background_values"] == [1]


def test_whu_dataset_rejects_test_rows_before_reading_pixels():
    with pytest.raises(ValueError, match="refuses test rows"):
        WHUBuildingDataset([{"split": "test"}], label_contract={"building_values": [1], "background_values": [0]})


@pytest.mark.parametrize(
    ("predicted", "actual", "std", "status"),
    [
        (1.0, 0.1, 0.2, "collapsed_all_positive"),
        (0.0, 0.1, 0.2, "collapsed_all_negative"),
        (0.2, 0.1, 0.0, "collapsed_constant_probability"),
        (0.2, 0.9, 0.2, "label_contract_suspect"),
        (0.2, 0.1, 0.2, "healthy_prediction_distribution"),
    ],
)
def test_prediction_distribution_gate_statuses(predicted, actual, std, status):
    result = prediction_distribution_gate(
        {"predicted_positive_fraction": predicted, "actual_positive_fraction": actual, "probability_std": std}
    )
    assert result["distribution_status"] == status


def test_metrics_add_distribution_fields_and_reject_nonfinite():
    result = summarize_metrics([binary_metrics(np.array([[0.1, 0.9]]), np.array([[0, 1]]))])
    assert result["distribution_status"] == "healthy_prediction_distribution"
    assert 0.0 <= result["predicted_positive_fraction"] <= 1.0
    with pytest.raises(ValueError, match="NaN"):
        binary_metrics(np.array([[np.nan]]), np.array([[0]]))


def test_native_recovery_tooling_has_test_guards():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    for name in (
        "init_building_v2_native_recovery.py",
        "audit_whu_label_semantics.py",
        "inspect_rsbuilding_native_runtime.py",
        "compare_rsbuilding_input_pipelines.py",
        "validate_rsbuilding_iter2000_native.py",
        "finalize_building_v2_native_recovery.py",
    ):
        text = (root / "tools" / name).read_text(encoding="utf-8")
        assert "test_pixel_reads" in text or "Test" in text
