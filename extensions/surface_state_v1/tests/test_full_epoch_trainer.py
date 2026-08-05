import json


def test_complete_mini_epoch_is_finite(mini_run):
    receipt = json.loads((mini_run / "TRAINING_RECEIPT.json").read_text())
    report = json.loads((mini_run / "VALIDATION_REPORT.json").read_text())
    assert receipt["status"] == "success"
    assert report["cwld"]["macro_mIoU"] >= 0
    assert report["cwld"]["test_read_count"] == 0
