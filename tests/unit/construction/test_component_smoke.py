from __future__ import annotations

import json

from urban_renewal.pipelines.construction_component_smoke import run_component_smoke


def test_component_smoke_receipt_is_development_only(tmp_path):
    result = run_component_smoke(tmp_path)
    receipt = json.loads((tmp_path / "outputs" / "urrs" / "smoke").glob("construction_component_smoke_*").__iter__().__next__().joinpath("COMPONENT_SMOKE_RECEIPT.json").read_text(encoding="utf-8"))

    assert result["status"] == "success"
    assert result["scope"] == "development_component_smoke_not_validation"
    assert result["tfr"]["2"]["invalid_timestep_zero"] is True
    assert result["tfr"]["4"]["gradient_present"] is True
    assert result["no_mti"]["integration"] == "no_mti"
    assert result["mti"]["enabled"] is False
    assert result["mti"]["selection_status"] == "experimental_unselected"
    assert result["tracking"]["split_events"] == 1
    assert result["area"]["audit"] == "pass"
    assert result["formal_metrics_allowed"] is False
    assert result["checkpoint_registerable"] is False
    assert receipt["test_pixel_reads"] == {"UCS": 0, "CWLD": 0, "WHU": 0, "SECOND": 0}
