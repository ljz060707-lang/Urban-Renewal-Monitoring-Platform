from pathlib import Path

import pytest

from tools import nanjing_jiangbei_acceptance as acceptance


REAL_NANJING_SOURCE = Path(r"D:\Data Set\Satellite\nanjing_jiangbei")


def test_parse_date_token_supports_compact_scene_dates():
    assert acceptance.parse_date_token("NJ_jianbei241020_zoom_19.tif") == "2024-10-20"
    assert acceptance.parse_date_token("scene_180208") == "2018-02-08"
    assert acceptance.parse_date_token("scene_without_acquisition_date") is None


def test_pair_specs_has_adjacent_and_each_to_final_pairs():
    records = [{"acquisition_date": f"202{year}-01-01"} for year in range(0, 7)]
    pairs = acceptance.pair_specs(records)

    assert len(pairs["adjacent"]) == 6
    assert len(pairs["cumulative"]) == 6
    assert pairs["adjacent"][0]["from"] == "2020-01-01"
    assert pairs["adjacent"][-1]["to"] == "2026-01-01"
    assert all(pair["to"] == "2026-01-01" for pair in pairs["cumulative"])
    assert all(not pair["derived_from_temporal_state_aggregation"] for group in pairs.values() for pair in group)


def test_state_ontology_is_stable_and_model_derived():
    assert sorted(acceptance.STATE_NAMES) == list(range(11))
    assert acceptance.STATE_NAMES[10] == "uncertain"


@pytest.mark.skipif(not REAL_NANJING_SOURCE.exists(), reason="real Nanjing Jiangbei imagery is not mounted")
def test_real_nanjing_input_audit_regression(tmp_path):
    audit = acceptance.audit_inputs(REAL_NANJING_SOURCE, tmp_path / "external_acceptance")

    assert len(audit["records"]) == 7
    assert audit["dates"] == [
        "2015-01-04",
        "2016-08-18",
        "2017-12-11",
        "2018-02-08",
        "2019-03-16",
        "2022-12-02",
        "2024-10-20",
    ]
    assert audit["common_grid"]["status"] == "pass"
    assert audit["spatial"]["status"] == "pass"
    assert audit["grid_ready"] is True
    assert (tmp_path / "external_acceptance" / "01_aligned_imagery" / "VALID_OVERLAP_MASK.tif").exists()
    assert all(record["valid_pixel_ratio"] == 1.0 for record in audit["records"])
