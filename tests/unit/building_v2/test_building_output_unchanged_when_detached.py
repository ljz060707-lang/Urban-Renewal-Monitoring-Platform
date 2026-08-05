from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_head_only_records_native_logit_invariance():
    text = (ROOT / "tools" / "native_b2_diagnostic_worker.py").read_text(encoding="utf-8")
    assert "building_logits_max_abs_diff" in text
    assert "building_logits_unchanged_within_float_error" in text

