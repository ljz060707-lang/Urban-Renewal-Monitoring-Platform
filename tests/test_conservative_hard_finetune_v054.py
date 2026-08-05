from pathlib import Path

import yaml


def test_conservative_smoke_uses_fixed_source_thresholds_and_small_hard_weight():
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "configs" / "change_public_s2looking_hard_finetune_conservative_smoke.yaml").read_text(encoding="utf-8"))
    assert config["training"]["epochs"] == 2
    assert config["training"]["learning_rate"] == 2e-6
    assert config["training"]["freeze_encoder"] is True
    assert config["sampling"]["hard_false_negative_weight"] == 1.25
    assert config["sampling"]["max_hard_sample_fraction"] == .10
    assert config["sampling"]["max_both_fraction_within_hard"] == .40
    assert config["model_selection"]["source_thresholds"] == {"binary": .50, "new": .60, "demolition": .30}
