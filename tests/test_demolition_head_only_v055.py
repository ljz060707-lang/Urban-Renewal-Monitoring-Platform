from pathlib import Path

import pandas as pd
import yaml

from urban_monitor.change.hard_examples import build_demolition_head_sampling_index


def test_demolition_head_smoke_config_is_strictly_isolated():
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "configs" / "change_public_s2looking_demolition_head_smoke.yaml").read_text(encoding="utf-8"))
    assert config["training"]["demolition_head_only"] is True
    assert config["training"]["epochs"] == 1
    assert config["training"]["learning_rate"] == 1e-5
    assert config["loss"]["stage_weights"]["directional"] == {"binary": 0.0, "new_building": 0.0, "demolition": 1.0, "semantic": 0.0}
    assert config["model_selection"]["source_thresholds"] == {"binary": .50, "new": .60, "demolition": .30}


def test_demolition_sampler_uses_train_only_balanced_subset(tmp_path: Path):
    run = tmp_path / "run"; (run / "prepared").mkdir(parents=True)
    rows = ([{"sample_id": f"train-{i:04d}", "split": "train", "has_binary": "true"} for i in range(100)]
            + [{"sample_id": "val", "split": "validation", "has_binary": "true"}, {"sample_id": "test", "split": "test", "has_binary": "true"}])
    pd.DataFrame(rows).to_csv(run / "prepared" / "public_change_index.csv", index=False)
    hard = pd.DataFrame(
        [{"sample_id": f"train-{i:04d}", "target_head": "demolition"} for i in range(6)]
        + [{"sample_id": f"train-{i + 6:04d}", "target_head": "both"} for i in range(30)]
        + [{"sample_id": "train-0090", "target_head": "new"}, {"sample_id": "val", "target_head": "demolition"}, {"sample_id": "test", "target_head": "demolition"}]
    )
    hard_path = tmp_path / "hard.csv"; hard.to_csv(hard_path, index=False)
    path, summary = build_demolition_head_sampling_index(run, hard_path, max_hard_fraction=.10, max_both_fraction_within_hard=.40)
    output = pd.read_csv(path)
    assert summary["hard_fraction"] <= .10
    assert summary["both_selected"] <= 4
    assert float(output.loc[output.sample_id == "train-0000", "sample_weight"].iloc[0]) == 2.0
    assert float(output.loc[output.sample_id == "train-0006", "sample_weight"].iloc[0]) == 1.25
    assert float(output.loc[output.sample_id == "train-0090", "sample_weight"].iloc[0]) == 1.0
    assert float(output.loc[output.sample_id == "val", "sample_weight"].iloc[0]) == 1.0
    assert float(output.loc[output.sample_id == "test", "sample_weight"].iloc[0]) == 1.0


def test_only_demolition_head_is_trainable_when_torch_is_available():
    import pytest
    torch = pytest.importorskip("torch")
    from urban_monitor.change.models import build_change_model
    from urban_monitor.change.training import configure_demolition_head_only, set_demolition_head_training_mode
    model = build_change_model({"pretrained": False, "semantic_classes": 2})
    trainable = configure_demolition_head_only(model)
    model.train()
    set_demolition_head_training_mode(model)
    assert model.training is False
    assert model.demolition_head.training is True
    assert model.stem.training is False
    assert trainable and all(name.startswith("demolition_head.") for name, _ in trainable)
    assert not any(parameter.requires_grad for name, parameter in model.named_parameters() if not name.startswith("demolition_head."))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=1e-5)
    assert all(parameter.requires_grad for group in optimizer.param_groups for parameter in group["params"])
    for _, parameter in trainable:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert all(torch.equal(parameter, before[name]) for name, parameter in model.named_parameters() if name in before)
