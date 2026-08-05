import torch

from urban_monitor.building_foundation_v2.model import BuildingFoundationNet


def test_static_outputs_and_feature_bundle_are_finite():
    model = BuildingFoundationNet(base_channels=4)
    outputs = model(torch.rand(2, 3, 64, 64))
    for key in ("building_probability", "boundary_probability", "quality_score", "uncertainty_score"):
        assert outputs[key].shape == (2, 1, 64, 64)
        assert torch.isfinite(outputs[key]).all()
        assert float(outputs[key].min()) >= 0 and float(outputs[key].max()) <= 1
    assert outputs["height_probability_or_value"] is None


def test_height_head_is_explicit():
    model = BuildingFoundationNet(base_channels=4, height_enabled=True)
    assert model(torch.rand(1, 3, 32, 32))["height_probability_or_value"].shape == (1, 1, 32, 32)
