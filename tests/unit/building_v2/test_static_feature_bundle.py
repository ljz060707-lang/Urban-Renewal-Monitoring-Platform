import torch

from urban_monitor.building_foundation_v2.model import BuildingFoundationNet


def test_feature_bundle_keys():
    model = BuildingFoundationNet(base_channels=4)
    outputs = model(torch.rand(1, 3, 16, 16))
    bundle = model.export_feature_bundle(outputs)
    assert set(bundle) == {"feature_tensor", "presence_probability", "boundary_probability", "height_probability_or_value", "quality_score", "uncertainty_score"}
