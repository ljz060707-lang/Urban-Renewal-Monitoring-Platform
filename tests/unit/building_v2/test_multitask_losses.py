import torch

from urban_monitor.building_foundation_v2.losses import multitask_loss
from urban_monitor.building_foundation_v2.model import BuildingFoundationNet


def test_multitask_loss_finite():
    model = BuildingFoundationNet(base_channels=4)
    image = torch.rand(2, 3, 32, 32)
    outputs = model(image)
    batch = {"mask": torch.randint(0, 2, (2, 1, 32, 32)).float(), "boundary": torch.rand(2, 1, 32, 32), "distance": torch.rand(2, 1, 32, 32)}
    parts = multitask_loss(outputs, batch)
    assert torch.isfinite(parts["total"])
