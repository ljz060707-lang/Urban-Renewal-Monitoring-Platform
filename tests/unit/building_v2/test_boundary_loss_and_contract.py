import torch

from urban_monitor.building_foundation_v2.boundary import boundary_loss


def test_boundary_bce_dice_loss_has_gradient_and_is_finite():
    logits = torch.zeros((1, 1, 8, 8), requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 2:6, 2:6] = 1
    value = boundary_loss(logits, target)
    value.backward()
    assert torch.isfinite(value)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

