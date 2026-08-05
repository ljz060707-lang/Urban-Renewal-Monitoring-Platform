import torch

from surface_state_v1.models.tfr import TemporalFeatureRefinement


def test_tfr_receives_finite_gradients():
    module = TemporalFeatureRefinement(24, 4)
    output = module(torch.rand(1, 2, 24, 8, 8))["refined_features"][0]
    output.mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
