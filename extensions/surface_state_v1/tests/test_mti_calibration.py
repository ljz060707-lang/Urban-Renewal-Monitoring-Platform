import torch

from surface_state_v1.models.mti import MultiTaskIntegration


def test_mti_output_length_matches_timesteps():
    result = MultiTaskIntegration()(torch.rand(1, 2, 8, 8), torch.rand(1, 1, 8, 8))
    assert result["optimal_building_sequence"].shape[1] == 2
    assert result["new_building_sequence"].shape[1] == 1
