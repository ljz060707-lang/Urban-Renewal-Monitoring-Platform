import torch

from urban_monitor.change.losses import boundary_target


def test_boundary_target_only_marks_edge_band():
    mask = torch.zeros(1, 1, 12, 12)
    mask[:, :, 3:9, 3:9] = 1
    edge = boundary_target(mask, 1)
    assert edge.sum() > 0
    assert edge[0, 0, 5, 5] == 0

