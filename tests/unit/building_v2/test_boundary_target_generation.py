import numpy as np

from urban_monitor.building_foundation_v2.boundary import boundary_object_stats, boundary_target


def test_boundary_target_widths_are_binary_and_preserve_small_object_signal():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    mask[12:25, 12:25] = 1
    for width in (1, 2, 3):
        target = boundary_target(mask, width)
        assert target.shape == mask.shape
        assert set(np.unique(target)).issubset({0.0, 1.0})
        assert target[2:5, 2:5].sum() > 0
    stats = boundary_object_stats(mask, 2)
    assert stats["object_count"] == 2
    assert stats["boundary_positive_fraction"] > 0


def test_ignore_pixels_are_not_boundary_or_background_supervision():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    mask[:2, :] = 255
    target, valid = boundary_target(mask, 2, return_valid=True)
    assert not valid[:2, :].any()
    assert not target[:2, :].any()
