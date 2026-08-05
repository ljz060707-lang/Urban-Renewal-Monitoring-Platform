import numpy as np

from surface_state_v1.training.metrics import Confusion


def test_ignore_pixels_do_not_enter_metrics():
    confusion = Confusion(2)
    confusion.update(np.array([[0, 1]]), np.array([[0, 255]]))
    assert confusion.matrix.sum() == 1
    assert np.isfinite(confusion.report(["a", "b"])["macro_mIoU"])
