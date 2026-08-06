from urban_monitor.compat import create_grad_scaler, grad_scaler_class


def test_grad_scaler_uses_available_native_class_without_global_patch():
    scaler = create_grad_scaler("cpu", enabled=False)
    assert isinstance(scaler, grad_scaler_class())
    assert scaler.is_enabled() is False
