from urban_monitor.compat import create_grad_scaler


def test_grad_scaler_disabled_contract_is_stable():
    scaler = create_grad_scaler("cpu", enabled=False)
    assert scaler.is_enabled() is False
