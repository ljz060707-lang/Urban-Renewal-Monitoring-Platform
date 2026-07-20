from __future__ import annotations

import platform

import pytest


@pytest.mark.train_smoke
def test_multitask_siamese_unet_forward_shapes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from urban_monitor.change.models import build_change_model

    model = build_change_model({"backend": "multitask_siamese_unet", "semantic_classes": 7, "pretrained": False})
    model.eval()
    with torch.no_grad():
        outputs = model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
    assert set(outputs) == {"binary_change", "new_building", "demolition", "semantic_t1", "semantic_t2", "uncertainty"}
    assert outputs["binary_change"].shape == (1, 1, 64, 64)
    assert outputs["semantic_t1"].shape == (1, 7, 64, 64)


def test_pretrained_never_downloads_silently():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from urban_monitor.change.models import build_change_model
    with pytest.raises(ValueError, match="automatic network download is disabled"):
        build_change_model({"pretrained": True})


def test_changemamba_has_clear_platform_gate():
    from urban_monitor.change.models.changemamba_adapter import build_changemamba
    if platform.system() != "Linux":
        with pytest.raises(RuntimeError, match="Linux/WSL"):
            build_changemamba({})
