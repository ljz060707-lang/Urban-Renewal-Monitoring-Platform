"""Synthetic-only S contract checks; no dataset or protected evaluation reads.

Run directly with --noconftest -p no:cacheprovider and python -B. Optional
real-B0 synthetic check uses XIAN_S_B0, XIAN_S_CONFIG, XIAN_S_RSBUILDING_REPO;
none are inferred and no checkpoint/data is opened unless explicitly supplied.
Serialization uses BytesIO only (no weight files created).
"""
import copy
import importlib.util
import io
import os
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
import torch.nn.functional as F

_path = Path(__file__).resolve().parents[1] / "src/xian_r5/cloud_joint/segmentation.py"
_spec = importlib.util.spec_from_file_location("xian_s_standalone", _path)
s = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = s
_spec.loader.exec_module(s)
torch.set_num_threads(2)


class SyntheticBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 64, 4, stride=4)
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.GELU())
            for _ in range(4)])
        self.norm3 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        return self.norm3(x)


class SyntheticDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = 256
        self.simple_fpn = nn.Conv2d(64, 256, 1)
        self.transformer_decoder = nn.Conv2d(256, 256, 1)
        self.building_a_embed = nn.Embedding(1, 256)
        self.building_b_embed = nn.Embedding(1, 256)
        self.cd_embed = nn.Embedding(1, 256)
        self.fusion_ab = nn.Sequential(nn.BatchNorm2d(256), nn.Conv2d(256, 256, 1))
        self.pred_head_ab = nn.Linear(256, 64)
        self.output_upscaling_ab = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 1), nn.GELU())
        self.fusion_cd = nn.Conv2d(256, 256, 1)
        self.pred_head_cd = nn.Linear(256, 64)
        self.output_upscaling_cd = nn.Conv2d(256, 64, 1)


class SyntheticB0(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SyntheticBackbone()
        self.decode_head = SyntheticDecoder()
        self.fail = False

    def _forward(self, x, samples):
        d = self.decode_head
        x = self.backbone(torch.cat((x, x), 0))
        x = d.transformer_decoder(d.simple_fpn(x))
        a, b = x.chunk(2, 0)
        a, b = d.fusion_ab(a), d.fusion_ab(b + 10)
        ma = d.output_upscaling_ab(a)
        if self.fail:
            raise RuntimeError("synthetic failure")
        mb = d.output_upscaling_ab(b)
        qa, qb = d.pred_head_ab(d.building_a_embed.weight), d.pred_head_ab(d.building_b_embed.weight)
        cd = d.output_upscaling_cd(d.fusion_cd(a - b))
        qc = d.pred_head_cd(d.cd_embed.weight)
        z = torch.cat(((ma * qa[:, :, None, None]).sum(1, keepdim=True),
                       (mb * qb[:, :, None, None]).sum(1, keepdim=True),
                       (cd * qc[:, :, None, None]).sum(1, keepdim=True)), 1)
        return F.interpolate(z, samples[0].metainfo["img_shape"], mode="bilinear", align_corners=False)


def make_model():
    torch.manual_seed(20261020)
    return s.RoofFacadeSegmenter(SyntheticB0())


def test_rgb_normalization_matches_b0_constants():
    rgb = torch.tensor([0, 127, 255], dtype=torch.uint8).view(1, 3, 1, 1)
    expected = (rgb.float() - torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)) / torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    torch.testing.assert_close(s.normalize_rgb(rgb), expected, rtol=0, atol=0)
    with pytest.raises(ValueError):
        s.normalize_rgb(rgb.float())


def test_native_roof_unchanged_and_a_feature_not_b_feature():
    model = make_model().eval()
    x = torch.randn(2, 3, 32, 32)
    features = []
    handle = model.b0.decode_head.output_upscaling_ab.register_forward_hook(
        lambda _m, _i, out: features.append(out))
    with torch.no_grad():
        native = model.b0._forward(x, [s._ImageSample({"img_shape": (32, 32)})] * 2)
    handle.remove()
    actual = model(x)
    torch.testing.assert_close(actual["roof_logits"], native[:, :1], rtol=0, atol=0)
    expected = F.interpolate(model.facade_aux(features[0]), (32, 32), mode="bilinear", align_corners=False)
    torch.testing.assert_close(actual["facade_logits"], expected, rtol=0, atol=0)
    assert not torch.equal(features[0], features[1])
    assert len(model.b0.decode_head.output_upscaling_ab._forward_hooks) == 0
    # Aux can change independently without softmax competition changing roof.
    with torch.no_grad():
        model.facade_aux.bias.add_(20)
    changed = model(x)
    torch.testing.assert_close(changed["roof_logits"], actual["roof_logits"], rtol=0, atol=0)
    assert torch.all(changed["facade_logits"] > actual["facade_logits"])


def test_hook_cleanup_on_failure_and_repeat_forward():
    model = make_model()
    model.b0.fail = True
    with pytest.raises(RuntimeError, match="synthetic failure"):
        model(torch.randn(1, 3, 32, 32))
    assert not model.b0.decode_head.output_upscaling_ab._forward_hooks
    model.b0.fail = False
    for _ in range(2):
        assert model(torch.zeros(1, 3, 32, 32))["facade_logits"].shape == (1, 1, 32, 32)
        assert not model.b0.decode_head.output_upscaling_ab._forward_hooks


@pytest.mark.parametrize("stage", ["heads", "last", "all", "frozen"])
def test_stage_gradients_frozen_buffers_and_optimizer_groups(stage):
    model = make_model().set_stage(stage).train()
    before = {k: v.clone() for k, v in model.named_buffers()}
    assert not any(m.training for m in model.b0.modules())
    out = model(torch.randn(1, 3, 32, 32, requires_grad=True))
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 7:24, 8:25] = 1
    result = s.native_segmentation_loss(out, target, facade_target=1 - target, boundary_trusted=True)
    result["loss"].backward()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert param.grad is None, name
        else:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
    for name, value in model.named_buffers():
        assert torch.equal(before[name], value), name
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any(any(token in n for token in ("fusion_cd", "pred_head_cd", "output_upscaling_cd", "cd_embed", "building_b_embed")) for n in names)
    if stage == "last":
        assert any(n.startswith("b0.backbone.stages.3.") for n in names)
        assert any(n.startswith("b0.backbone.norm3.") for n in names)
        assert not any(n.startswith("b0.backbone.stages.2.") for n in names)
    groups = model.parameter_groups()
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(names)
    if stage == "frozen":
        assert not ids
    else:
        # Synthetic single optimizer update checks only the adapter contract.
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        torch.optim.AdamW(groups).step()
        assert all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if n in frozen)


def test_masked_nan_gradients_loss_coefficients_and_boundary_denominator():
    roof = torch.randn(1, 1, 3, 3, requires_grad=True)
    facade = torch.randn_like(roof, requires_grad=True)
    target = torch.tensor([[[0, 1, 0], [1, 255, 1], [0, 1, 0]]])
    with torch.no_grad():
        roof[:, :, 1, 1] = float("nan")
        facade[:, :, 1, 1] = float("nan")
    result = s.native_segmentation_loss({"roof_logits": roof, "facade_logits": facade}, target,
                                      facade_target=target, boundary_trusted=True, facade_source_weight=.7)
    torch.testing.assert_close(result["loss"], result["roof_bce"] + result["roof_dice"] + .2 * result["roof_boundary"] + .07 * result["facade_bce"])
    assert result["boundary_valid_pairs"] == 8  # 12 adjacencies minus center's four
    assert result["roof_valid_pixels"] == result["facade_valid_pixels"] == 8
    result["loss"].backward()
    for value in (roof, facade):
        assert torch.isfinite(value.grad).all()
        assert value.grad[0, 0, 1, 1] == 0
        assert value.grad.count_nonzero() > 0


def test_all_ignore_skips_update_and_missing_facade_is_not_background():
    roof = torch.full((1, 1, 2, 2), float("nan"), requires_grad=True)
    facade = torch.full_like(roof, float("nan"), requires_grad=True)
    output = {"roof_logits": roof, "facade_logits": facade}
    ignore = torch.full((1, 2, 2), 255)
    result = s.native_segmentation_loss(output, ignore, facade_target=ignore, boundary_trusted=True)
    assert result["loss"] is None and result["skip_update"]
    assert result["roof_bce"] is result["roof_dice"] is result["facade_bce"] is result["roof_boundary"] is None
    assert roof.grad is facade.grad is None
    output["roof_logits"] = torch.zeros_like(roof, requires_grad=True)
    result = s.native_segmentation_loss(output, torch.zeros_like(ignore))
    assert result["facade_bce"] is None and result["facade_valid_pixels"] == 0
    result["loss"].backward()
    assert facade.grad is None


def test_independent_overlap_and_source_weight_zero():
    output = {k: torch.zeros(1, 1, 2, 2, requires_grad=True)
              for k in ("roof_logits", "facade_logits")}
    target = torch.ones(1, 2, 2, dtype=torch.long)
    result = s.native_segmentation_loss(output, target, facade_target=target)
    result["loss"].backward()
    assert (output["roof_logits"].grad < 0).all()
    assert (output["facade_logits"].grad < 0).all()
    output["facade_logits"] = torch.full((1, 1, 2, 2), float("nan"), requires_grad=True)
    result = s.native_segmentation_loss(output, target, facade_target=target, facade_source_weight=0)
    assert result["facade_bce"] is None and torch.isfinite(result["loss"])


def test_binary_target_and_native_boundary_contracts():
    roof, facade = s.bandon_targets(torch.tensor([0, 1, 2, 255]))
    assert roof.tolist() == [0, 1, 0, 255]
    assert facade.tolist() == [0, 0, 1, 255]
    with pytest.raises(ValueError):
        s.bandon_targets(torch.tensor([3]))
    output = {k: torch.zeros(1, 1, 2, 2, requires_grad=True) for k in ("roof_logits", "facade_logits")}
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    assert s.native_segmentation_loss(output, target)["roof_boundary"] is None
    for weight in (-1, 2, float("nan")):
        with pytest.raises(ValueError):
            s.native_segmentation_loss(output, target, facade_source_weight=weight)
    with pytest.raises(ValueError):
        s.native_segmentation_loss(output, target + 2)
    with torch.no_grad():
        output["roof_logits"][0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        s.native_segmentation_loss(output, target)


def test_strict_save_load_exact_outputs_stage_and_rejected_bad_states():
    model = make_model().set_stage("last").eval()
    sample = torch.randn(1, 3, 32, 32)
    expected = model(sample)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, map_location="cpu", weights_only=True)
    restored = make_model().eval()
    restored.load_state_dict(state, strict=True)
    assert restored.stage == "last"
    assert restored.b0.backbone.norm3.weight.requires_grad
    assert not restored.b0.backbone.stages[0][0].weight.requires_grad
    for key, value in restored(sample).items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    for change in ("missing", "unexpected", "shape", "metadata"):
        invalid = copy.deepcopy(state)
        if change == "missing":
            del invalid["facade_aux.weight"]
        elif change == "unexpected":
            invalid["not_a_parameter"] = torch.zeros(1)
        elif change == "shape":
            invalid["facade_aux.weight"] = torch.zeros(2, 64, 1, 1)
        else:
            invalid["_extra_state"]["initialization_sha256"] = "S2_FAILED_WEIGHTS"
        with pytest.raises((RuntimeError, ValueError)):
            make_model().load_state_dict(invalid, strict=True)


def test_initialization_rejects_any_other_digest_before_deserialization(monkeypatch):
    # No file created/read: a mismatching in-memory payload stands for S2/other.
    monkeypatch.setattr(s.Path, "open", lambda *_args, **_kwargs: io.BytesIO(b"not B0"))

    def forbidden_factory():
        pytest.fail("factory must not run before original B0 digest is verified")

    with pytest.raises(ValueError, match="frozen original B0"):
        s.RoofFacadeSegmenter.from_b0(forbidden_factory, "memory-only.pth")


@pytest.mark.parametrize("shape", [(1, 3, 32, 64), (1, 6, 32, 32), (1, 3, 33, 33)])
def test_reject_wrong_shape(shape):
    with pytest.raises(ValueError):
        make_model()(torch.zeros(shape))


def test_optional_real_b0_on_synthetic_rgb():
    names = ("XIAN_S_B0", "XIAN_S_CONFIG", "XIAN_S_RSBUILDING_REPO")
    if not all(os.environ.get(k) for k in names):
        pytest.skip("explicit B0/config/repo required; no automatic data access")
    if not torch.cuda.is_available():
        pytest.skip("upstream B0 positional sin/cos hardcodes FP16; needs CUDA")
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    sys.path.insert(0, os.environ["XIAN_S_RSBUILDING_REPO"])
    import opencd.models  # noqa: F401 (registry, no dataset evaluation)
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from opencd.registry import MODELS
    init_default_scope("opencd")
    config = Config.fromfile(os.environ["XIAN_S_CONFIG"])
    model = s.RoofFacadeSegmenter.from_b0(lambda: MODELS.build(config.model), os.environ["XIAN_S_B0"])
    model.cuda().eval()
    assert model.initialization_sha256 == s.B0_SHA256
    x = s.normalize_rgb(torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8, device="cuda"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        reference = model.b0._forward(x, [s._ImageSample({"img_shape": (64, 64)})])[:, :1]
        actual = model(x)
    torch.testing.assert_close(actual["roof_logits"], reference, atol=0, rtol=0)
    model.set_stage("last").train()
    before = {n: b.clone() for n, b in model.named_buffers()}
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(x)
    target = torch.zeros(1, 64, 64, dtype=torch.long, device="cuda")
    target[:, 16:48, 16:48] = 1
    loss = s.native_segmentation_loss(output, target, facade_target=1-target, boundary_trusted=True)
    loss["loss"].backward()
    names = {n: p for n, p in model.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(p.grad).all() for p in names.values())
    assert any(n.startswith("b0.backbone.stages.3.") and p.grad.count_nonzero() for n, p in names.items())
    assert any(n.startswith("b0.backbone.norm3.") and p.grad.count_nonzero() for n, p in names.items())
    assert any(n.startswith("b0.decode_head.pred_head_ab.") and p.grad.count_nonzero() for n, p in names.items())
    assert model.facade_aux.weight.grad.count_nonzero()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(torch.equal(before[n], b) for n, b in model.named_buffers())
    # Full real architecture serialization, strictly restored, still synthetic.
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        expected = model(x)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = s.RoofFacadeSegmenter(MODELS.build(config.model)).cuda().eval()
    restored.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True), strict=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        actual = restored(x)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert restored.initialization_sha256 == s.B0_SHA256
