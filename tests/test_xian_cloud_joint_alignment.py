"""Synthetic only: numerical interface, geometry, gradients and strict restore.

No datasets, imagery, GT files, frozen alignment imports or training loops.
"""
import copy
import inspect
import io
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from xian_r5.cloud_joint.alignment import (
    AlignmentConfig, AlignmentPredictor, COORDINATES, TARGET_SEMANTICS,
    alignment_checkpoint, known_transform_loss, load_alignment_checkpoint,
    pixel_shift_to_map, real_weak_alignment_loss, reference_map_loss, translate_and_aggregate,
    translate_vertices,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(913)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(previous)


def scene(batch=2, count=2):
    masks = torch.zeros(batch, count, 24, 32)
    if count:
        masks[:, 0, 6:11, 5:10] = 1
    if count > 1:
        masks[:, 1, 13:17, 19:25] = 1
    hr = torch.rand(batch, 3, 24, 32) * .2 + masks.sum(1, keepdim=True) * .6
    lr = F.avg_pool2d(hr, 4)
    return hr, lr, masks


def model():
    return AlignmentPredictor(AlignmentConfig(width=8, context_size=12,
                                              max_source_shift_px=4., max_residual_shift_px=2.))


def test_forward_is_numeric_bounded_and_inference_conditioned():
    net = model().eval()
    hr, lr, masks = scene()
    original = masks.clone()
    p = net(hr, lr, masks)
    assert p["source_shift_px"].shape == (2, 2)
    assert p["instance_residual_px"].shape == (2, 2, 2)
    assert p["total_shift_px"].shape == (2, 2, 2)
    assert p["unmatched_logits"].shape == (2, 2)
    assert torch.isfinite(p["total_shift_px"]).all()
    assert p["source_shift_px"].abs().max() <= 4
    assert p["instance_residual_px"].abs().max() <= 2
    torch.testing.assert_close(p["total_shift_px"], p["source_shift_px"][:, None] + p["instance_residual_px"])
    assert not torch.equal(p["total_shift_px"], net(hr, 1 - lr, masks)["total_shift_px"])
    assert not torch.equal(p["total_shift_px"], net(1 - hr, lr, masks)["total_shift_px"])
    torch.testing.assert_close(masks, original, rtol=0, atol=0)
    assert p["target_semantics"] == TARGET_SEMANTICS
    assert p["coordinate_convention"] == COORDINATES
    assert p["application_status"] == "CANDIDATE_NOT_ADMITTED"
    assert p["uncertainty_status"] == "UNCALIBRATED"
    assert "total_dx_m" not in p


def test_gt_free_forward_and_no_file_access():
    net = model()
    hr, lr, masks = scene()
    def deny(*args, **kwargs):
        raise AssertionError("Inference file access forbidden")
    with patch("builtins.open", deny), patch("io.open", deny), patch("os.open", deny):
        output = net(hr, lr, masks)
    assert torch.isfinite(output["unmatched_logits"]).all()
    parameters = set(inspect.signature(net.forward).parameters)
    assert parameters == {"hr_rgb", "lr_image", "instance_masks", "source_valid", "lr_valid"}
    for key in ("gt2019", "gt_path", "gt_boxes", "cached_offset", "checkpoint", "lr_building_tif"):
        with pytest.raises(TypeError):
            net(hr, lr, masks, **{key: "forbidden"})
    with pytest.raises(ValueError):
        net("gt/path.tif", lr, masks)


def test_empty_padding_and_all_missing_support_are_explicit():
    net = model()
    hr, lr, masks = scene(count=3)
    p = net(hr, lr, masks, source_valid=torch.zeros_like(hr[:, :1]),
            lr_valid=torch.zeros_like(lr[:, :1], dtype=torch.bool))
    assert not p["instance_valid"][:, 2].any()
    assert p["source_support_incomplete"].all()
    assert p["reference_support_missing"].all()
    assert p["ood_risk"].all()
    hr, lr, masks = scene(count=0)
    p = net(hr, lr, masks)
    assert p["total_shift_px"].shape == (2, 0, 2)
    assert p["unmatched_probability"].shape == (2, 0)
    r = translate_and_aggregate(masks, p["total_shift_px"], factor=4)
    assert r["fractions"].shape == (2, 1, 6, 8)
    assert torch.equal(r["union"], torch.zeros_like(r["union"]))
    with pytest.raises(ValueError, match="NO_VALID"):
        known_transform_loss(p, torch.zeros_like(p["total_shift_px"]), torch.zeros(2, 0, dtype=torch.bool))


def test_synthetic_known_transform_backward_reaches_both_encoders_and_heads():
    net = model()
    hr, lr, masks = scene()
    p = net(hr, lr, masks)
    # Known perturbations in target-canvas pixels; no real-world accuracy claim.
    target = torch.tensor([[[1., -1.], [-1., .5]], [[.5, 1.], [1., -1.]]])
    matched = torch.tensor([[True, True], [True, False]])
    losses = known_transform_loss(p, target, matched)
    losses["loss"].backward()
    assert losses["supervised_count"].item() == 4
    assert losses["matched_count"].item() == 3
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for module in (net.hr_encoder, net.lr_encoder, net.fusion, net.source_head, net.instance_head):
        assert sum(float(v.grad.abs().sum()) for v in module.parameters()) > 0


@pytest.mark.parametrize("shift", [(2., 3.), (-2., -3.), (0., 0.)])
def test_forward_pixel_sign_integer_shape_and_area(shift):
    _, _, masks = scene(batch=1, count=1)
    offsets = torch.tensor([[shift]])
    r = translate_and_aggregate(masks, offsets, factor=4)
    expected = torch.roll(masks, shifts=(int(shift[1]), int(shift[0])), dims=(-2, -1))
    torch.testing.assert_close(r["moved_masks"], expected, atol=2e-6, rtol=0)
    torch.testing.assert_close(r["input_area_px2"], r["raster_area_px2"], atol=1e-5, rtol=0)
    assert not r["support_overflow"].any()
    torch.testing.assert_close(r["fractions"].sum() * 16, masks.sum(), atol=1e-5, rtol=0)


def test_fractional_translation_derivative_and_no_scale_or_double_move():
    _, _, masks = scene(batch=1, count=1)
    offsets = torch.tensor([[[.3, -.4]]], requires_grad=True)
    r = translate_and_aggregate(masks, offsets, factor=4)
    columns = torch.arange(32, dtype=torch.float32)
    centroid_x = (r["union"].sum(-2) * columns).sum() / masks.sum()
    centroid_x.backward()
    torch.testing.assert_close(offsets.grad[0, 0], torch.tensor([1., 0.]), atol=1e-5, rtol=0)
    torch.testing.assert_close(r["raster_area_px2"], r["input_area_px2"], atol=1e-5, rtol=0)
    assert not r["support_overflow"].any()
    assert torch.autograd.gradcheck(
        lambda d: translate_and_aggregate(masks.double(), d)["union"],
        (offsets.detach().double().requires_grad_(),), eps=1e-5, atol=1e-4)


def test_vertices_preserve_whole_geometry_and_affine_sign():
    vertices = torch.tensor([[[[1., 2.], [5., 2.], [5., 7.], [1., 7.]]]])
    shift = torch.tensor([[[3., 4.]]])
    moved = translate_vertices(vertices, shift)
    torch.testing.assert_close(moved - moved.roll(1, -2), vertices - vertices.roll(1, -2))
    torch.testing.assert_close(moved - vertices, shift.unsqueeze(-2).expand_as(vertices))
    result = pixel_shift_to_map(shift, torch.tensor([[.5, 0.], [0., -.5]]), map_unit="metre")
    torch.testing.assert_close(result["vector"], torch.tensor([[[1.5, -2.]]]))
    assert result["accuracy_status"] == "NOT_CERTIFIED"
    result = pixel_shift_to_map(shift, torch.tensor([[1., .2], [-.1, -1.]]), map_unit="unknown_map_unit")
    assert result["unit"] == "unknown_map_unit"
    with pytest.raises(ValueError):
        pixel_shift_to_map(shift, torch.eye(2), map_unit="assumed_metres")


def test_union_avoids_duplicate_area_and_keeps_unknown_denominator():
    _, _, masks = scene(batch=1, count=1)
    masks = masks.expand(-1, 2, -1, -1)
    target_valid = torch.ones(1, 1, 24, 32)
    target_valid[:, :, :, :8] = 0
    r = translate_and_aggregate(masks, torch.zeros(1, 2, 2), factor=4, target_valid=target_valid)
    torch.testing.assert_close(r["union"].sum(), masks[:, 0].sum())
    torch.testing.assert_close(r["collision_mass_px2"], masks[:, 0].sum().reshape(1))
    assert r["support_overflow"].all()
    assert r["unknown_fraction"].sum() == 12
    assert torch.equal(r["valid_fraction"] + r["unknown_fraction"], torch.ones_like(r["valid_fraction"]))


def test_overflow_and_extreme_predictions_are_flagged_not_precise_clamps():
    _, _, masks = scene(batch=1, count=1)
    r = translate_and_aggregate(masks, torch.tensor([[[100., -100.]]]))
    assert r["support_overflow"].all()
    torch.testing.assert_close(r["lost_area_px2"], masks.sum((-2, -1)))
    net = model()
    payload = alignment_checkpoint(net)
    payload["state_dict"]["source_head.2.bias"].fill_(12)
    net = load_alignment_checkpoint(payload)
    hr, lr, masks = scene()
    p = net(hr, lr, masks)
    assert p["near_training_bound"].all()
    assert p["ood_risk"].all()
    assert p["domain_ood_status"] == "NOT_ASSESSED"


def test_all_unmatched_cannot_disable_shift_or_map_supervision():
    hr, lr, masks = scene(batch=1)
    p = model()(hr, lr, masks)
    labels = torch.zeros_like(p["total_shift_px"])
    matched = torch.ones(1, 2, dtype=torch.bool)
    good = dict(p, unmatched_logits=torch.full((1, 2), -20.))
    bad = dict(p, unmatched_logits=torch.full((1, 2), 20., requires_grad=True))
    a, b = known_transform_loss(good, labels, matched), known_transform_loss(bad, labels, matched)
    assert b["loss"] > a["loss"] + 10
    torch.testing.assert_close(a["known_transform"], b["known_transform"])
    r = translate_and_aggregate(masks, p["total_shift_px"], factor=4)
    target = masks.amax(1, keepdim=True)
    loss = reference_map_loss(r, target, torch.ones_like(target))["loss"]
    assert r["union"].sum() > 40
    bad["unmatched_logits"].grad = None
    (b["loss"] + loss).backward()
    assert bad["unmatched_logits"].grad.sum() > 0
    all_absent = known_transform_loss(p, labels, torch.zeros_like(matched))
    assert all_absent["known_transform"] is None
    assert all_absent["matched_count"] == 0


def test_map_loss_correct_alignment_beats_wrong_and_empty_and_backpropagates():
    _, _, masks = scene(batch=1, count=1)
    known = torch.tensor([[[1.25, -.5]]])
    target = translate_and_aggregate(masks, known)["union"].detach()
    valid = torch.ones_like(target)
    correct = reference_map_loss(translate_and_aggregate(masks, known, factor=4), target, valid)
    wrong_shift = torch.tensor([[[-.4, .3]]], requires_grad=True)
    wrong = reference_map_loss(translate_and_aggregate(masks, wrong_shift, factor=4), target, valid)
    empty = reference_map_loss(translate_and_aggregate(masks * 0, known, factor=4), target, valid)
    assert correct["loss"] < 1e-5
    assert wrong["loss"] > correct["loss"]
    assert empty["loss"] > correct["loss"]
    wrong["loss"].backward()
    assert torch.isfinite(wrong_shift.grad).all() and wrong_shift.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="NO_VALID"):
        reference_map_loss(translate_and_aggregate(masks, known), target, valid * 0)
    with pytest.raises(ValueError, match="coverage"):
        reference_map_loss(translate_and_aggregate(masks, known), target, None)


def test_joint_reference_map_backward_reaches_images_and_model():
    hr, lr, masks = scene(batch=1)
    hr.requires_grad_()
    lr.requires_grad_()
    masks.requires_grad_()
    net = model()
    p = net(hr, lr, masks)
    r = translate_and_aggregate(masks, p["total_shift_px"], factor=4)
    target = translate_and_aggregate(masks.detach(), torch.ones(1, 2, 2))["union"]
    reference_map_loss(r, target, torch.ones_like(target))["loss"].backward()
    for tensor in (hr, lr, masks):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    assert net.source_head[-1].weight.grad.abs().sum() > 0
    assert net.instance_head[-1].weight.grad.abs().sum() > 0


def test_checkpoint_strict_roundtrip_bitwise_and_rng_preservation():
    net = model().eval()
    hr, lr, masks = scene()
    buffer = io.BytesIO()
    torch.save(alignment_checkpoint(net), buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location="cpu", weights_only=True)
    rng = torch.get_rng_state().clone()
    restored = load_alignment_checkpoint(payload, expected_config=net.config)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not restored.training
    for key, value in net(hr, lr, masks).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, restored(hr, lr, masks)[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="config"):
        load_alignment_checkpoint(payload, expected_config=AlignmentConfig())


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype", "nan", "semantics", "coords", "config", "metadata"])
def test_checkpoint_rejects_corruption(corruption):
    payload = copy.deepcopy(alignment_checkpoint(model()))
    state = payload["state_dict"]
    key = next(iter(state))
    if corruption == "missing":
        del state[key]
    elif corruption == "extra":
        state["gt_cached_vectors"] = torch.zeros(2)
    elif corruption == "shape":
        state[key] = torch.zeros(1)
    elif corruption == "dtype":
        state[key] = state[key].double()
    elif corruption == "nan":
        state[key].flatten()[0] = float("nan")
    elif corruption == "semantics":
        payload["target_semantics"] = "ROOF_TO_GROUND_FOOTPRINT"
    elif corruption == "coords":
        payload["coordinates"] = "METRES_EAST_NORTH"
    elif corruption == "config":
        payload["config"]["max_source_shift_px"] = float("inf")
    else:
        payload["gt_path"] = "forbidden"
    with pytest.raises(ValueError):
        load_alignment_checkpoint(payload)


@pytest.mark.parametrize("case", ["nan", "range", "shape", "dtype", "validity", "factor"])
def test_invalid_inputs_fail_explicitly(case):
    hr, lr, masks = scene()
    net = model()
    with pytest.raises(ValueError):
        if case == "nan":
            hr[0, 0, 0, 0] = float("nan")
            net(hr, lr, masks)
        elif case == "range":
            net(hr * 255, lr, masks)
        elif case == "shape":
            net(hr, lr, masks[..., :-1])
        elif case == "dtype":
            net(hr, lr.double(), masks)
        elif case == "validity":
            net(hr, lr, masks, source_valid=torch.full((2, 1, 24, 32), .5))
        else:
            translate_and_aggregate(masks, torch.zeros(2, 2, 2), factor=5)


def test_fresh_process_gt_free_forward():
    code = """
import torch
from unittest.mock import patch
from xian_r5.cloud_joint.alignment import AlignmentPredictor, AlignmentConfig
torch.set_num_threads(1)
model = AlignmentPredictor(AlignmentConfig(width=8, context_size=8)).eval()
def deny(*a, **k): raise AssertionError('GT or file read')
with patch('builtins.open', deny), patch('io.open', deny), patch('os.open', deny):
    result = model(torch.ones(1,3,8,8), torch.ones(1,3,2,2), torch.ones(1,1,8,8))
assert torch.isfinite(result['total_shift_px']).all()
print('SYNTHETIC_GT_FREE_PROCESS_PASS')
"""
    run = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=45)
    assert run.returncode == 0, run.stderr
    assert "SYNTHETIC_GT_FREE_PROCESS_PASS" in run.stdout


def weak_scene(shifts=None, count=2):
    hr, lr, masks = scene(batch=1, count=count)
    p = model()(hr, lr, masks)
    if shifts is not None:
        p = dict(p, total_shift_px=shifts)
    r = translate_and_aggregate(masks, p["total_shift_px"], factor=4)
    return p, r, masks.amax(1, keepdim=True) if count else masks.sum(1, keepdim=True)


def test_weak_positive_only_coverage_does_not_invent_background_or_geometric_support():
    p, r, target = weak_scene(torch.zeros(1, 2, 2))
    coverage = target.bool()
    unknown_changed = torch.where(coverage, target, torch.ones_like(target))
    a = real_weak_alignment_loss(p, r, target, coverage)
    b = real_weak_alignment_loss(p, r, unknown_changed, coverage)
    torch.testing.assert_close(a["loss"], b["loss"], atol=0, rtol=0)
    for name in ("reference_map", "map_l1", "map_dice", "fraction_l1"):
        torch.testing.assert_close(a["components"][name], b["components"][name], atol=0, rtol=0)
    assert a["counters"]["valid_pixel_count"] == 49
    assert a["counters"]["confirmed_background_pixel_count"] == 0
    assert a["counters"]["unknown_label_pixel_count"] == 24 * 32 - 49
    assert a["stable_track_status"] == "NOT_AVAILABLE"
    assert a["components"]["stable_track"] is None
    # Label coverage must not masquerade as canvas clipping. A second instance
    # outside positive-only GT still exists and remains in the original count.
    only_first = torch.zeros_like(coverage)
    only_first[:, :, 6:11, 5:10] = True
    masks = scene(batch=1)[2]
    restricted = translate_and_aggregate(masks, p["total_shift_px"], factor=4, target_valid=only_first)
    assert restricted["lost_area_px2"].sum() > 0
    c = real_weak_alignment_loss(p, restricted, target, only_first)
    assert c["components"]["support_overflow"] == 0
    assert c["counters"]["original_instance_count"] == 2


def test_weak_geometry_terms_have_original_denominators_and_recovery_gradients():
    shifts = torch.tensor([[[100., 100.], [100., 100.]]], requires_grad=True)
    p, r, target = weak_scene(shifts)
    result = real_weak_alignment_loss(p, r, target, target.bool())
    assert result["components"]["support_overflow"] == 1
    assert result["components"]["total_shift_l2"] == 10000
    assert result["counters"]["canvas_overflow_instance_count"] == 2
    assert result["counters"]["original_instance_mass_px2"] == 49
    result["loss"].backward()
    assert shifts.grad.abs().sum() > 0 and torch.isfinite(shifts.grad).all()
    # Identical input instances are geometric conflicts even outside GT coverage.
    masks = scene(batch=1, count=1)[2].expand(-1, 2, -1, -1)
    shifts = torch.zeros(1, 2, 2)
    p = dict(p, total_shift_px=shifts)
    r = translate_and_aggregate(masks, shifts, factor=4)
    target = masks.amax(1, keepdim=True)
    collision = real_weak_alignment_loss(p, r, target, target.bool())
    torch.testing.assert_close(collision["components"]["collision"], torch.tensor(.5))
    assert collision["counters"]["original_instance_mass_px2"] == 50


def test_weak_all_unmatched_penalty_preserves_original_pixel_and_object_denominators():
    p, r, target = weak_scene()
    good = dict(p, unmatched_logits=torch.full((1, 2), -20.))
    logits = torch.full((1, 2), 40., requires_grad=True)
    bad = dict(p, unmatched_logits=logits)
    valid = target.bool()
    a = real_weak_alignment_loss(good, r, target, valid)
    b = real_weak_alignment_loss(bad, r, target, valid)
    assert b["loss"] > a["loss"] + 3
    torch.testing.assert_close(a["components"]["reference_map"], b["components"]["reference_map"])
    for key in ("original_instance_count", "original_instance_mass_px2", "valid_pixel_count"):
        torch.testing.assert_close(a["counters"][key], b["counters"][key])
    b["components"]["unmatched_coverage"].backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.sum() > .9
    assert b["unmatched_constraint_status"] == "WEAK_PRIOR_NOT_MATCH_TRUTH"
    assert all(not v.requires_grad for v in b["counters"].values())
    # A fabricated acceptance subset cannot replace the original denominator.
    with pytest.raises(ValueError, match="denominator"):
        real_weak_alignment_loss(dict(p, instance_valid=torch.zeros(1, 2, dtype=torch.bool)), r, target, valid)


def test_weak_collapse_constraint_is_per_source_not_hidden_by_other_batch_items():
    hr, lr, masks = scene(batch=2)
    p = model()(hr, lr, masks)
    logits = torch.tensor([[20., 20.], [-20., -20.]], requires_grad=True)
    p = dict(p, unmatched_logits=logits)
    r = translate_and_aggregate(masks, p["total_shift_px"], factor=4)
    target = masks.amax(1, keepdim=True)
    result = real_weak_alignment_loss(p, r, target, target.bool())
    assert result["components"]["unmatched_coverage"] > 8
    result["components"]["unmatched_coverage"].backward()
    assert logits.grad[0].sum() > 0
    assert logits.grad[1].sum() == 0
    assert result["counters"]["original_instance_count_by_source"].tolist() == [2, 2]


def stable_edge(confirmed=True, same_frame=True):
    return dict(indices=torch.tensor([[0, 0, 0, 1]], dtype=torch.long),
                source_delta_px=torch.tensor([[-3., 0.]]),
                confirmed_stable=torch.tensor([confirmed]), same_target_frame=torch.tensor([same_frame]),
                weights=torch.ones(1))


def test_weak_stable_evidence_compares_corrected_positions_not_equal_offsets():
    shifts = torch.tensor([[[3., 0.], [0., 0.]]], requires_grad=True)
    p, r, target = weak_scene(shifts)
    result = real_weak_alignment_loss(p, r, target, target.bool(), stable_edge())
    assert result["stable_track_status"] == "AVAILABLE_SUPPORTED_EDGES"
    assert result["components"]["stable_track"] == 0
    assert result["counters"]["stable_edges_used"] == 1
    shifted = dict(p, total_shift_px=torch.zeros_like(shifts, requires_grad=True))
    mismatch = real_weak_alignment_loss(shifted, r, target, target.bool(), stable_edge())
    assert mismatch["components"]["stable_track"] > 0
    mismatch["components"]["stable_track"].backward()
    assert shifted["total_shift_px"].grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["unconfirmed", "wrong_frame", "zero_weight", "empty", "padding"])
def test_weak_missing_stable_evidence_is_none_not_zero(kind):
    p, r, target = weak_scene()
    edges = stable_edge()
    if kind == "unconfirmed":
        edges["confirmed_stable"][:] = False
    elif kind == "wrong_frame":
        edges["same_target_frame"][:] = False
    elif kind == "zero_weight":
        edges["weights"][:] = 0
    elif kind == "empty":
        edges = {k: v[:0] for k, v in edges.items()}
    else:
        hr, lr, masks = scene(batch=1, count=3)
        p = model()(hr, lr, masks)
        r = translate_and_aggregate(masks, p["total_shift_px"], factor=4)
        target = masks.amax(1, keepdim=True)
        edges["indices"][0, 3] = 2
    result = real_weak_alignment_loss(p, r, target, target.bool(), edges)
    assert result["stable_track_status"] == "NOT_AVAILABLE"
    assert result["components"]["stable_track"] is None
    assert result["counters"]["stable_edges_used"] == 0


@pytest.mark.parametrize("kind", ["none", "empty", "learned", "out_of_range", "self_edge", "schema"])
def test_weak_rejects_unusable_coverage_and_malformed_evidence(kind):
    p, r, target = weak_scene()
    valid = target.bool()
    edges = None
    if kind == "none":
        valid = None
    elif kind == "empty":
        valid = torch.zeros_like(valid)
    elif kind == "learned":
        valid = valid.float().requires_grad_()
    else:
        edges = stable_edge()
        if kind == "out_of_range":
            edges["indices"][0, 0] = 100
        elif kind == "self_edge":
            edges["indices"][0, 3] = 0
        else:
            del edges["confirmed_stable"]
    with pytest.raises(ValueError):
        real_weak_alignment_loss(p, r, target, valid, edges)


def test_weak_empty_instance_scene_reports_unavailable_priors_with_map_loss():
    p, r, target = weak_scene(count=0)
    result = real_weak_alignment_loss(p, r, target, torch.ones_like(target, dtype=torch.bool))
    assert result["counters"]["original_instance_count"] == 0
    assert result["components"]["unmatched_coverage"] is None
    assert result["unmatched_constraint_status"] == "NOT_AVAILABLE"
    assert result["components"]["stable_track"] is None
    torch.testing.assert_close(result["loss"], result["components"]["reference_map"])


def test_weak_feature256_bounds_and_known_content_shift_loss_backward_no_files():
    config = AlignmentConfig(max_source_shift_px=12., max_residual_shift_px=4.)
    net = AlignmentPredictor(config)
    target_instances = torch.zeros(1, 2, 256, 256)
    target_instances[0, 0, 60:76, 80:96] = 1
    target_instances[0, 1, 140:160, 170:190] = 1
    # Content AND instance masks move together: common +/-6 plus residual +/-2.
    content_shift = torch.tensor([[[8., -4.], [4., -8.]]])
    source_instances = translate_and_aggregate(target_instances, content_shift)["moved_masks"]
    hr = source_instances.amax(1, keepdim=True).expand(-1, 3, -1, -1).contiguous()
    target = target_instances.amax(1, keepdim=True)
    lr = F.avg_pool2d(target.expand(-1, 3, -1, -1), 16)
    def deny(*args, **kwargs):
        raise AssertionError("No data or file reads in weak loss")
    with patch("builtins.open", deny), patch("io.open", deny), patch("os.open", deny):
        p = net(hr, lr, source_instances)
        r = translate_and_aggregate(source_instances, p["total_shift_px"], factor=16)
        weak = real_weak_alignment_loss(p, r, target, target.bool())
        synthetic = known_transform_loss(p, -content_shift, torch.ones(1, 2, dtype=torch.bool))
        (weak["loss"] + synthetic["loss"]).backward()
    assert p["source_shift_px"].abs().max() <= 12
    assert p["instance_residual_px"].abs().max() <= 4
    for module in (net.hr_encoder, net.lr_encoder, net.source_head, net.instance_head):
        assert sum(v.grad.abs().sum().item() for v in module.parameters()) > 0
        assert all(torch.isfinite(v.grad).all() for v in module.parameters())
    assert weak["supervision_status"] == "SCENE_WEAK_SUPERVISION_NOT_INSTANCE_DISPLACEMENT_TRUTH"
