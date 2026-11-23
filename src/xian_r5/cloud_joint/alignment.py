"""Small GT-free learned alignment candidate; no production admission implied.

All geometry is on a pre-mapped HR target canvas. Positive (dx, dy) means
right/down in pixel coordinates, not east/north or metres. HR and LR must
describe the SAME extent after external CRS/affine adaptation. One batch item
is one complete source context, never an independently fitted JPG/compute tile.
No RoMa/AnS implementation or physical parallax identification is claimed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TARGET_SEMANTICS = "REFERENCE_BUILDING_PLANIMETRY"
COORDINATES = "TARGET_CANVAS_PIXEL_DX_RIGHT_DY_DOWN"
CHECKPOINT_FORMAT = "XIAN_CLOUD_JOINT_A_V1"


@dataclass(frozen=True)
class AlignmentConfig:
    lr_channels: int = 3
    width: int = 16
    context_size: int = 32
    max_source_shift_px: float = 64.0
    max_residual_shift_px: float = 16.0

    def __post_init__(self):
        for name in ("lr_channels", "width", "context_size"):
            value = getattr(self, name)
            if type(value) is not int or value < (4 if name == "context_size" else 1):
                raise ValueError("Invalid positive integer config: " + name)
        for name in ("max_source_shift_px", "max_residual_shift_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Finite positive pixel bound required: " + name)


def _float_tensor(value, name, ndim=None):
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(name + " must be a floating tensor, never a path/manifest/cache")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(name + " has incorrect rank")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite; supply explicit validity separately")


def _probability(value, name):
    _float_tensor(value, name, 4)
    if bool(((value < 0) | (value > 1)).any()):
        raise ValueError(name + " must lie in [0,1]")


def _validity(value, shape, reference, name):
    if value is None:
        return torch.ones(shape, device=reference.device, dtype=reference.dtype)
    if not isinstance(value, Tensor) or tuple(value.shape) != tuple(shape) or value.device != reference.device:
        raise ValueError(name + " must have the documented shape/device")
    if value.dtype != torch.bool:
        _float_tensor(value, name)
        if bool(((value != 0) & (value != 1)).any()):
            raise ValueError(name + " must be bool or exactly 0/1")
    return value.to(reference.dtype)


def _geometry_inputs(masks, shifts):
    _probability(masks, "instance_masks")
    _float_tensor(shifts, "total_shift_px", 3)
    if masks.shape[0] < 1 or min(masks.shape[-2:]) < 2:
        raise ValueError("Nonempty batch and canvas at least 2x2 required")
    if shifts.shape != masks.shape[:2] + (2,) or shifts.device != masks.device or shifts.dtype != masks.dtype:
        raise ValueError("Shift shape/device/dtype must agree with masks")


def _canvas_overflow(masks, shifts):
    """Full nonzero pixel support, including fractional translation at edges."""
    h, w = masks.shape[-2:]
    x = torch.arange(w, device=masks.device, dtype=masks.dtype)
    y = torch.arange(h, device=masks.device, dtype=masks.dtype)
    occupied_x, occupied_y = (masks > 0).any(-2), (masks > 0).any(-1)
    xmin = torch.where(occupied_x, x, float("inf")).amin(-1)
    xmax = torch.where(occupied_x, x, -float("inf")).amax(-1)
    ymin = torch.where(occupied_y, y, float("inf")).amin(-1)
    ymax = torch.where(occupied_y, y, -float("inf")).amax(-1)
    dx, dy = shifts.unbind(-1)
    return (xmin + dx < 0) | (xmax + dx > w - 1) | (ymin + dy < 0) | (ymax + dy > h - 1)


class AlignmentPredictor(nn.Module):
    """GT-free numerical forward with one common translation per source context.

    Inputs: hr_rgb [B,3,H,W], lr_image [B,C,h,w], both normalized to [0,1];
    instance_masks [B,N,H,W] are native inference candidates (not GT boxes).
    Optional validity is [B,1,H,W]/[B,1,h,w]. Zero masks are padding, explicitly
    flagged. N=0 is supported. No optional LR prediction prior is consumed.

    Outputs are candidates even with random initialization or a loaded state.
    Caller owns source grouping, observation IDs, dates, transform ledger,
    complete owner geometry/halo and independent validation/admission.
    """

    def __init__(self, config: AlignmentConfig = AlignmentConfig()):
        super().__init__()
        if not isinstance(config, AlignmentConfig):
            raise TypeError("AlignmentConfig required")
        self.config = config
        c = config.width
        self.hr_encoder = nn.Sequential(nn.Conv2d(4, c, 3, padding=1), nn.SiLU())
        self.lr_encoder = nn.Sequential(nn.Conv2d(config.lr_channels, c, 3, padding=1), nn.SiLU())
        self.fusion = nn.Sequential(nn.Conv2d(c * 4 + 2, c, 3, padding=1), nn.SiLU())
        self.source_head = nn.Sequential(nn.Linear(c * 16, c * 2), nn.SiLU(), nn.Linear(c * 2, 2))
        self.instance_head = nn.Sequential(nn.Linear(c * 2 + 3, c * 2), nn.SiLU(), nn.Linear(c * 2, 3))
        # Small nonzero weights preserve gradients through both image encoders.
        with torch.no_grad():
            for head in (self.source_head[-1], self.instance_head[-1]):
                head.weight.mul_(0.01)
                head.bias.zero_()

    def forward(self, hr_rgb, lr_image, instance_masks, *, source_valid=None, lr_valid=None):
        _probability(hr_rgb, "hr_rgb")
        _probability(lr_image, "lr_image")
        _probability(instance_masks, "instance_masks")
        b, _, h, w = hr_rgb.shape
        if b < 1 or min(h, w, *lr_image.shape[-2:]) < 2:
            raise ValueError("Nonempty batch and image dimensions >=2 required")
        if hr_rgb.shape[1] != 3 or lr_image.shape[:2] != (b, self.config.lr_channels):
            raise ValueError("Incorrect HR/LR channels or batch")
        if instance_masks.shape[0] != b or instance_masks.shape[-2:] != (h, w):
            raise ValueError("Instance masks must use the HR target canvas")
        for value in (lr_image, instance_masks):
            if value.device != hr_rgb.device or value.dtype != hr_rgb.dtype:
                raise ValueError("Input tensors must share dtype/device")
        sv = _validity(source_valid, (b, 1, h, w), hr_rgb, "source_valid")
        lv = _validity(lr_valid, (b, 1, *lr_image.shape[-2:]), lr_image, "lr_valid")
        n = instance_masks.shape[1]
        union = instance_masks.amax(1, keepdim=True) if n else hr_rgb[:, :1] * 0
        size = (self.config.context_size,) * 2
        hr = self.hr_encoder(F.adaptive_avg_pool2d(torch.cat((hr_rgb * sv, union * sv), 1), size))
        lr = self.lr_encoder(F.interpolate(lr_image * lv, size=size, mode="bilinear", align_corners=False))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, size[0], device=hr.device, dtype=hr.dtype),
                                torch.linspace(-1, 1, size[1], device=hr.device, dtype=hr.dtype), indexing="ij")
        xy = torch.stack((xx, yy)).unsqueeze(0).expand(b, -1, -1, -1)
        features = self.fusion(torch.cat((hr, lr, hr * lr, (hr - lr).abs(), xy), 1))
        scene = features.mean((-2, -1))
        source_unit = torch.tanh(self.source_head(F.adaptive_avg_pool2d(features, (4, 4)).flatten(1)))
        source_shift = source_unit * self.config.max_source_shift_px
        area = instance_masks.sum((-2, -1))
        valid = area > 0
        if n:
            weights = F.adaptive_avg_pool2d(instance_masks * sv, size)
            roi = torch.einsum("bnhw,bchw->bnc", weights, features) / weights.sum((-2, -1)).clamp_min(1e-8)[..., None]
            xs = torch.linspace(-1, 1, w, device=hr.device, dtype=hr.dtype)
            ys = torch.linspace(-1, 1, h, device=hr.device, dtype=hr.dtype)
            cx = (instance_masks.sum(-2) * xs).sum(-1) / area.clamp_min(1e-8)
            cy = (instance_masks.sum(-1) * ys).sum(-1) / area.clamp_min(1e-8)
            shape = torch.stack((cx, cy, area / (h * w)), -1)
            raw = self.instance_head(torch.cat((roi, scene[:, None].expand(-1, n, -1), shape), -1))
        else:
            raw = scene.sum(-1)[:, None, None].expand(-1, 0, 3)
        residual_unit = torch.tanh(raw[..., :2])
        residual = residual_unit * self.config.max_residual_shift_px
        total = source_shift[:, None] + residual
        source_incomplete = ((instance_masks * sv).sum((-2, -1)) < area - 1e-5) | ~valid
        lvc = F.interpolate(lv, size=(h, w), mode="nearest")
        reference_missing = ((instance_masks * lvc).sum((-2, -1)) < area - 1e-5) | (lv.sum((-2, -1)) == 0)
        overflow = _canvas_overflow(instance_masks, total)
        near_bound = (source_unit.abs().amax(-1)[:, None] >= .95) | (residual_unit.abs().amax(-1) >= .95)
        return dict(source_shift_px=source_shift, instance_residual_px=residual,
                    total_shift_px=total, unmatched_logits=raw[..., 2],
                    unmatched_probability=torch.sigmoid(raw[..., 2]), instance_valid=valid,
                    source_support_incomplete=source_incomplete, reference_support_missing=reference_missing,
                    support_overflow=overflow, near_training_bound=near_bound,
                    ood_risk=near_bound | overflow | source_incomplete | reference_missing,
                    target_semantics=TARGET_SEMANTICS, coordinate_convention=COORDINATES,
                    component_interpretation="MODEL_INTERNAL_NOT_IDENTIFIED_PHYSICAL_PARALLAX",
                    uncertainty_status="UNCALIBRATED", domain_ood_status="NOT_ASSESSED",
                    application_status="CANDIDATE_NOT_ADMITTED")


def translate_vertices(vertices: Tensor, total_shift_px: Tensor) -> Tensor:
    """Translate every vertex of [B,N,K,2] once; no clipping/scale/rotation.

    Rings and multipart topology are owned by the caller; each point receives
    the same vector. This exact vector route preserves geometry and its area.
    """
    _float_tensor(vertices, "vertices", 4)
    _float_tensor(total_shift_px, "total_shift_px", 3)
    if vertices.shape[-1] != 2 or total_shift_px.shape != vertices.shape[:2] + (2,):
        raise ValueError("Vertex/translation shape mismatch")
    if total_shift_px.device != vertices.device or total_shift_px.dtype != vertices.dtype:
        raise ValueError("Vertex/translation dtype/device mismatch")
    return vertices + total_shift_px.unsqueeze(-2)


def pixel_shift_to_map(total_shift_px: Tensor, affine_linear: Tensor, *, map_unit: str) -> dict:
    """Apply the explicit 2x2 affine linear part; never infer GSD or map units.

    For north-up affine [[gsd,0],[0,-gsd]], down pixels become negative map y.
    Unknown pixel scale stays pixel-only by not calling this function.
    """
    _float_tensor(total_shift_px, "total_shift_px")
    _float_tensor(affine_linear, "affine_linear", 2)
    if total_shift_px.shape[-1] != 2 or affine_linear.shape != (2, 2):
        raise ValueError("Explicit 2x2 affine and (...,2) vector required")
    if map_unit not in {"metre", "degree", "unknown_map_unit"}:
        raise ValueError("Declare the actual CRS unit; no implicit metres")
    if affine_linear.device != total_shift_px.device or affine_linear.dtype != total_shift_px.dtype:
        raise ValueError("Affine/vector dtype/device mismatch")
    if bool(torch.linalg.det(affine_linear).abs() < 1e-12):
        raise ValueError("Singular affine")
    return dict(vector=total_shift_px @ affine_linear.T, unit=map_unit,
                accuracy_status="NOT_CERTIFIED")


def translate_and_aggregate(instance_masks, total_shift_px, *, factor=1, target_valid=None):
    """Differentiable rigid whole-instance raster transport and union surrogate.

    Backward sampling uses (output - forward_shift), align_corners=False.
    No unmatched weights: rejected predictions cannot erase hard objects.
    max union is exact for binary aligned masks; fractional bilinear overlap
    is a surrogate, NOT label_transfer's exact polygon/cell integration.
    factor is an integer pixel nesting ratio, not implicitly ten metres.
    Unavailable target area remains explicit; denominator is full cell area.
    Raster clipping/invalid support is measured, never called preservation.
    Memory is O(B*N*H*W); caller must supply complete instances with halos.
    """
    _geometry_inputs(instance_masks, total_shift_px)
    b, n, h, w = instance_masks.shape
    if type(factor) is not int or factor < 1 or h % factor or w % factor:
        raise ValueError("factor must divide both canvas dimensions exactly")
    tv = _validity(target_valid, (b, 1, h, w), instance_masks, "target_valid")
    if n:
        yy, xx = torch.meshgrid(torch.arange(h, device=instance_masks.device, dtype=instance_masks.dtype),
                                torch.arange(w, device=instance_masks.device, dtype=instance_masks.dtype), indexing="ij")
        base = torch.stack((2 * (xx + .5) / w - 1, 2 * (yy + .5) / h - 1), -1)
        scale = total_shift_px.new_tensor([2 / w, 2 / h])
        grid = base[None, None] - (total_shift_px * scale)[..., None, None, :]
        moved = F.grid_sample(instance_masks.reshape(b * n, 1, h, w), grid.reshape(b * n, h, w, 2),
                              mode="bilinear", padding_mode="zeros", align_corners=False).reshape(b, n, h, w)
        union = moved.amax(1, keepdim=True)
    else:
        moved = instance_masks + total_shift_px.sum() * 0
        union = instance_masks.sum(1, keepdim=True)
    source_area = instance_masks.sum((-2, -1))
    raster_area = moved.sum((-2, -1))
    supported_area = (moved * tv).sum((-2, -1))
    return dict(moved_masks=moved, union=union, fractions=F.avg_pool2d(union * tv, factor),
                valid_fraction=F.avg_pool2d(tv, factor), unknown_fraction=1 - F.avg_pool2d(tv, factor),
                input_area_px2=source_area, raster_area_px2=raster_area,
                supported_area_px2=supported_area,
                lost_area_px2=(source_area - supported_area).clamp_min(0),
                support_overflow=_canvas_overflow(instance_masks, total_shift_px) |
                                 (supported_area < source_area - 1e-4),
                collision_mass_px2=moved.sum((-3, -2, -1)) - union.sum((-3, -2, -1)),
                fraction_semantics="DIFFERENTIABLE_RASTER_SURROGATE_FULL_CELL_DENOMINATOR")


def known_transform_loss(prediction, target_shift_px, matched, *, supervision_valid=None):
    """Known synthetic labels ONLY; predicted unmatched never gates regression.

    matched=False is an observed synthetic disappearance, not inferred change.
    Unsupported labels are excluded with an explicit mask. Empty denominators
    raise; an all-unmatched labelled batch has BCE and a disabled regression
    (None), not a fabricated zero displacement result.
    """
    total, logits = prediction["total_shift_px"], prediction["unmatched_logits"]
    _float_tensor(total, "predicted shift", 3)
    _float_tensor(target_shift_px, "target_shift_px", 3)
    _float_tensor(logits, "unmatched_logits", 2)
    if target_shift_px.shape != total.shape or logits.shape != total.shape[:2]:
        raise ValueError("Synthetic target/prediction shape mismatch")
    if target_shift_px.device != total.device or target_shift_px.dtype != total.dtype:
        raise ValueError("Synthetic target dtype/device mismatch")
    if not isinstance(matched, Tensor) or matched.shape != logits.shape or matched.dtype != torch.bool or matched.device != total.device:
        raise ValueError("matched must be an explicit bool tensor")
    valid = prediction["instance_valid"].bool()
    if supervision_valid is not None:
        valid = valid & _validity(supervision_valid, logits.shape, total, "supervision_valid").bool()
    if not bool(valid.any()):
        raise ValueError("NO_VALID_SYNTHETIC_SUPERVISION")
    positives = valid & matched
    bce = F.binary_cross_entropy_with_logits(logits[valid], (~matched[valid]).to(logits.dtype))
    regression = F.smooth_l1_loss(total[positives], target_shift_px[positives]) if bool(positives.any()) else None
    loss = bce if regression is None else bce + regression
    return dict(loss=loss, known_transform=regression, unmatched_bce=bce,
                supervised_count=valid.sum(), matched_count=positives.sum(),
                predicted_unmatched_rate=(logits[valid] > 0).to(total.dtype).mean())


def reference_map_loss(raster, target_roof, valid):
    """Fixed target GT support, ungated union L1+Dice and nested fraction L1.

    This minimal scene supervision is not a latent identity matcher. valid is
    declared label coverage (including background ONLY where confirmed), never
    a predicted mask or unmatched decision. For positive-only GT, valid must
    exclude unknown exterior pixels. GT never enters predictor/transport.
    """
    union = raster["union"]
    _probability(target_roof, "target_roof")
    if target_roof.shape != union.shape or target_roof.device != union.device or target_roof.dtype != union.dtype:
        raise ValueError("Reference map shape/device/dtype mismatch")
    if valid is None:
        raise ValueError("Explicit GT coverage required")
    mask = _validity(valid, union.shape, union, "valid")
    count = mask.sum()
    if not bool(count > 0):
        raise ValueError("NO_VALID_REFERENCE_SUPERVISION")
    l1 = ((union - target_roof).abs() * mask).sum() / count
    intersection = (union * target_roof * mask).sum()
    # Squared denominator also gives zero for identical soft occupancy maps.
    dice = 1 - (2 * intersection + 1e-6) / (((union.square() + target_roof.square()) * mask).sum() + 1e-6)
    fh, fw = raster["fractions"].shape[-2:]
    h, w = union.shape[-2:]
    factor = h // fh
    if h % fh or w % fw or w // fw != factor:
        raise ValueError("Nested fraction grid required")
    block_valid = F.avg_pool2d(mask, factor)
    fraction_l1 = (F.avg_pool2d((union - target_roof) * mask, factor).abs().sum() /
                   block_valid.sum())
    return dict(loss=l1 + dice + fraction_l1, map_l1=l1, map_dice=dice,
                fraction_l1=fraction_l1, valid_pixel_count=count,
                reference_positive_pixel_mass=(target_roof * mask).sum())


def _stable_track_loss(stable_edges, total, original):
    """Only evidence-supported edges in one explicit target pixel frame."""
    zero_count = torch.zeros((), device=total.device, dtype=torch.long)
    if stable_edges is None:
        return None, zero_count, zero_count
    keys = {"indices", "source_delta_px", "confirmed_stable", "same_target_frame", "weights"}
    if not isinstance(stable_edges, Mapping) or set(stable_edges) != keys:
        raise ValueError("Exact stable_edges evidence schema required")
    indices = stable_edges["indices"]
    if (not isinstance(indices, Tensor) or indices.dtype != torch.long or indices.ndim != 2 or
            indices.shape[1] != 4 or indices.device != total.device):
        raise ValueError("stable_edges.indices must be int64 [E,4] on prediction device")
    e = indices.shape[0]
    delta, weights = stable_edges["source_delta_px"], stable_edges["weights"]
    for value, name, shape in ((delta, "source_delta_px", (e, 2)), (weights, "weights", (e,))):
        _float_tensor(value, "stable_edges." + name)
        if value.shape != shape or value.device != total.device or value.dtype != total.dtype:
            raise ValueError("Stable edge tensor shape/device/dtype mismatch: " + name)
        if value.requires_grad:
            raise ValueError("Stable edge evidence must be fixed, not model predictions")
    if bool(((weights < 0) | (weights > 1)).any()):
        raise ValueError("Stable edge weights must lie in [0,1]")
    flags = []
    for name in ("confirmed_stable", "same_target_frame"):
        value = stable_edges[name]
        if not isinstance(value, Tensor) or value.dtype != torch.bool or value.shape != (e,) or value.device != total.device:
            raise ValueError("Explicit bool [E] stable edge evidence required: " + name)
        flags.append(value)
    provided = zero_count + e
    if not e:
        return None, provided, zero_count
    bi, ni, bj, nj = indices.unbind(1)
    b, n = original.shape
    if bool(((indices < 0).any(1) | (bi >= b) | (bj >= b) | (ni >= n) | (nj >= n)).any()):
        raise ValueError("Stable edge index outside original instance table")
    if bool(((bi == bj) & (ni == nj)).any()):
        raise ValueError("Self edges are not independent stable track evidence")
    eligible = flags[0] & flags[1] & original[bi, ni] & original[bj, nj] & (weights > 0)
    if not bool(eligible.any()):
        return None, provided, zero_count
    # Compare corrected positions, NOT equal shifts: native centroids may differ.
    error = total[bi[eligible], ni[eligible]] - total[bj[eligible], nj[eligible]] + delta[eligible]
    per_edge = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none").mean(-1)
    chosen_weights = weights[eligible]
    return (per_edge * chosen_weights).sum() / chosen_weights.sum(), provided, eligible.sum()


def real_weak_alignment_loss(prediction, raster, target_roof, valid, stable_edges=None):
    """Bounded scene-level weak supervision; not a latent correspondence solver.

    GT is read ONLY by reference_map_loss inside explicit fixed label coverage
    valid [B,1,H,W]. For positive-only roof support set exterior valid=False;
    never fill unknown exterior as negative/background. Geometric penalties
    below are label-independent priors over original instances, not extra GT.

    Fixed initial weights: map=1, canvas overflow=1, collision=.1,
    total_shift_l2=.001, unmatched coverage=.1, supported stable track=.1.
    total_shift_l2 is mean squared per-axis displacement in INPUT pixel units
    (featurepx for the feature256 runner), never metres. Caller configures the
    predictor's 12/4 featurepx bounds; this loss does not alter architecture.

    Original instance count/mass are detached from unmatched/acceptance. Map
    loss is NEVER multiplied by predicted match scores. A per-source 10% soft
    matched-mass floor is an explicitly weak anti-collapse prior, NOT a measured
    match rate, per-instance positive label or production coverage threshold.
    The log-domain hinge stays differentiable for large unmatched logits.

    Canvas lost mass = input_area - raster_area; GT label coverage is NOT a
    geometric support polygon. In particular raster.lost_area/support_overflow
    (which may include target_valid) are not used to push roofs into positive-
    only GT. Nonrectangular physical support requires a separate outer check.

    Optional stable_edges mapping (all on prediction device): indices int64
    [E,4]=(batch_i,instance_i,batch_j,instance_j), source_delta_px float [E,2]
    =native_centroid_i-native_centroid_j in the SAME target pixel frame,
    confirmed_stable/same_target_frame bool [E], weights float [E] in [0,1].
    Evidence/deltas must be fixed and independently supported, not GT-derived
    cached inference vectors. Missing/unsupported edges return None and
    NOT_AVAILABLE, never a fabricated zero. Different dates without stability
    evidence stay weak scene supervision; they are not displacement truth.

    Returns differentiable loss/components, detached counters, explicit status
    and fixed weights. No file access, data loading, train loop or GT forward.
    """
    total, logits = prediction["total_shift_px"], prediction["unmatched_logits"]
    _float_tensor(total, "predicted shift", 3)
    _float_tensor(logits, "unmatched_logits", 2)
    _probability(raster["moved_masks"], "moved_masks")
    _probability(raster["union"], "union")
    b, n = total.shape[:2]
    if total.shape[-1] != 2 or logits.shape != (b, n):
        raise ValueError("Weak prediction shape mismatch")
    area, raster_area = raster["input_area_px2"], raster["raster_area_px2"]
    for value, name, shape in ((logits, "logits", (b, n)), (area, "input_area_px2", (b, n)),
                               (raster_area, "raster_area_px2", (b, n))):
        _float_tensor(value, name)
        if value.shape != shape or value.device != total.device or value.dtype != total.dtype:
            raise ValueError("Weak raster/prediction shape/device/dtype mismatch: " + name)
    if bool(((area < 0) | (raster_area < 0)).any()):
        raise ValueError("Nonnegative instance masses required")
    moved, union = raster["moved_masks"], raster["union"]
    if (moved.shape[:2] != (b, n) or union.shape != (b, 1, *moved.shape[-2:]) or
            moved.device != total.device or moved.dtype != total.dtype or
            union.device != total.device or union.dtype != total.dtype):
        raise ValueError("Weak moved masks must use the original instance table/canvas")
    original_area = area.detach()
    original = original_area > 0
    instance_valid = prediction["instance_valid"]
    if (not isinstance(instance_valid, Tensor) or instance_valid.dtype != torch.bool or
            instance_valid.shape != original.shape or instance_valid.device != total.device or
            not torch.equal(instance_valid, original)):
        raise ValueError("Original instance denominator cannot be changed by acceptance/unmatched")
    if valid is None or (isinstance(valid, Tensor) and valid.requires_grad):
        raise ValueError("Independent fixed label coverage required")
    coverage = _validity(valid, union.shape, union, "valid").detach()
    ref = reference_map_loss(raster, target_roof, coverage)
    count_by_source = original.sum(-1)
    count, original_mass = original.sum(), original_area.sum()
    canvas_lost = (original_area - raster_area).clamp_min(0)
    overflow = collision = displacement = collapse = None
    matched_mass = torch.sigmoid(-logits).masked_fill(~original, 0).sum()
    if bool(count > 0):
        overflow = canvas_lost.sum() / original_mass.clamp_min(1e-8)
        collision = ((moved.sum(1, keepdim=True) - union).clamp_min(0).sum() /
                     original_mass.clamp_min(1e-8))
        displacement = total[original].square().sum() / (2 * count)
        active = count_by_source > 0
        log_match = F.logsigmoid(-logits[active]).masked_fill(~original[active], -float("inf"))
        log_rate = torch.logsumexp(log_match, -1) - count_by_source[active].to(total.dtype).log()
        collapse = (F.relu(math.log(.1) - log_rate) * count_by_source[active]).sum() / count
    track, edges_provided, edges_used = _stable_track_loss(stable_edges, total, original)
    components = dict(reference_map=ref["loss"], map_l1=ref["map_l1"], map_dice=ref["map_dice"],
                      fraction_l1=ref["fraction_l1"], support_overflow=overflow,
                      collision=collision, total_shift_l2=displacement,
                      unmatched_coverage=collapse, stable_track=track)
    weights = dict(reference_map=1., support_overflow=1., collision=.1,
                   total_shift_l2=.001, unmatched_coverage=.1, stable_track=.1)
    loss = sum(components[name] * weight for name, weight in weights.items() if components[name] is not None)
    counters = dict(original_instance_count=count, original_instance_count_by_source=count_by_source,
                    original_instance_mass_px2=original_mass, valid_pixel_count=ref["valid_pixel_count"],
                    unknown_label_pixel_count=(coverage == 0).sum(),
                    confirmed_background_pixel_count=((coverage > 0) & (target_roof == 0)).sum(),
                    reference_positive_pixel_mass=ref["reference_positive_pixel_mass"],
                    expected_matched_instance_mass=matched_mass,
                    predicted_unmatched_count=((logits > 0) & original).sum(),
                    canvas_lost_mass_px2=canvas_lost.sum(),
                    canvas_overflow_instance_count=((canvas_lost > 1e-4) & original).sum(),
                    stable_edges_provided=edges_provided, stable_edges_used=edges_used)
    return dict(loss=loss, components=components, weights=weights,
                counters={key: value.detach() for key, value in counters.items()},
                stable_track_status="AVAILABLE_SUPPORTED_EDGES" if track is not None else "NOT_AVAILABLE",
                unmatched_constraint_status="WEAK_PRIOR_NOT_MATCH_TRUTH" if collapse is not None else "NOT_AVAILABLE",
                supervision_status="SCENE_WEAK_SUPERVISION_NOT_INSTANCE_DISPLACEMENT_TRUTH",
                support_penalty_scope="CANVAS_ONLY_NOT_GT_LABEL_COVERAGE",
                minimum_matched_fraction_prior=.1)


def alignment_checkpoint(model: AlignmentPredictor) -> dict:
    """In-memory tensor-only payload; runner owns atomic storage/resume/provenance."""
    return dict(format=CHECKPOINT_FORMAT, config=asdict(model.config),
                target_semantics=TARGET_SEMANTICS, coordinates=COORDINATES,
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})


def load_alignment_checkpoint(payload: Mapping, *, expected_config=None) -> AlignmentPredictor:
    """Strict config/semantics/key/shape/dtype/finite load, no path/network access.

    Caller deserializes a trusted tensor payload (weights_only=True where
    supported). Loading proves architecture compatibility, not training or
    quality. Returned CPU model is eval; runner explicitly .to(...)/.train().
    """
    keys = {"format", "config", "target_semantics", "coordinates", "state_dict"}
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise ValueError("Exact alignment checkpoint schema required")
    if (payload["format"] != CHECKPOINT_FORMAT or payload["target_semantics"] != TARGET_SEMANTICS or
            payload["coordinates"] != COORDINATES):
        raise ValueError("Checkpoint format/semantics/coordinates mismatch")
    if not isinstance(payload["config"], dict) or set(payload["config"]) != set(asdict(AlignmentConfig())):
        raise ValueError("Exact config schema required")
    config = AlignmentConfig(**payload["config"])
    if expected_config is not None and config != expected_config:
        raise ValueError("Checkpoint config differs from expected training contract")
    # Construction must not perturb a resumed runner's CPU RNG sequence.
    with torch.random.fork_rng(devices=[]):
        model = AlignmentPredictor(config)
    state = payload["state_dict"]
    expected = model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("Strict state_dict keys required")
    for key, value in state.items():
        _float_tensor(value, "state_dict." + key)
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError("State shape/dtype mismatch: " + key)
    model.load_state_dict(state, strict=True)
    return model.eval()
