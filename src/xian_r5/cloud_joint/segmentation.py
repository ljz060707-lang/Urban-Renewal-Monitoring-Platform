"""Native B0 roof + independent facade auxiliary adapter (torch only).

No data, trainer, geospatial imports, downloads, or checkpoint writes on import.
Production initialization: ``RoofFacadeSegmenter.from_b0(factory, b0_path)``.
The factory builds the *original* FoundationEncoderDecoder configuration only.
Forward accepts ImageNet-normalized RGB NCHW, square side divisible by 32.
It returns two independent N1HW logits; apply sigmoid separately, never softmax.
Only source-frame native labels belong in ``native_segmentation_loss``.

All B0 modules stay in eval mode, including when trainable: this fixes running
buffers and disables stochastic depth for small-batch fine tuning. Gradients
remain enabled. Training stages are heads -> last -> all, controlled externally
after coverage epochs, with unrelated B/CD-only parameters always frozen.
The temporary feature hook is local to one forward and removed in finally.
Concurrent calls on the same model instance are unsupported (one model/worker).
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

B0_SHA256 = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
ROOF_SEMANTICS = "VISIBLE_ROOF_IN_SOURCE_IMAGE_FRAME"
_ROOF_MODULES = (
    "simple_fpn", "transformer_decoder", "building_a_embed", "fusion_ab",
    "pred_head_ab", "output_upscaling_ab",
)


def normalize_rgb(rgb: Tensor) -> Tensor:
    """Match original B0 RGB->BGR->preprocessor RGB normalization directly."""
    if rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("expected uint8 RGB N3HW")
    mean = rgb.new_tensor([123.675, 116.28, 103.53], dtype=torch.float32)
    std = rgb.new_tensor([58.395, 57.12, 57.375], dtype=torch.float32)
    return (rgb.float() - mean[None, :, None, None]) / std[None, :, None, None]


@dataclass
class _ImageSample:
    # Original decoder forward reads metainfo only, never supervision.
    metainfo: dict


class RoofFacadeSegmenter(nn.Module):
    """Own an original B0 and a 64->1 facade head; no replacement roof head.

    The constructor accepts an already loaded dependency for testing/restore.
    Use from_b0 for verified initialization; no S2/S44 state is accepted there.
    State dict contains full B0, facade parameters, stage and initialization
    provenance. Strict restore needs an identical architecture, not old paths.
    """

    def __init__(self, b0: nn.Module):
        super().__init__()
        backbone, decoder = b0.backbone, b0.decode_head
        if len(backbone.stages) != 4 or not isinstance(backbone.norm3, nn.Module):
            raise ValueError("expected verified four-stage Swin with norm3")
        if decoder.channels != 256:
            raise ValueError("expected original 256-channel B0 decoder")
        for name in _ROOF_MODULES:
            if not isinstance(getattr(decoder, name, None), nn.Module):
                raise ValueError("missing native roof module: " + name)
        up = decoder.output_upscaling_ab
        if not isinstance(up, nn.Sequential) or len(up) != 5:
            raise ValueError("unexpected native A/B upscaling structure")
        if not isinstance(up[3], nn.ConvTranspose2d) or up[3].out_channels != 64:
            raise ValueError("expected native 64-channel A feature")
        self.b0 = b0
        self.facade_aux = nn.Conv2d(64, 1, kernel_size=1)
        # Respect a factory's chosen device without requiring CUDA.
        ref = next(b0.parameters())
        self.facade_aux.to(device=ref.device, dtype=ref.dtype)
        self.initialization_sha256 = None
        self.set_stage("heads")

    @classmethod
    def from_b0(cls, factory: Callable[[], nn.Module], checkpoint) -> "RoofFacadeSegmenter":
        """Verify original B0 bytes, then strict-load; factory must not download."""
        path = Path(checkpoint)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != B0_SHA256:
                raise ValueError("initialization is not the frozen original B0")
            # Original MMEngine B0 contains HistoryBuffer metadata and cannot
            # be read with torch2.0 weights_only. Trust ONLY this exact digest,
            # load from the same verified open handle, and discard all metadata.
            stream.seek(0)
            payload = torch.load(stream, map_location="cpu", weights_only=False)
        state = payload["state_dict"] if "state_dict" in payload else payload
        b0 = factory()
        b0.load_state_dict(state, strict=True)
        model = cls(b0)
        model.initialization_sha256 = B0_SHA256
        return model

    def set_stage(self, stage: str):
        """heads/last/all/frozen; caller must rebuild optimizer after transition.

        'last' includes stages.3 and norm3; 'all' includes the whole image
        backbone. Neither unlocks building_b_embed or CD-specific modules.
        'frozen' supports P2: no parameter or buffer updates.
        """
        if stage not in {"heads", "last", "all", "frozen"}:
            raise ValueError("unknown training stage: " + str(stage))
        for param in self.parameters():
            param.requires_grad_(False)
            param.grad = None
        if stage != "frozen":
            for name in _ROOF_MODULES:
                getattr(self.b0.decode_head, name).requires_grad_(True)
            self.facade_aux.requires_grad_(True)
        if stage == "last":
            self.b0.backbone.stages[3].requires_grad_(True)
            self.b0.backbone.norm3.requires_grad_(True)
        elif stage == "all":
            self.b0.backbone.requires_grad_(True)
        self.stage = stage
        self.train(self.training)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.b0.eval()  # no frozen buffer drift; eval does not disable gradients
        return self

    def parameter_groups(self, head_lr=3e-4, backbone_lr=1e-5):
        """Disjoint groups, to be constructed again after each stage change."""
        groups = []
        for prefix, lr in (("b0.backbone.", backbone_lr), (None, head_lr)):
            params = [p for n, p in self.named_parameters() if p.requires_grad
                      and (n.startswith("b0.backbone.") if prefix else
                           not n.startswith("b0.backbone."))]
            if params:
                groups.append({"params": params, "lr": lr})
        return groups

    def get_extra_state(self):
        return {"schema": "XIAN_CLOUD_S_V1", "stage": self.stage,
                "initialization_sha256": self.initialization_sha256,
                "roof_semantics": ROOF_SEMANTICS,
                "normalization": "RGB_IMAGENET", "frozen_buffers": True}

    def set_extra_state(self, state):
        if (not isinstance(state, dict) or state.get("schema") != "XIAN_CLOUD_S_V1"
                or state.get("roof_semantics") != ROOF_SEMANTICS
                or state.get("normalization") != "RGB_IMAGENET"
                or state.get("frozen_buffers") is not True
                or state.get("initialization_sha256") not in (None, B0_SHA256)):
            raise ValueError("incompatible S checkpoint metadata")
        self.set_stage(state["stage"])
        self.initialization_sha256 = state["initialization_sha256"]

    def forward(self, rgb_normalized: Tensor):
        x = rgb_normalized
        if (x.ndim != 4 or x.shape[0] < 1 or x.shape[1] != 3
                or not x.is_floating_point() or x.shape[-2] != x.shape[-1]
                or x.shape[-1] < 32 or x.shape[-1] % 32):
            raise ValueError("expected normalized RGB N3SS, S>=32 divisible by32")
        features = []

        def capture(_module, _inputs, output):
            features.append(output)  # A then B in verified native decoder

        hook = self.b0.decode_head.output_upscaling_ab.register_forward_hook(capture)
        try:
            samples = [_ImageSample({"img_shape": tuple(x.shape[-2:])})
                       for _ in range(x.shape[0])]
            native = self.b0._forward(x, samples)
        finally:
            hook.remove()
        if (len(features) != 2 or features[0].ndim != 4
                or features[0].shape[:2] != (x.shape[0], 64)
                or native.shape != (x.shape[0], 3, *x.shape[-2:])):
            raise RuntimeError("native B0 feature/output contract changed")
        facade = F.interpolate(self.facade_aux(features[0]), x.shape[-2:],
                               mode="bilinear", align_corners=False)
        return {"roof_logits": native[:, 0:1], "facade_logits": facade}


def bandon_targets(semantic: Tensor):
    """0 background,1 roof,2 projected facade,255 ignore -> separate binary GT.

    Projected facade remains auxiliary, never exact visible-facade truth.
    Independent target tensors also allow overlapping labels from other sources.
    """
    if not torch.all((semantic == 0) | (semantic == 1) |
                     (semantic == 2) | (semantic == 255)):
        raise ValueError("unexpected BANDON label value")
    ignored = semantic == 255
    roof = (semantic == 1).long().masked_fill(ignored, 255)
    facade = (semantic == 2).long().masked_fill(ignored, 255)
    return roof, facade


def _target(target: Tensor, logits: Tensor):
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.shape != logits.shape or target.device != logits.device:
        raise ValueError("target must match N1HW logits shape and device")
    if not torch.all((target == 0) | (target == 1) | (target == 255)):
        raise ValueError("native binary target must be 0/1/255")
    return target


def _binary_terms(logits: Tensor, target: Tensor):
    valid = target != 255
    count = int(valid.sum().item())
    if not count:
        return None, None, count
    # Select first, including BEFORE sigmoid: NaN in IGNORE cannot contaminate.
    z, y = logits[valid].float(), target[valid].float()
    if not torch.isfinite(z).all():
        raise FloatingPointError("nonfinite logit on supervised native support")
    bce = F.binary_cross_entropy_with_logits(z, y)
    p = z.sigmoid()
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce, dice, count


def _boundary_loss(logits: Tensor, target: Tensor):
    # Pairwise probability edges. Only pairs whose BOTH GT pixels are valid
    # enter sigmoid; holes/NoData and crop exterior never create fake edges.
    sums, count = [], 0
    for axis in (-2, -1):
        left, right = [slice(None)] * 4, [slice(None)] * 4
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        a, b = target[left], target[right]
        valid = (a != 255) & (b != 255)
        n = int(valid.sum().item())
        if n:
            p = (logits[left][valid].float().sigmoid() -
                 logits[right][valid].float().sigmoid()).abs()
            edge = (a[valid] != b[valid]).float()
            sums.append(F.l1_loss(p, edge, reduction="sum"))
            count += n
    return (sum(sums) / count if count else None), count


def native_segmentation_loss(outputs, roof_target: Tensor, *,
                             facade_target: Optional[Tensor] = None,
                             facade_source_weight: float = 1.0,
                             boundary_trusted: bool = False):
    """BCE + Dice + .2*valid-pair boundary + .1*source_weight*facade BCE.

    Inputs are native-frame roof GT only; map GT belongs to A/joint trainer.
    boundary_trusted must be explicitly enabled for trustworthy native labels.
    Facade source reliability is a fixed scalar [0,1], not predicted filtering.
    Missing terms are None with explicit denominators; entirely unsupported
    batch returns loss=None. Caller MUST skip backward AND optimizer/scheduler
    update on None (AdamW can change parameters even with zero gradients).
    Loss arithmetic is FP32 under AMP. Zero valid pairs disables boundary only.
    """
    weight = float(facade_source_weight)
    if not 0 <= weight <= 1:
        raise ValueError("facade_source_weight must be fixed within [0,1]")
    roof, facade = outputs["roof_logits"], outputs["facade_logits"]
    if roof.ndim != 4 or roof.shape[1] != 1 or facade.shape != roof.shape:
        raise ValueError("expected two independent N1HW logits")
    target = _target(roof_target, roof)
    rbce, rdice, nr = _binary_terms(roof, target)
    boundary, nb = _boundary_loss(roof, target) if boundary_trusted else (None, 0)
    fbce, nf = None, 0
    if facade_target is not None:
        ft = _target(facade_target, facade)
        if weight > 0:
            fbce, _unused_dice, nf = _binary_terms(facade, ft)
    terms = []
    if rbce is not None:
        terms.extend((rbce, rdice))
    if boundary is not None:
        terms.append(.2 * boundary)
    if fbce is not None:
        terms.append(.1 * weight * fbce)
    return {"loss": sum(terms) if terms else None,
            "roof_bce": rbce, "roof_dice": rdice,
            "roof_boundary": boundary, "facade_bce": fbce,
            "roof_valid_pixels": nr, "boundary_valid_pairs": nb,
            "facade_valid_pixels": nf, "skip_update": not terms}
