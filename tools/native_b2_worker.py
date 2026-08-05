"""Guarded native Open-CD worker for the B2 boundary auxiliary ablation.

The worker is intentionally self-contained because the RSBuilding environment
owns Open-CD/MMEngine.  It registers a small auxiliary head at runtime and
loads the protected iter2000 state dict explicitly.  No Test dataloader is
constructed: both the train and validation lists are checked before Runner
construction and are the only lists passed to the runner.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def has_test(value: object) -> bool:
    lowered = str(value).lower().replace("/", "\\")
    return "\\test\\" in lowered or lowered.endswith("\\test.txt") or lowered.endswith("\\test_samples.txt")


def _boundary_target_torch(torch, F, data_samples, width: int):
    rows = []
    for sample in data_samples:
        raw = sample.gt_sem_seg_from.data
        if raw.ndim == 2:
            raw = raw.unsqueeze(0)
        rows.append(raw[:1])
    target = torch.stack(rows, dim=0).float()
    valid = target != 255
    building = (target > 0) & valid
    k = int(width) * 2 + 1
    pad = int(width)
    eroded = -F.max_pool2d(-building.float(), kernel_size=k, stride=1, padding=pad) > 0.5
    dilated = F.max_pool2d(building.float(), kernel_size=k, stride=1, padding=pad) > 0.5
    boundary = (dilated & ~eroded & valid).float()
    return boundary, valid.float()


def register_boundary_decoder(torch, nn, F, MODELS, Foundation_Decoder):
    """Register a native decoder subclass without editing the external repo."""
    if "B2BoundaryDecoder" in MODELS.module_dict:
        return MODELS.module_dict["B2BoundaryDecoder"]

    @MODELS.register_module()
    class B2BoundaryDecoder(Foundation_Decoder):
        def __init__(self, *args, boundary_width=2, boundary_weight=0.1, boundary_channels=64, **kwargs):
            super().__init__(*args, **kwargs)
            self.boundary_width = int(boundary_width)
            self.boundary_weight = float(boundary_weight)
            in_channels = int(getattr(self, "channels", 256))
            hidden = int(boundary_channels)
            self.boundary_head = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=max(1, min(8, hidden)), num_channels=hidden),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=1),
            )
            self._boundary_feature = None
            self.last_boundary_logits = None
            self.boundary_gradient_seen = False
            for parameter in self.boundary_head.parameters():
                parameter.register_hook(self._record_boundary_gradient)
            self.fusion_ab.register_forward_hook(self._capture_boundary_feature)

        def _record_boundary_gradient(self, gradient):
            self.boundary_gradient_seen = bool(torch.isfinite(gradient).all() and gradient.detach().abs().max() > 0)
            return gradient

        def _capture_boundary_feature(self, module, inputs, output):
            # fusion_ab is called for A and B.  Both observations are copies in
            # the WHU native contract; keeping the latest B feature is stable.
            self._boundary_feature = output

        def forward(self, x, batch_data_samples):
            self._boundary_feature = None
            native_logits = super().forward(x, batch_data_samples)
            if self._boundary_feature is None:
                raise RuntimeError("B2 boundary feature hook did not receive fusion_ab output")
            feature = self._boundary_feature
            boundary_logits = self.boundary_head(feature)
            shape = batch_data_samples[0].metainfo["img_shape"]
            boundary_logits = F.interpolate(boundary_logits, size=shape, mode="bilinear", align_corners=False)
            self.last_boundary_logits = boundary_logits
            return native_logits

        def loss(self, inputs, batch_data_samples, train_cfg, finetune_cfg):
            native_losses = super().loss(inputs, batch_data_samples, train_cfg, finetune_cfg)
            boundary, valid = _boundary_target_torch(torch, F, batch_data_samples, self.boundary_width)
            logits = self.last_boundary_logits
            if logits is None:
                raise RuntimeError("B2 boundary logits missing after native forward")
            boundary = boundary.to(device=logits.device)
            valid = valid.to(device=logits.device)
            masked_logits = logits[valid.bool()]
            masked_target = boundary[valid.bool()]
            if masked_logits.numel() == 0:
                loss = logits.sum() * 0.0
            else:
                bce = F.binary_cross_entropy_with_logits(masked_logits, masked_target, reduction="mean")
                probability = torch.sigmoid(masked_logits)
                dice = 1.0 - (2.0 * (probability * masked_target).sum() + 1.0) / (probability.sum() + masked_target.sum() + 1.0)
                loss = bce + dice
            if not torch.isfinite(loss):
                raise FloatingPointError("B2 boundary loss is NaN/Inf")
            native_losses["loss_boundary"] = loss * self.boundary_weight
            native_losses["boundary_loss_unweighted"] = loss.detach()
            native_losses["boundary_positive_fraction"] = masked_target.detach().mean() if masked_target.numel() else logits.sum() * 0.0
            return native_losses

    return B2BoundaryDecoder


def load_b0_state(model, checkpoint: Path, allowed_missing_prefix="decode_head.boundary_head."):
    import torch

    payload = torch.load(str(checkpoint), map_location="cpu")
    state = payload.get("state_dict", payload)
    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    invalid_missing = [key for key in missing if not key.startswith(allowed_missing_prefix)]
    if unexpected or invalid_missing:
        raise RuntimeError(f"B2 checkpoint inheritance failed: missing={invalid_missing}, unexpected={unexpected}")
    return {"source_state_keys": len(state), "target_state_keys": len(model.state_dict()), "missing_boundary_keys": missing, "unexpected_keys": unexpected, "non_boundary_missing": invalid_missing}


def make_lists(path: Path) -> None:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if has_test(line):
            raise RuntimeError(f"locked Test path in explicit list: {line}")


def main() -> int:
    if len(sys.argv) != 10:
        raise SystemExit("usage: native_b2_worker.py repo config checkpoint work_dir train_list val_list max_iters mode boundary_weight")
    repo, config_path, checkpoint, work_dir, train_list, val_list = map(Path, sys.argv[1:7])
    max_iters, mode, boundary_weight = int(sys.argv[7]), sys.argv[8], float(sys.argv[9])
    result = {"status": "failed", "mode": mode, "boundary_weight": boundary_weight, "test_used": False, "test_pixel_reads": 0}
    try:
        for path in (repo, config_path, checkpoint, work_dir, train_list, val_list):
            if has_test(path):
                raise RuntimeError(f"locked Test path rejected before Runner construction: {path}")
        make_lists(train_list); make_lists(val_list)
        sys.path.insert(0, str(repo))
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from mmengine.config import Config
        from mmengine.registry import DefaultScope
        import opencd, opencd.datasets, opencd.evaluation, opencd.models, opencd.visualization  # noqa: F401
        from opencd.registry import MODELS
        from opencd.models.decode_heads.foundation_decoder import Foundation_Decoder_swin_v1
        DefaultScope.get_instance("opencd", scope_name="opencd")
        B2BoundaryDecoder = register_boundary_decoder(torch, nn, F, MODELS, Foundation_Decoder_swin_v1)
        cfg = Config.fromfile(str(config_path))
        cfg.model.decode_head.type = "B2BoundaryDecoder"
        cfg.model.decode_head.boundary_width = 2
        cfg.model.decode_head.boundary_weight = boundary_weight
        cfg.model.decode_head.boundary_channels = 64
        cfg.work_dir = str(work_dir)
        cfg.train_cfg.max_iters = int(max_iters)
        cfg.train_cfg.val_interval = int(max_iters) if mode != "b2" else min(200, int(max_iters))
        cfg.val_interval = cfg.train_cfg.val_interval
        cfg.default_hooks.checkpoint.interval = cfg.train_cfg.val_interval
        cfg.default_hooks.logger.interval = max(1, min(20, int(max_iters)))
        cfg.train_data_list = str(train_list)
        cfg.test_data_list = str(val_list)
        for loader in (cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
            loader.num_workers = 0; loader.persistent_workers = False; loader.batch_size = 1
        if hasattr(cfg.train_dataloader.dataset, "dataset"):
            cfg.train_dataloader.dataset.dataset.data_list = str(train_list)
            if mode in {"one_sample", "smoke"} or mode.startswith("one_sample"):
                cfg.train_dataloader.dataset.skip_type_keys = ["RandomMosaic_Modified"]
        else:
            cfg.train_dataloader.dataset.data_list = str(train_list)
        cfg.val_dataloader.dataset.data_list = str(val_list)
        cfg.test_dataloader.dataset.data_list = str(val_list)
        cfg.model.finetune_cfg = ["neck", "decoder", "ab mask head", "cd mask head", "ab query head", "cd query head"]
        cfg.model.decode_head.finetune_cfg = cfg.model.finetune_cfg
        cfg.optim_wrapper.optimizer.lr = min(float(cfg.optim_wrapper.optimizer.lr), 1e-5)
        # The one-sample gate is an architecture/gradient diagnostic, not a
        # candidate training schedule.  The native config warms up for 1000
        # iterations, so a 20--500 step overfit would otherwise run at a tiny
        # fraction of the declared LR and produce a false negative.  Only the
        # explicitly named constant-LR diagnostic mode bypasses that warmup;
        # smoke and pilot retain the source scheduler unchanged.
        if mode.startswith("one_sample") and "constantlr" in mode:
            cfg.param_scheduler = []
        cfg.optim_wrapper.paramwise_cfg = dict(custom_keys={
            "backbone": dict(lr_mult=0.0, decay_mult=1.0),
            "decode_head.boundary_head": dict(lr_mult=1.0, decay_mult=1.0),
            "decode_head": dict(lr_mult=0.25, decay_mult=1.0),
        }, norm_decay_mult=0.0)
        cfg.resume = False
        cfg.load_from = None
        from mmengine.runner import Runner
        runner = Runner.from_cfg(cfg)
        runner.model.init_weights = lambda: None
        inheritance = load_b0_state(runner.model, checkpoint)
        # B2 trains the native decoder/building heads plus the boundary head;
        # the backbone stays frozen to isolate the auxiliary ablation.
        for name, param in runner.model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
            elif name.startswith("decode_head.boundary_head."):
                param.requires_grad = True
        registry = []
        for name, param in runner.model.named_parameters():
            if name.startswith("backbone."):
                module, group, lr = "encoder", "frozen", 0.0
            elif name.startswith("decode_head.boundary_head."):
                module, group, lr = "boundary_head", "boundary", float(cfg.optim_wrapper.optimizer.lr)
            else:
                module, group, lr = "native_decoder_building_head", "native_decoder_building", float(cfg.optim_wrapper.optimizer.lr) * 0.25
            registry.append({"parameter_name": name, "module": module, "shape": list(param.shape), "requires_grad": bool(param.requires_grad), "parameter_count": int(param.numel()), "lr_group": group, "learning_rate": lr, "weight_decay": float(cfg.optim_wrapper.optimizer.weight_decay)})
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "B2_PARAMETER_REGISTRY.csv").write_text("parameter_name,module,shape,requires_grad,parameter_count,lr_group,learning_rate,weight_decay\n" + "\n".join(",".join([str(item[k]).replace(",", ";") for k in ("parameter_name", "module", "shape", "requires_grad", "parameter_count", "lr_group", "learning_rate", "weight_decay")]) for item in registry) + "\n", encoding="utf-8")
        (work_dir / "B2_CHECKPOINT_INHERITANCE.json").write_text(json.dumps(inheritance, indent=2) + "\n", encoding="utf-8")
        runner.train()
        decoder = next((module for module in runner.model.modules() if hasattr(module, "boundary_head")), None)
        boundary_logits = getattr(decoder, "last_boundary_logits", None) if decoder is not None else None
        boundary_std = float(boundary_logits.detach().float().std().cpu()) if boundary_logits is not None else None
        result.update({"status": "pass", "max_iters": max_iters, "trainable_parameters": sum(item["parameter_count"] for item in registry if item["requires_grad"]), "total_parameters": sum(item["parameter_count"] for item in registry), "frozen_parameters": sum(item["parameter_count"] for item in registry if not item["requires_grad"]), "parameter_registry": str((work_dir / "B2_PARAMETER_REGISTRY.csv").resolve()), "checkpoint_inheritance": inheritance, "work_dir": str(work_dir), "boundary_gradient_nonzero": bool(getattr(decoder, "boundary_gradient_seen", False)), "boundary_output_std": boundary_std, "boundary_output_nonconstant": bool(boundary_std is not None and boundary_std > 1e-6), "nonfinite": False})
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
