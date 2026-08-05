"""Native RSBuilding worker for bounded B2 boundary diagnostics.

This worker is intentionally separate from the candidate B2 worker.  It can
only run on an explicit Train/Validation list and never constructs a Test
dataset from the locked Test roots.  ``inspect`` is a single forward pass;
the probe modes are detached/head-only/shared-scope diagnostics, not a B2
pilot or candidate training run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any


def has_test(value: object) -> bool:
    lowered = str(value).lower().replace("/", "\\")
    return "\\test\\" in lowered or lowered.endswith("\\test.txt") or lowered.endswith("\\test_samples.txt")


def _target(torch, F, data_samples, width: int):
    rows = []
    for sample in data_samples:
        raw = sample.gt_sem_seg_from.data
        if raw.ndim == 2:
            raw = raw.unsqueeze(0)
        rows.append(raw[:1])
    target = torch.stack(rows, dim=0).float()
    valid = target != 255
    building = (target > 0) & valid
    kernel = int(width) * 2 + 1
    pad = int(width)
    eroded = -F.max_pool2d(-building.float(), kernel_size=kernel, stride=1, padding=pad) > 0.5
    dilated = F.max_pool2d(building.float(), kernel_size=kernel, stride=1, padding=pad) > 0.5
    return (dilated & ~eroded & valid).float(), valid.float()


def register_decoder(torch, nn, F, MODELS, FoundationDecoder):
    name = "B2DiagnosticBoundaryDecoder"
    if name in MODELS.module_dict:
        return MODELS.module_dict[name]

    @MODELS.register_module()
    class B2DiagnosticBoundaryDecoder(FoundationDecoder):
        def __init__(self, *args, boundary_width=2, boundary_channels=64,
                     diagnostic_mode="head_only", boundary_loss_variant="l0",
                     boundary_positive_weight=1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.boundary_width = int(boundary_width)
            self.diagnostic_mode = str(diagnostic_mode)
            self.boundary_loss_variant = str(boundary_loss_variant)
            self.boundary_positive_weight = float(boundary_positive_weight)
            hidden = int(boundary_channels)
            in_channels = int(getattr(self, "channels", 256))
            self.boundary_head = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
                nn.GroupNorm(max(1, min(8, hidden)), hidden),
                nn.GELU(),
                nn.Conv2d(hidden, 1, 1),
            )
            self._feature = None
            self.last_boundary_logits = None
            self.last_native_logits = None
            self.initial_native_logits = None
            self.history: list[dict[str, Any]] = []
            self.gradient_history: list[dict[str, Any]] = []
            self.boundary_gradient_seen = False
            self._feature_shape = None
            self._capture_eval = False
            self.eval_native_snapshot = None
            self.fusion_ab.register_forward_hook(self._capture_feature)
            for parameter in self.boundary_head.parameters():
                parameter.register_hook(self._record_gradient)

        def train(self, mode: bool = True):
            # A detached probe must measure native output invariance, not
            # stochastic DropPath/Dropout noise from the frozen teacher.
            super().train(mode)
            if mode and self.diagnostic_mode == "head_only":
                for name, module in self.named_children():
                    if name != "boundary_head":
                        module.eval()
                self.boundary_head.train(True)
            return self

        def _capture_feature(self, module, inputs, output):
            self._feature = output
            self._feature_shape = list(output.shape)

        def _record_gradient(self, gradient):
            self.boundary_gradient_seen = bool(torch.isfinite(gradient).all() and gradient.detach().abs().max() > 0)
            return gradient

        def forward(self, x, batch_data_samples):
            if self.diagnostic_mode == "head_only":
                for name, module in self.named_children():
                    if name != "boundary_head":
                        module.eval()
            self._feature = None
            native_logits = super().forward(x, batch_data_samples)
            if self._feature is None:
                raise RuntimeError("diagnostic boundary feature hook did not fire")
            feature = self._feature.detach() if self.diagnostic_mode == "head_only" else self._feature
            boundary_logits = self.boundary_head(feature)
            shape = batch_data_samples[0].metainfo["img_shape"]
            boundary_logits = F.interpolate(boundary_logits, size=shape, mode="bilinear", align_corners=False)
            self.last_boundary_logits = boundary_logits
            self.last_native_logits = native_logits
            if self._capture_eval:
                self.eval_native_snapshot = native_logits.detach().float().cpu()
            if self.initial_native_logits is None:
                self.initial_native_logits = native_logits.detach().float().cpu()
            return native_logits

        @staticmethod
        def _sum_native(losses):
            values = [value for key, value in losses.items() if str(key).startswith("loss_") and key != "loss_boundary" and hasattr(value, "ndim")]
            return sum(values) if values else None

        def _boundary_loss(self, boundary, valid, logits):
            selected_logits = logits[valid.bool()]
            selected_target = boundary[valid.bool()]
            if selected_logits.numel() == 0:
                return logits.sum() * 0.0
            if self.boundary_loss_variant == "l1":
                positive = torch.as_tensor(self.boundary_positive_weight, device=logits.device, dtype=logits.dtype)
                bce = F.binary_cross_entropy_with_logits(selected_logits, selected_target, pos_weight=positive)
            else:
                bce = F.binary_cross_entropy_with_logits(selected_logits, selected_target)
            probability = torch.sigmoid(selected_logits)
            dice = 1.0 - (2.0 * (probability * selected_target).sum() + 1.0) / (probability.sum() + selected_target.sum() + 1.0)
            return bce + dice

        def loss(self, inputs, batch_data_samples, train_cfg, finetune_cfg):
            native_logits = self.forward(inputs, batch_data_samples)
            native_losses = self.loss_by_feat(native_logits, batch_data_samples)
            boundary, valid = _target(torch, F, batch_data_samples, self.boundary_width)
            boundary = boundary.to(device=native_logits.device)
            valid = valid.to(device=native_logits.device)
            boundary_loss = self._boundary_loss(boundary, valid, self.last_boundary_logits)
            if not torch.isfinite(boundary_loss):
                raise FloatingPointError("diagnostic boundary loss is NaN/Inf")
            native_total = self._sum_native(native_losses)
            if self.diagnostic_mode == "head_only":
                total = boundary_loss
            else:
                total = boundary_loss + (native_total if native_total is not None else native_logits.sum() * 0.0)
            if self.diagnostic_mode in {"shared_last_block", "original_shared_scope"} and native_total is not None:
                shared = [parameter for name, parameter in self.named_parameters() if parameter.requires_grad and not name.startswith("boundary_head.")]
                if self.diagnostic_mode == "shared_last_block":
                    shared = [parameter for name, parameter in self.named_parameters() if parameter.requires_grad and (name.startswith("fusion_ab.") or name.startswith("boundary_head.")) and not name.startswith("boundary_head.")]
                native_grads = torch.autograd.grad(native_total, shared, retain_graph=True, allow_unused=True)
                boundary_grads = torch.autograd.grad(boundary_loss, shared, retain_graph=True, allow_unused=True)
                ng = torch.cat([(g.detach().float() if g is not None else torch.zeros_like(parameter).float()).reshape(-1) for g, parameter in zip(native_grads, shared)]) if shared else native_logits.new_zeros(1)
                bg = torch.cat([(g.detach().float() if g is not None else torch.zeros_like(parameter).float()).reshape(-1) for g, parameter in zip(boundary_grads, shared)]) if shared else native_logits.new_zeros(1)
                cosine = torch.dot(ng, bg) / (torch.linalg.vector_norm(ng) * torch.linalg.vector_norm(bg) + 1e-12)
                self.gradient_history.append({"iteration": len(self.history), "native_gradient_norm": float(torch.linalg.vector_norm(ng).detach().cpu()), "boundary_gradient_norm": float(torch.linalg.vector_norm(bg).detach().cpu()), "gradient_cosine": float(cosine.detach().cpu())})
            self.history.append({
                "iteration": len(self.history),
                "boundary_loss": float(boundary_loss.detach().float().cpu()),
                "native_loss": float(native_total.detach().float().cpu()) if native_total is not None else None,
                "boundary_output_std": float(self.last_boundary_logits.detach().float().std().cpu()),
                "boundary_output_mean": float(self.last_boundary_logits.detach().float().mean().cpu()),
            })
            return {"loss_boundary": boundary_loss, "loss_native": native_total if native_total is not None else boundary_loss * 0.0}

    return B2DiagnosticBoundaryDecoder


def make_lists(path: Path):
    if has_test(path):
        raise RuntimeError(f"locked Test path rejected: {path}")
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if has_test(line):
            raise RuntimeError(f"locked Test path in explicit list: {line}")


def deterministic_dataset(cfg, copy):
    # The native config wraps its train dataset in MultiImageMixDataset.  A
    # copied validation dataset gives a deterministic load/annotation/pack
    # path without mutating the source config or changing the prior B2 run.
    cfg.train_dataloader.dataset = copy.deepcopy(cfg.val_dataloader.dataset)
    cfg.train_dataloader.dataset.data_list = cfg.train_data_list
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.train_dataloader.sampler.shuffle = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument("--mode", choices=("inspect", "head_only", "shared_last_block", "original_shared_scope"), default="inspect")
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--boundary-positive-weight", type=float, default=1.0)
    parser.add_argument("--boundary-loss-variant", choices=("l0", "l1"), default="l0")
    args = parser.parse_args()
    result: dict[str, Any] = {"status": "failed", "mode": args.mode, "test_used": False, "test_pixel_reads": 0}
    try:
        for path in (args.repo, args.config, args.checkpoint, args.work_dir, args.train_list, args.val_list):
            if has_test(path):
                raise RuntimeError(f"locked Test path rejected before Runner construction: {path}")
        make_lists(args.train_list); make_lists(args.val_list)
        sys.path.insert(0, str(args.repo))
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from mmengine.config import Config
        from mmengine.registry import DefaultScope
        import opencd, opencd.datasets, opencd.evaluation, opencd.models, opencd.visualization  # noqa: F401
        from opencd.registry import MODELS
        from opencd.models.decode_heads.foundation_decoder import Foundation_Decoder_swin_v1
        from mmengine.runner import Runner
        DefaultScope.get_instance("opencd", scope_name="opencd")
        decoder_cls = register_decoder(torch, nn, F, MODELS, Foundation_Decoder_swin_v1)
        cfg = Config.fromfile(str(args.config))
        cfg.model.decode_head.type = decoder_cls.__name__
        cfg.model.decode_head.boundary_width = 2
        cfg.model.decode_head.boundary_channels = 64
        cfg.model.decode_head.diagnostic_mode = args.mode
        cfg.model.decode_head.boundary_loss_variant = args.boundary_loss_variant
        cfg.model.decode_head.boundary_positive_weight = float(args.boundary_positive_weight)
        cfg.work_dir = str(args.work_dir)
        cfg.train_cfg.max_iters = int(args.max_iters)
        cfg.train_cfg.val_interval = int(args.max_iters)
        cfg.val_interval = int(args.max_iters)
        cfg.default_hooks.checkpoint.interval = int(args.max_iters)
        cfg.default_hooks.logger.interval = max(1, min(100, int(args.max_iters)))
        cfg.train_data_list = str(args.train_list)
        cfg.test_data_list = str(args.val_list)
        cfg.val_dataloader.dataset.data_list = str(args.val_list)
        cfg.test_dataloader.dataset.data_list = str(args.val_list)
        for loader in (cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
            loader.num_workers = 0; loader.persistent_workers = False; loader.batch_size = 1
        deterministic_dataset(cfg, __import__("copy"))
        cfg.param_scheduler = []
        cfg.optim_wrapper.optimizer.lr = 1e-3 if args.mode == "head_only" else 1e-5
        cfg.optim_wrapper.optimizer.weight_decay = 0.0
        cfg.optim_wrapper.clip_grad.max_norm = 1.0
        cfg.resume = False; cfg.load_from = None
        cfg.model.finetune_cfg = ["neck", "decoder", "ab mask head", "cd mask head", "ab query head", "cd query head"]
        cfg.model.decode_head.finetune_cfg = cfg.model.finetune_cfg
        runner = Runner.from_cfg(cfg)
        runner.model.init_weights = lambda: None
        payload = torch.load(str(args.checkpoint), map_location="cpu")
        state = payload.get("state_dict", payload)
        inherited = runner.model.load_state_dict(state, strict=False)
        invalid_missing = [key for key in inherited.missing_keys if not key.startswith("decode_head.boundary_head.")]
        if inherited.unexpected_keys or invalid_missing:
            raise RuntimeError(f"checkpoint inheritance failed: missing={invalid_missing}, unexpected={inherited.unexpected_keys}")
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        decoder = model.decode_head
        # Freeze first, then open only the diagnostic scope.
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in decoder.boundary_head.parameters():
            parameter.requires_grad = True
        if args.mode == "shared_last_block":
            for name, parameter in decoder.fusion_ab.named_parameters():
                parameter.requires_grad = True
        elif args.mode == "original_shared_scope":
            for name, parameter in model.named_parameters():
                if name.startswith("decode_head.") and not name.startswith("decode_head.boundary_head.") and not name.startswith("backbone."):
                    parameter.requires_grad = True
        # Runner builds the optimizer before the diagnostic freeze is applied.
        # Remove frozen tensors from its groups explicitly so a detached probe
        # cannot silently update a native parameter.
        optimizer = getattr(runner.optim_wrapper, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "param_groups"):
            for group in optimizer.param_groups:
                group["params"] = [parameter for parameter in group["params"] if parameter.requires_grad]
        registry = [{"name": name, "requires_grad": bool(parameter.requires_grad), "count": int(parameter.numel())} for name, parameter in model.named_parameters()]
        result.update({"checkpoint": str(args.checkpoint.resolve()), "checkpoint_state_keys": len(state), "missing_boundary_keys": list(inherited.missing_keys), "unexpected_keys": list(inherited.unexpected_keys), "feature_module_path": "decode_head.fusion_ab", "feature_stride": 4, "trainable_parameters": sum(row["count"] for row in registry if row["requires_grad"]), "total_parameters": sum(row["count"] for row in registry), "frozen_parameters": sum(row["count"] for row in registry if not row["requires_grad"]), "trainable_parameter_names": [row["name"] for row in registry if row["requires_grad"]]})
        args.work_dir.mkdir(parents=True, exist_ok=True)
        (args.work_dir / "PARAMETER_REGISTRY.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        native_before = None
        native_after = None
        if args.mode != "inspect":
            model.eval()
            decoder._capture_eval = True
            runner.val()
            native_before = decoder.eval_native_snapshot.clone() if decoder.eval_native_snapshot is not None else None
            decoder._capture_eval = False
        if args.mode == "inspect":
            runner.val()
        else:
            runner.train()
            model.eval()
            decoder._capture_eval = True
            runner.val()
            native_after = decoder.eval_native_snapshot.clone() if decoder.eval_native_snapshot is not None else None
            decoder._capture_eval = False
        max_native_diff = None
        if native_before is not None and native_after is not None:
            max_native_diff = float((native_after - native_before).abs().max())
        if decoder.initial_native_logits is not None and decoder.last_native_logits is not None:
            max_native_diff = float((decoder.last_native_logits.detach().float().cpu() - decoder.initial_native_logits).abs().max())
        result.update({"status": "pass", "feature_shape": decoder._feature_shape, "history": decoder.history, "gradient_history": decoder.gradient_history, "boundary_gradient_nonzero": bool(decoder.boundary_gradient_seen), "boundary_output_std": float(decoder.history[-1]["boundary_output_std"]) if decoder.history else None, "boundary_output_nonconstant": bool(decoder.history and decoder.history[-1]["boundary_output_std"] > 1e-6), "building_logits_max_abs_diff": max_native_diff, "building_logits_unchanged_within_float_error": bool(max_native_diff is not None and max_native_diff <= 1e-6) if args.mode == "head_only" else None, "nonfinite": any(not math.isfinite(float(row["boundary_loss"])) for row in decoder.history), "work_dir": str(args.work_dir.resolve())})
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
