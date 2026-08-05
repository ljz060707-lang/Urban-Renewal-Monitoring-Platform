"""Native Open-CD worker used by the guarded B1 pilot.

This file is invoked only through the RSBuilding Python environment.  It
receives explicit Train/Validation manifests and refuses any path containing
the locked Test split before constructing an MMEngine Runner.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def has_test(value: str) -> bool:
    lowered = str(value).lower().replace("/", "\\")
    return "\\test\\" in lowered or lowered.endswith("\\test.txt")


def main() -> int:
    if len(sys.argv) != 9:
        raise SystemExit("usage: native_b1_worker.py repo config checkpoint work_dir train_list val_list max_iters mode")
    repo, config_path, checkpoint, work_dir, train_list, val_list = map(Path, sys.argv[1:7])
    max_iters, mode = int(sys.argv[7]), sys.argv[8]
    result = {"status": "failed", "mode": mode, "test_used": False, "test_pixel_reads": 0}
    try:
        for path in (train_list, val_list, config_path, checkpoint):
            if has_test(path) or has_test(path.read_text(encoding="utf-8", errors="ignore") if path.suffix == ".txt" else ""):
                raise RuntimeError(f"locked Test path rejected before Runner construction: {path}")
        sys.path.insert(0, str(repo))
        import torch
        from mmengine.config import Config
        from mmengine.registry import DefaultScope
        import opencd, opencd.datasets, opencd.evaluation, opencd.models, opencd.visualization  # noqa: F401
        DefaultScope.get_instance("opencd", scope_name="opencd")
        cfg = Config.fromfile(str(config_path))
        cfg.load_from = str(checkpoint)
        cfg.work_dir = str(work_dir)
        cfg.train_cfg.max_iters = max_iters
        # B1 is a bounded control experiment: retain the native full
        # Validation loop and observe it every 100 iterations.  The tiny
        # sanity runs validate only at their end to keep them fast.
        validation_interval = 100 if mode == "b1" and max_iters >= 100 else max_iters
        cfg.train_cfg.val_interval = validation_interval
        cfg.val_interval = validation_interval
        cfg.default_hooks.checkpoint.interval = validation_interval
        cfg.default_hooks.logger.interval = max(1, min(20, max_iters))
        cfg.train_data_list = str(train_list)
        cfg.test_data_list = str(val_list)
        for loader in (cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
            loader.num_workers = 0
            loader.persistent_workers = False
            loader.batch_size = 1
        # Keep the native MultiImageMixDataset wrapper.  It owns the final
        # MultiImgPackSegInputs transform which converts the paired image/
        # annotation dictionary into the ``inputs``/``data_samples`` contract
        # expected by the Open-CD model.  Replacing it with the inner dataset
        # makes the runner reach ``train_step`` with an unpacked dictionary
        # (KeyError: ``inputs``).  We only replace the inner data list, so the
        # native packing and annotation transforms remain identical to the
        # production pipeline.
        if hasattr(cfg.train_dataloader.dataset, "dataset"):
            cfg.train_dataloader.dataset.dataset.data_list = str(train_list)
            # The native mosaic transform samples three *other* items.  A
            # one-sample overfit sanity check cannot satisfy that contract;
            # skip only the mosaic stage for that explicitly named mode while
            # retaining the native crop/augmentation and final pack transform.
            if mode == "one_sample":
                cfg.train_dataloader.dataset.skip_type_keys = ["RandomMosaic_Modified"]
        else:
            cfg.train_dataloader.dataset.data_list = str(train_list)
        cfg.val_dataloader.dataset.data_list = str(val_list)
        cfg.test_dataloader.dataset.data_list = str(val_list)
        cfg.model.finetune_cfg = ["neck", "decoder", "ab mask head", "cd mask head", "ab query head", "cd query head"]
        cfg.model.decode_head.finetune_cfg = cfg.model.finetune_cfg
        cfg.optim_wrapper.optimizer.lr = min(float(cfg.optim_wrapper.optimizer.lr), 1e-5)
        cfg.resume = False
        from mmengine.runner import Runner
        runner = Runner.from_cfg(cfg)
        # Runner.train() normally initializes random backbone weights before
        # loading ``load_from``.  That path invokes CUDA erfinv and requires a
        # CUDA NVRTC builtins DLL that is absent in this native environment.
        # B1 is a strict iter2000 continuation, so random initialization is
        # scientifically wrong and technically unnecessary.  Disable only
        # that pre-load initializer; Runner.load_or_resume() still performs a
        # strict checkpoint load immediately afterwards.
        runner.model.init_weights = lambda: None
        registry = []
        for name, param in runner.model.named_parameters():
            registry.append({"name": name, "shape": list(param.shape), "count": int(param.numel()), "requires_grad": bool(param.requires_grad), "module": "decode_head" if name.startswith("decode_head") else "backbone" if name.startswith("backbone") else "other", "lr_group": "b1_decoder" if param.requires_grad else "frozen", "lr": float(cfg.optim_wrapper.optimizer.lr) if param.requires_grad else 0.0, "weight_decay": float(cfg.optim_wrapper.optimizer.weight_decay), "initial_digest": float(param.detach().float().sum().item())})
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "TRAINABLE_PARAMETER_REGISTRY.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        runner.train()
        result.update({"status": "pass", "max_iters": max_iters, "trainable_parameters": sum(item["count"] for item in registry if item["requires_grad"]), "total_parameters": sum(item["count"] for item in registry), "work_dir": str(work_dir)})
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
