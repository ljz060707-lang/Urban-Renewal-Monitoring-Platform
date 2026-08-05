"""Inspect the native RSBuilding runtime and iter_2000 checkpoint schema.

All model imports and checkpoint deserialization happen in the separately
managed native RSBuilding environment.  No dataset is opened by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
DEFAULT_REPO = Path(r"D:\Data Set\RSBuilding\repo")
DEFAULT_NATIVE = Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe")
DEFAULT_CONFIG = DEFAULT_REPO / "configs_finetune" / "Swin_T" / "Swin_T_whu.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


NATIVE_CODE = r'''
import hashlib, json, os, sys, traceback
from pathlib import Path
repo = Path(sys.argv[1]).resolve()
checkpoint = Path(sys.argv[2]).resolve()
config_path = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(repo))
result = {"repo": str(repo), "checkpoint": str(checkpoint), "config": str(config_path), "imports": {}, "checkpoint": {}, "config_audit": {}, "model_load": {}}
def simple(value):
    if value is None or isinstance(value, (bool, int, float, str)): return value
    if isinstance(value, (list, tuple)): return [simple(v) for v in value[:20]]
    if isinstance(value, dict): return {str(k): simple(v) for k, v in list(value.items())[:40]}
    return str(value)
try:
    import torch
    result["imports"]["torch"] = torch.__version__
    import mmengine
    result["imports"]["mmengine"] = getattr(mmengine, "__version__", "unknown")
    import mmcv
    result["imports"]["mmcv"] = getattr(mmcv, "__version__", "unknown")
    import mmseg
    result["imports"]["mmseg"] = getattr(mmseg, "__version__", "unknown")
    import opencd
    # The native package intentionally keeps registration imports explicit.
    import opencd.models, opencd.datasets, opencd.evaluation, opencd.visualization
    result["imports"]["opencd"] = getattr(opencd, "__version__", "unknown")
    result["imports"]["opencd_path"] = str(Path(opencd.__file__).resolve())
    result["cuda"] = {"is_available": bool(torch.cuda.is_available()), "device_count": int(torch.cuda.device_count())}
except Exception as exc:
    result["imports_error"] = {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
try:
    import torch
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload.get("model_state_dict", {})) if isinstance(payload, dict) else {}
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    prefixes = {}
    for key in state:
        prefix = str(key).split(".", 1)[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    result["checkpoint"] = {
        "top_level_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        "state_dict_key": "state_dict" if isinstance(payload, dict) and "state_dict" in payload else "model_state_dict",
        "state_key_count": len(state),
        "state_prefix_counts": prefixes,
        "meta_keys": sorted(meta.keys()) if isinstance(meta, dict) else [],
        "meta_selected": {k: simple(meta.get(k)) for k in ("epoch", "iter", "seed", "experiment_name", "dataset_meta", "mmengine_version", "time") if isinstance(meta, dict) and k in meta},
        "cfg_sha256": hashlib.sha256(str(meta.get("cfg", "")).encode("utf-8")).hexdigest() if isinstance(meta, dict) else None,
        "cfg_excerpt": str(meta.get("cfg", ""))[:1200] if isinstance(meta, dict) else "",
    }
except Exception as exc:
    result["checkpoint_error"] = {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
try:
    from mmengine.config import Config
    cfg = Config.fromfile(str(config_path))
    result["config_audit"] = {
        "model_type": str(cfg.model.get("type")) if "model" in cfg else None,
        "backbone_type": str(cfg.model.get("backbone", {}).get("type")) if "model" in cfg else None,
        "decode_head_type": str(cfg.model.get("decode_head", {}).get("type")) if "model" in cfg else None,
        "data_preprocessor": simple(cfg.model.get("data_preprocessor", cfg.get("data_preprocessor", {}))) if "model" in cfg else {},
        "train_data_list": str(cfg.get("train_data_list", "")),
        "val_data_list": str(cfg.get("val_dataloader", {}).get("dataset", {}).get("data_list", cfg.get("test_data_list", ""))),
        "test_data_list": str(cfg.get("test_dataloader", {}).get("dataset", {}).get("data_list", "")),
        "train_loop": simple(cfg.get("train_cfg", {})),
        "val_loop": simple(cfg.get("val_cfg", {})),
    }
    for key in ("train_data_list", "val_data_list", "test_data_list"):
        value = result["config_audit"].get(key) or ""
        result["config_audit"][key + "_contains_locked_test"] = ("\\test\\" in value.lower() or "/test/" in value.lower() or value.lower().endswith("\\test.txt") or value.lower().endswith("/test.txt"))
except Exception as exc:
    result["config_error"] = {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
try:
    import opencd.models, opencd.datasets, opencd.evaluation, opencd.visualization
    from mmengine.registry import DefaultScope
    DefaultScope.get_instance('opencd', scope_name='opencd')
    from opencd.registry import MODELS
    cfg = Config.fromfile(str(config_path))
    model = MODELS.build(cfg.model)
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload.get("model_state_dict", {}))
    incompatible = model.load_state_dict(state, strict=True)
    result["model_load"] = {"status": "pass", "missing_keys": list(getattr(incompatible, "missing_keys", [])), "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])), "parameter_count": sum(p.numel() for p in model.parameters())}
except Exception as exc:
    result["model_load"] = {"status": "failed", "type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
print(json.dumps(result, ensure_ascii=False, default=str))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--native-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--native-python", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / "02_native_recovery"
    out.mkdir(parents=True, exist_ok=True)
    if not args.native_python.exists():
        raise SystemExit(f"native Python missing: {args.native_python}")
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint missing: {args.checkpoint}")
    digest = sha256(args.checkpoint)
    if digest != EXPECTED_SHA:
        raise SystemExit(f"iter_2000 SHA256 mismatch: {digest}")
    completed = subprocess.run([str(args.native_python), "-c", NATIVE_CODE, str(args.native_repo), str(args.checkpoint), str(args.config)], capture_output=True, text=True)
    if not completed.stdout.strip():
        payload = {"status": "failed", "returncode": completed.returncode, "stderr": completed.stderr[-10000:]}
    else:
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception:
            payload = {"status": "failed", "returncode": completed.returncode, "stdout": completed.stdout[-10000:], "stderr": completed.stderr[-10000:]}
    payload.update({"created_at_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_sha256": digest, "expected_checkpoint_sha256": EXPECTED_SHA, "test_pixel_reads": 0, "test_used": False})
    payload["status"] = "pass" if completed.returncode == 0 and payload.get("model_load", {}).get("status") == "pass" and not payload.get("config_audit", {}).get("test_data_list_contains_locked_test") else "failed"
    write_json(out / "NATIVE_RUNTIME_AUDIT.json", payload)
    write_json(out / "CHECKPOINT_SCHEMA.json", {"checkpoint_sha256": digest, **payload.get("checkpoint", {}), "test_pixel_reads": 0})
    write_json(out / "NATIVE_CONFIG_AUDIT.json", {"config_path": str(args.config.resolve()), **payload.get("config_audit", {}), "test_pixel_reads": 0})
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
