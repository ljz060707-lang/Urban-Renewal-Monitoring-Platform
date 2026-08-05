"""Compare the project adapter with the native RSBuilding input contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--native-audit", type=Path, default=None)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    native_path = args.native_audit or run / "02_native_recovery" / "NATIVE_CONFIG_AUDIT.json"
    native = json.loads(native_path.read_text(encoding="utf-8")) if native_path.exists() else {}
    pre = native.get("data_preprocessor", {})
    native_contract = {
        "channel_count": 6,
        "bgr_to_rgb": bool(pre.get("bgr_to_rgb", True)),
        "mean": pre.get("mean", [123.675, 116.28, 103.53] * 2),
        "std": pre.get("std", [58.395, 57.12, 57.375] * 2),
        "size_divisor": pre.get("size_divisor", 32),
        "image_resize": "native pipeline preserves 512 crop/pad contract; validation uses native full image shape",
        "label_threshold": 128,
        "label_building_semantics": "uint8 >=128 -> 1",
        "augmentation": ["RandomCrop 512", "horizontal/vertical flip", "ExchangeTime", "PhotoMetricDistortion", "RandomMosaic"],
    }
    project_contract = {
        "channel_count": 3,
        "bgr_to_rgb": False,
        "mean": None,
        "std": None,
        "size_divisor": None,
        "image_resize": "rasterio bilinear to configured image_size",
        "label_threshold": 0,
        "label_building_semantics": "label > 0 -> 1",
        "augmentation": ["deterministic horizontal flip in existing adapter"],
    }
    mismatches = {
        "channel_count": True,
        "normalization": True,
        "color_order": native_contract["bgr_to_rgb"] != project_contract["bgr_to_rgb"],
        "label_semantics": True,
        "crop_and_augmentation": True,
    }
    payload = {
        "status": "blocked_project_adapter_not_native_equivalent",
        "native_contract": native_contract,
        "project_adapter_current_contract": project_contract,
        "mismatches": mismatches,
        "required_before_formal_candidate": [
            "make label polarity come from SELECTED_WHU_LABEL_CONTRACT.json",
            "provide an explicit native 6-channel/normalization path for native model work",
            "reproduce native label threshold and ignore behavior",
            "run one-sample and eight-sample overfit after contract approval",
        ],
        "source_config": str(native.get("config_path", "")),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_pixel_reads": 0,
        "test_used": False,
    }
    write_json(run / "03_input_pipeline" / "INPUT_PIPELINE_COMPARISON.json", payload)
    report = [
        "# Native RSBuilding input pipeline comparison",
        "",
        "The existing project adapter is not equivalent to the native model contract.",
        "",
        f"- Native channels: `{native_contract['channel_count']}`; project adapter: `{project_contract['channel_count']}`",
        f"- Native BGR→RGB: `{native_contract['bgr_to_rgb']}`; project adapter: `{project_contract['bgr_to_rgb']}`",
        f"- Native mean/std: `{native_contract['mean']}` / `{native_contract['std']}`",
        f"- Native label semantics: `{native_contract['label_building_semantics']}`",
        f"- Local contract audit: `outputs/.../01_label_contract/SELECTED_WHU_LABEL_CONTRACT.json`",
        "- Formal training gate: `blocked_project_adapter_not_native_equivalent` until the explicit contract is applied.",
        "",
        "No Test pixels were read.",
    ]
    (run / "03_input_pipeline" / "INPUT_PIPELINE_COMPARISON.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
