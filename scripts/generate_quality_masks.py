#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from urban_monitor.config import load_config
from urban_monitor.manifest import load_manifest
from urban_monitor.quality import compose_quality_masks, generate_quality_mask
from urban_monitor.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conservative RGB quality masks and optionally update the manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    q = cfg.get("quality", {})
    records = load_manifest(cfg["manifest"], cfg["data_root"])
    root = Path(cfg["data_root"])
    generated: dict[str, Path] = {}
    composite_root = Path(cfg["output_root"]) / "composite_quality_masks"
    rgb_root = Path(cfg["output_root"]) / "rgb_quality_masks"
    for record in records:
        rgb = record.quality_mask_path
        if rgb is None or not rgb.exists():
            rgb = prepare_output(rgb_root / f"{record.date}.tif", force=args.force)
            generate_quality_mask(
                record.image_path,
                rgb,
                bright_threshold=int(q.get("bright_threshold", 248)),
                bright_max_saturation=float(q.get("bright_max_saturation", 0.12)),
                dark_threshold=int(q.get("dark_threshold", 0)),
                dilation_pixels=int(q.get("dilation_pixels", 2)),
                blur_block_size=int(q.get("blur_block_size", 128)),
                min_laplacian_variance=float(q.get("min_laplacian_variance", 0.0)),
                min_global_dynamic_range=float(q.get("min_global_dynamic_range", 12.0)),
            )
        target = prepare_output(composite_root / f"{record.date}.tif", force=args.force)
        compose_quality_masks(
            record.image_path,
            target,
            manual_mask_path=record.manual_mask_path,
            provider_mask_path=record.provider_mask_path,
            rgb_mask_path=rgb,
            registration_issue=record.registration_status == "failed",
        )
        generated[record.date] = target
        print(target)

    if args.update_manifest:
        manifest_path = Path(cfg["manifest"])
        frame = pd.read_csv(manifest_path, dtype=str, encoding="utf-8-sig").fillna("")
        if "quality_mask_path" not in frame.columns:
            frame["quality_mask_path"] = ""
        for idx, row in frame.iterrows():
            date = str(row["date"])[:10]
            target = generated[date]
            try:
                frame.at[idx, "quality_mask_path"] = str(target.relative_to(root))
            except ValueError:
                frame.at[idx, "quality_mask_path"] = str(target)
        frame.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        print(f"updated {manifest_path}")


if __name__ == "__main__":
    main()
