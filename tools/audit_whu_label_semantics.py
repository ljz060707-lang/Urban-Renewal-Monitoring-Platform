"""Audit WHU Train/Validation label values and select an explicit label contract.

This tool intentionally never enumerates or opens ``WHU_Building/test``.  The
local WHU copy is known to contain a 0/1 mask whose polarity must not be
inferred from a filename or from an old project adapter.  The result records
the native RSBuilding expectation, the observed values, a deterministic visual
contrast diagnostic, and a human-readable preview grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_NATIVE = {
    "native_label_threshold": 128,
    "native_building_values": [1, 255],
    "native_background_values": [0],
    "native_class_order": ["unchanged / no_building", "changed / building"],
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    if split not in {"train", "validation"}:
        raise ValueError("WHU label audit accepts only train and validation")
    folder = root / ("val" if split == "validation" else "train")
    image_dir, label_dir = folder / "image", folder / "label"
    images = {p.stem: p for p in image_dir.glob("*.tif")}
    labels = {p.stem: p for p in label_dir.glob("*.tif")}
    if set(images) != set(labels):
        raise ValueError(f"pairing mismatch for {split}: images={len(images)} labels={len(labels)}")
    return [(images[key], labels[key]) for key in sorted(images)]


def stretch(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    channels: list[np.ndarray] = []
    for band in arr[:3]:
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            channels.append(np.zeros_like(band, dtype=np.uint8))
            continue
        lo, hi = np.percentile(finite, [2, 98])
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            channels.append(np.zeros_like(band, dtype=np.uint8))
        else:
            channels.append(np.clip((band - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8))
    while len(channels) < 3:
        channels.append(channels[-1].copy() if channels else np.zeros(arr.shape[-2:], dtype=np.uint8))
    return np.stack(channels, axis=-1)


def save_previews(selected: list[tuple[Path, Path, str, float]], out: Path, building_values: list[int]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - optional reporting dependency
        write_json(out / "PREVIEW_GENERATION_FAILURE.json", {"status": "failed", "error": str(exc)})
        return
    tile_w, tile_h = 192, 192
    columns, rows = 4, max(1, math.ceil(len(selected) / 4))
    grid = Image.new("RGB", (columns * tile_w, rows * tile_h), (245, 245, 245))
    polarity = Image.new("RGB", (columns * tile_w * 2, rows * tile_h), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    pdraw = ImageDraw.Draw(polarity)
    for idx, (image_path, label_path, split, fraction) in enumerate(selected):
        import rasterio  # type: ignore
        with rasterio.open(image_path) as src:
            rgb = stretch(src.read())
        with rasterio.open(label_path) as src:
            mask = np.asarray(src.read(1))
        image = Image.fromarray(rgb).resize((tile_w, tile_h))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        selected_building = np.isin(mask, np.asarray(building_values)).astype(np.uint8)
        overlay.putalpha(Image.fromarray(selected_building * 150).resize(image.size))
        building = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        xy = ((idx % columns) * tile_w, (idx // columns) * tile_h)
        grid.paste(building, xy)
        draw.text((xy[0] + 3, xy[1] + 3), f"{split}:{image_path.stem} z={fraction:.3f}", fill=(255, 0, 0))
        # Side-by-side polarity comparison: native-positive (1) and selected
        # local-positive (0).  This makes an inverted local mask obvious.
        native = Image.fromarray(((mask > 0).astype(np.uint8) * 255)).convert("L").resize((tile_w, tile_h))
        local = Image.fromarray(((mask == 0).astype(np.uint8) * 255)).convert("L").resize((tile_w, tile_h))
        px = ((idx % columns) * tile_w * 2, (idx // columns) * tile_h)
        polarity.paste(Image.merge("RGB", (native, native, native)), px)
        polarity.paste(Image.merge("RGB", (local, local, local)), (px[0] + tile_w, px[1]))
        pdraw.text((px[0] + 3, px[1] + 3), "native 1=building", fill=(255, 0, 0))
        pdraw.text((px[0] + tile_w + 3, px[1] + 3), "local 0=building", fill=(255, 0, 0))
    grid.save(out / "WHU_LABEL_PREVIEW_GRID.png")
    polarity.save(out / "WHU_MASK_POLARITY_COMPARISON.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=100)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.run_dir / "01_label_contract"
    out.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise SystemExit(f"WHU root missing: {root}")
    all_rows: list[dict[str, Any]] = []
    unique_counts: dict[str, int] = {}
    preview_rows: list[tuple[Path, Path, str, float]] = []
    contrast_values: list[float] = []
    edge_contrast_values: list[float] = []
    sample_limit = max(1, int(args.sample_count))
    for split in ("train", "validation"):
        split_pairs = pairs(root, split)
        for index, (image_path, label_path) in enumerate(split_pairs):
            # Explicit guard protects the locked Test directory even if a
            # malformed symlink or manifest is introduced later.
            if "\\test\\" in str(image_path).lower() or "/test/" in str(image_path).lower():
                raise RuntimeError("locked Test path encountered; audit aborted before pixels were read")
            import rasterio  # type: ignore
            with rasterio.open(label_path) as src:
                label = np.asarray(src.read(1))
                height, width = src.height, src.width
                dtype = str(src.dtypes[0])
            values, counts = np.unique(label, return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                key = str(int(value)) if isinstance(value, (int, np.integer)) else str(value)
                unique_counts[key] = unique_counts.get(key, 0) + int(count)
            zero_fraction = float(np.mean(label == 0))
            one_fraction = float(np.mean(label > 0))
            contrast = None
            if index < sample_limit:
                with rasterio.open(image_path) as src:
                    rgb = src.read([1, 2, 3]).astype(np.float32)
                luminance = rgb.mean(axis=0)
                from scipy import ndimage  # type: ignore
                gradient = np.hypot(ndimage.sobel(luminance, axis=0), ndimage.sobel(luminance, axis=1))
                zero = label == 0
                one = label > 0
                if zero.any() and one.any():
                    contrast = float(luminance[zero].mean() - luminance[one].mean())
                    contrast_values.append(contrast)
                    # Buildings in the local RGB tiles have substantially
                    # more local edge structure than the complement.  This is
                    # a diagnostic tie-breaker; the native class contract and
                    # the fixed visual grid remain the authority.
                    edge_contrast_values.append(float(gradient[one].mean() - gradient[zero].mean()))
                preview_rows.append((image_path, label_path, split, zero_fraction))
            all_rows.append({
                "sample_id": f"whu_{split}_{image_path.stem}",
                "split": split,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "label_dtype": dtype,
                "width": width,
                "height": height,
                "zero_fraction": zero_fraction,
                "one_fraction": one_fraction,
                "visual_zero_minus_one_luminance": contrast,
            })
    csv_path = out / "WHU_POSITIVE_FRACTION_BY_SAMPLE.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    dist_path = out / "WHU_LABEL_VALUE_DISTRIBUTION.csv"
    with dist_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["value", "pixel_count"])
        writer.writerows(sorted(unique_counts.items(), key=lambda item: item[0]))
    # The sampled local masks consistently show zero-valued regions aligned
    # with building footprints and substantially darker than the complement.
    median_contrast = float(np.median(contrast_values)) if contrast_values else None
    median_edge_contrast = float(np.median(edge_contrast_values)) if edge_contrast_values else None
    observed_values = sorted(int(v) for v in unique_counts if v.lstrip("-").isdigit())
    # Native Open-CD expects a positive building class and mmcv decodes the
    # local 0/1 TIFF as 0/255.  The deterministic edge diagnostic agrees with
    # that interpretation (the nonzero regions are the structured footprints).
    local_zero_is_building = observed_values == [0, 1] and (median_edge_contrast is not None and median_edge_contrast < 0)
    selected = {
        "label_polarity": "zero_is_building" if local_zero_is_building else "nonzero_is_building",
        "building_values": [0] if local_zero_is_building else [1],
        "background_values": [1] if local_zero_is_building else [0],
        "native_building_values": EXPECTED_NATIVE["native_building_values"],
        "native_label_threshold": EXPECTED_NATIVE["native_label_threshold"],
        "native_local_polarity_match": not local_zero_is_building,
        "visual_evidence": {
            "sample_count_per_split": sample_limit,
            "median_zero_minus_nonzero_luminance": median_contrast,
            "median_nonzero_minus_zero_edge_strength": median_edge_contrast,
            "positive_contrast_means_selected_local_zero": bool(local_zero_is_building),
            "method": "native class contract plus deterministic RGB edge diagnostic and fixed preview grid; human review remains required",
        },
        "contract_status": "suspect_native_mismatch" if local_zero_is_building else "verified_pending_manual_preview",
        "reason": "local WHU labels are uint8 0/1 and edge/preview evidence conflicts with native semantics" if local_zero_is_building else "local polarity agrees with native threshold semantics; preview still requires human sign-off",
        "test_pixel_reads": 0,
        "test_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # A semantic match is not yet a training authorization: the fixed preview
    # still needs a human sign-off.  Keep the gate closed until the audit is
    # explicitly promoted to ``verified``.
    selected["training_enabled"] = selected["contract_status"] == "verified"
    write_json(out / "SELECTED_WHU_LABEL_CONTRACT.json", selected)
    audit = {
        "status": selected["contract_status"],
        "root": str(root),
        "splits_read": ["train", "validation"],
        "train_count": sum(r["split"] == "train" for r in all_rows),
        "validation_count": sum(r["split"] == "validation" for r in all_rows),
        "unique_values": observed_values,
        "value_pixel_counts": unique_counts,
        "native_expectation": EXPECTED_NATIVE,
        "selected_contract": selected,
        "sample_count_per_split": sample_limit,
        "preview_paths": ["WHU_LABEL_PREVIEW_GRID.png", "WHU_MASK_POLARITY_COMPARISON.png"],
        "training_gate": "blocked_until_contract_review" if not selected["training_enabled"] else "manual_preview_required",
        "test_pixel_reads": 0,
        "test_used": False,
        "raw_data_modified": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "WHU_LABEL_SEMANTICS_AUDIT.json", audit)
    save_previews(preview_rows, out, selected["building_values"])
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    return 0 if selected["contract_status"] != "suspect_native_mismatch" else 2


if __name__ == "__main__":
    raise SystemExit(main())
