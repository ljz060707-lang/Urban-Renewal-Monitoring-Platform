from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject


QUALITY_REASON_CODES = {
    0: "usable",
    1: "nodata",
    2: "bright_cloud_or_glare",
    3: "dark_or_shadow_candidate",
    4: "blur",
    5: "low_contrast",
    6: "provider_invalid",
    7: "manual_invalid",
    8: "registration_issue",
    9: "occluded",
    10: "unknown",
}


def geometry_quality_fraction(mask_path: str | Path | None, geometry) -> float:
    """Return the fraction of pixels marked usable inside a geometry.

    Missing masks default to usable for backward compatibility. Real projects
    should generate or review a quality mask for every date.
    """
    if mask_path is None or not Path(mask_path).exists():
        return 1.0
    with rasterio.open(mask_path) as src:
        inside = geometry_mask([geometry], out_shape=(src.height, src.width), transform=src.transform, invert=True)
        if not inside.any():
            return 0.0
        values = src.read(1)
        return float(np.mean(values[inside] > 0))


def _to_uint8_rgb(data: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(data[:3] if data.shape[0] >= 3 else np.repeat(data[:1], 3, axis=0), 0, -1)
    if rgb.dtype == np.uint8:
        return rgb
    out = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        band = rgb[..., c].astype(np.float32)
        finite = np.isfinite(band)
        if not finite.any():
            continue
        lo, hi = np.nanpercentile(band[finite], [1, 99])
        out[..., c] = np.clip((band - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return out


def generate_quality_mask(
    image_path: str | Path,
    output_path: str | Path,
    *,
    bright_threshold: int = 248,
    bright_max_saturation: float = 0.12,
    dark_threshold: int = 0,
    dilation_pixels: int = 2,
    blur_block_size: int = 128,
    min_laplacian_variance: float = 0.0,
    min_global_dynamic_range: float = 12.0,
) -> Path:
    """Create a conservative RGB-only usable-area mask.

    Band 1 is binary usable/unusable. Band 2 stores reason codes:
    0 usable, 1 nodata, 2 bright cloud/glare, 3 extreme dark/shadow,
    4 locally blurred, 5 globally low contrast/hazy.

    RGB heuristics cannot replace provider QA or human review. They are designed
    to catch obvious failures and make uncertainty explicit, not certify image
    quality.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    with rasterio.open(image_path) as src:
        data = src.read()
        rgb = _to_uint8_rgb(data)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[..., 1].astype(np.float32) / 255.0

        finite = np.all(np.isfinite(data[: min(data.shape[0], 3)]), axis=0)
        nonzero = np.any(data[: min(data.shape[0], 3)] != 0, axis=0)
        usable = finite & nonzero
        reason = np.zeros((src.height, src.width), dtype=np.uint8)
        reason[~usable] = 1

        bright = (gray >= bright_threshold) & (saturation <= bright_max_saturation) & usable
        if dilation_pixels > 0 and bright.any():
            kernel = np.ones((2 * dilation_pixels + 1, 2 * dilation_pixels + 1), np.uint8)
            bright = cv2.dilate(bright.astype(np.uint8), kernel, iterations=1).astype(bool)
        reason[bright] = 2
        usable[bright] = False

        if dark_threshold > 0:
            dark = (gray <= dark_threshold) & (saturation <= 0.20) & usable
            reason[dark] = 3
            usable[dark] = False

        if min_laplacian_variance > 0:
            for r0 in range(0, src.height, blur_block_size):
                for c0 in range(0, src.width, blur_block_size):
                    block = gray[r0 : r0 + blur_block_size, c0 : c0 + blur_block_size]
                    if block.size == 0:
                        continue
                    variance = float(cv2.Laplacian(block, cv2.CV_64F).var())
                    if variance < min_laplacian_variance:
                        block_slice = np.s_[r0 : r0 + block.shape[0], c0 : c0 + block.shape[1]]
                        block_usable = usable[block_slice]
                        reason_block = reason[block_slice]
                        reason_block[block_usable] = 4
                        usable[block_slice] = False

        valid_gray = gray[finite & nonzero]
        if valid_gray.size and float(np.percentile(valid_gray, 98) - np.percentile(valid_gray, 2)) < min_global_dynamic_range:
            reason[usable] = 5
            usable[:] = False

        profile = src.profile.copy()
        profile.update(count=2, dtype="uint8", compress="deflate")
        profile.pop("nodata", None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(usable.astype(np.uint8), 1)
            dst.write(reason, 2)
            dst.set_band_description(1, "usable")
            dst.set_band_description(2, "quality_reason")
            dst.update_tags(reason_codes=";".join(f"{key}:{value}" for key, value in QUALITY_REASON_CODES.items()))
    return output_path


def _mask_on_reference(path: str | Path, reference) -> np.ndarray:
    with rasterio.open(path) as source:
        destination = np.zeros((reference.height, reference.width), dtype=np.uint8)
        reproject(
            source=source.read(1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            resampling=Resampling.nearest,
        )
    return destination


def compose_quality_masks(
    image_path: str | Path,
    output_path: str | Path,
    *,
    manual_mask_path: str | Path | None = None,
    provider_mask_path: str | Path | None = None,
    rgb_mask_path: str | Path | None = None,
    registration_issue: bool = False,
) -> Path:
    """Compose usable evidence with manual > provider > RGB > default priority.

    Input masks use band 1 where non-zero means usable. Manual/provider masks
    may use any grid or CRS and are reprojected by nearest-neighbour.
    """
    output = Path(output_path)
    with rasterio.open(image_path) as reference:
        usable = np.ones((reference.height, reference.width), dtype=bool)
        reason = np.zeros((reference.height, reference.width), dtype=np.uint8)
        if rgb_mask_path and Path(rgb_mask_path).exists():
            rgb = _mask_on_reference(rgb_mask_path, reference)
            rgb_invalid = rgb == 0
            usable[rgb_invalid] = False
            with rasterio.open(rgb_mask_path) as rgb_source:
                if rgb_source.count >= 2 and rgb_source.crs == reference.crs and rgb_source.transform == reference.transform:
                    reason[rgb_invalid] = rgb_source.read(2)[rgb_invalid]
                else:
                    reason[rgb_invalid] = 10
        if provider_mask_path and Path(provider_mask_path).exists():
            provider_invalid = _mask_on_reference(provider_mask_path, reference) == 0
            usable[provider_invalid] = False
            reason[provider_invalid] = 6
        if manual_mask_path and Path(manual_mask_path).exists():
            manual_invalid = _mask_on_reference(manual_mask_path, reference) == 0
            usable[manual_invalid] = False
            reason[manual_invalid] = 7
        if registration_issue:
            usable[:] = False
            reason[:] = 8
        profile = reference.profile.copy()
        profile.update(count=2, dtype="uint8", compress="deflate", nodata=None)
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as destination:
            destination.write(usable.astype(np.uint8), 1)
            destination.write(reason, 2)
            destination.set_band_description(1, "usable")
            destination.set_band_description(2, "quality_reason")
            destination.update_tags(
                priority="manual>provider>rgb_heuristic>default",
                reason_codes=";".join(f"{key}:{value}" for key, value in QUALITY_REASON_CODES.items()),
            )
    return output
