"""Render explicitly unaccepted image-grid ties as review-only raster previews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from shapely.geometry import mapping

from .field import LocalDisplacementField


PREVIEW_QUALITY_CODES = {
    0: "NO_SOURCE_COVERAGE_AFTER_CANDIDATE_SHIFT",
    1: "UNACCEPTED_ALIGNMENT_CANDIDATE_POSITION_UNCERTAINTY_UNKNOWN",
    2: "UNRESOLVED_NO_NUMERIC_SUBPIXEL_TIE",
    3: "CANDIDATE_CELL_EDGE_PIECEWISE_WARP_SEAM_RISK",
    4: "AMBIGUOUS_OVERLAPPING_FIELD_SUPPORT",
    5: "MOSAIC_COMPONENT_BOUNDARY_UNRESOLVED",
    6: "SOURCE_COMPONENT_COVERAGE_INCOMPLETE",
    7: "SOURCE_COMPONENT_ID_MISMATCH",
    8: "SOURCE_COMPONENT_SCOPE_UNKNOWN",
    255: "OUTSIDE_FIELD_SUPPORT",
}

_SOURCE_COMPONENT_QUALITY_CODES = {
    "MOSAIC_COMPONENT_BOUNDARY_UNRESOLVED": 5,
    "SOURCE_COMPONENT_COVERAGE_INCOMPLETE": 6,
    "SOURCE_COMPONENT_ID_MISMATCH": 7,
    "SOURCE_COMPONENT_SCOPE_UNKNOWN": 8,
}


@dataclass(frozen=True)
class CandidateAlignmentPreview:
    """RGBA preview, categorical quality grid, and per-cell provenance summary."""

    rgba: np.ndarray
    quality: np.ndarray
    cells: tuple[dict[str, Any], ...]


def _edge_pixels(mask: np.ndarray, width: int) -> np.ndarray:
    """Find cell-edge pixels without adding a SciPy dependency."""
    if width < 1:
        return np.zeros(mask.shape, dtype=bool)
    interior = mask.copy()
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros(mask.shape, dtype=bool)
            src_y0, src_y1 = max(0, -dy), min(mask.shape[0], mask.shape[0] - dy)
            src_x0, src_x1 = max(0, -dx), min(mask.shape[1], mask.shape[1] - dx)
            dst_y0, dst_y1 = max(0, dy), min(mask.shape[0], mask.shape[0] + dy)
            dst_x0, dst_x1 = max(0, dx), min(mask.shape[1], mask.shape[1] + dx)
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
            interior &= shifted
    return mask & ~interior


def render_candidate_alignment_preview(
    moving_rgb: np.ndarray,
    source_validity: np.ndarray,
    source_transform: Any,
    reference_shape: tuple[int, int],
    reference_transform: Any,
    field: LocalDisplacementField,
    *,
    seam_width_px: int = 1,
) -> CandidateAlignmentPreview:
    """Warp only numeric, unaccepted local ties into the reference raster grid.

    This is a transparent, candidate-only visualization surface. It does not
    accept the field, interpolate missing cells, claim positional accuracy,
    or perform roof-to-ground/parallax correction. Cells without a numeric
    candidate remain transparent and are identified in the quality grid.
    """
    image = np.asarray(moving_rgb)
    valid = np.asarray(source_validity, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or valid.shape != image.shape[:2]:
        raise ValueError("Source RGB image and validity mask must share an H×W grid")
    if min(reference_shape) < 1 or seam_width_px < 0:
        raise ValueError("Reference shape must be positive and seam width non-negative")
    if field.crs != "EPSG:32649" or field.correction_scope != "IMAGE_TO_REFERENCE_GRID":
        raise ValueError("Candidate preview requires an EPSG:32649 image-grid field")
    if any(cell.accepted for cell in field.cells):
        raise ValueError("Candidate-only preview cannot consume an accepted correction field")

    src_transform = Affine(*tuple(source_transform)[:6])
    dst_transform = Affine(*tuple(reference_transform)[:6])
    dst_height, dst_width = (int(value) for value in reference_shape)
    dst_shape = (dst_height, dst_width)
    source_float = np.asarray(image, dtype=np.float32)
    source_float = np.where(valid[:, :, None], source_float, np.nan)

    masks: list[np.ndarray] = []
    support_count = np.zeros(dst_shape, dtype=np.uint16)
    for cell in field.cells:
        mask = rasterize(
            [(mapping(cell.support_geometry), 1)],
            out_shape=dst_shape,
            transform=dst_transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        if not mask.any():
            raise ValueError(f"Field cell has no pixels on the reference grid: {cell.cell_id}")
        masks.append(mask)
        support_count += mask.astype(np.uint16)

    ambiguous = support_count > 1
    quality = np.full(dst_shape, 255, dtype=np.uint8)
    covered = support_count == 1
    quality[covered] = 2
    quality[ambiguous] = 4
    rgba = np.zeros((4, dst_height, dst_width), dtype=np.uint8)
    cell_receipts: list[dict[str, Any]] = []

    for cell, mask in zip(field.cells, masks):
        usable = (
            cell.correction_dx_m_candidate is not None
            and cell.correction_dy_m_candidate is not None
            and not cell.accepted
            and not cell.applied
        )
        cell_pixels = mask & ~ambiguous
        receipt: dict[str, Any] = {
            "cell_id": cell.cell_id,
            "status": cell.status,
            "preview_candidate_rendered": False,
            "correction_dx_m_candidate": cell.correction_dx_m_candidate,
            "correction_dy_m_candidate": cell.correction_dy_m_candidate,
            "best_correlation": cell.metadata.get("best_correlation"),
            "peak_margin": cell.metadata.get("peak_margin"),
            "positional_uncertainty_m": cell.positional_uncertainty_m,
            "position_accuracy_claimed": False,
            "source_component_scope": cell.metadata.get("source_component_scope"),
            "source_component_ids": cell.metadata.get("source_component_ids", []),
            "source_component_coverage_fraction": cell.metadata.get(
                "source_component_coverage_fraction"),
            "source_component_issue": cell.source_component_issue,
        }
        if not usable:
            cell_receipts.append(receipt)
            continue

        if cell.source_component_issue is not None:
            quality_code = _SOURCE_COMPONENT_QUALITY_CODES[cell.source_component_issue]
            quality[cell_pixels] = quality_code
            receipt["preview_blocked_reason"] = cell.source_component_issue
            receipt["preview_quality_code"] = quality_code
            receipt["preview_pixel_count"] = 0
            receipt["seam_risk_pixel_count"] = 0
            cell_receipts.append(receipt)
            continue

        shifted_transform = (
            Affine.translation(
                float(cell.correction_dx_m_candidate),
                float(cell.correction_dy_m_candidate),
            )
            * src_transform
        )
        warped = np.full((3, dst_height, dst_width), np.nan, dtype=np.float32)
        for band in range(3):
            reproject(
                source=source_float[:, :, band],
                destination=warped[band],
                src_transform=shifted_transform,
                src_crs=field.crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=field.crs,
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
        warped_valid = np.zeros(dst_shape, dtype=np.uint8)
        reproject(
            source=valid.astype(np.uint8),
            destination=warped_valid,
            src_transform=shifted_transform,
            src_crs=field.crs,
            src_nodata=0,
            dst_transform=dst_transform,
            dst_crs=field.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
        finite = np.isfinite(warped).all(axis=0)
        has_source = cell_pixels & (warped_valid > 0) & finite
        quality[cell_pixels & ~has_source] = 0
        if has_source.any():
            rgb = np.clip(np.rint(warped[:, has_source]), 0, 255).astype(np.uint8)
            rgba[:3, has_source] = rgb
            rgba[3, has_source] = 255
            quality[has_source] = 1
            edge = _edge_pixels(mask, seam_width_px) & has_source
            quality[edge] = 3
        receipt["preview_candidate_rendered"] = bool(has_source.any())
        receipt["preview_pixel_count"] = int(has_source.sum())
        receipt["seam_risk_pixel_count"] = int(
            ((_edge_pixels(mask, seam_width_px)) & has_source).sum()
        )
        cell_receipts.append(receipt)

    return CandidateAlignmentPreview(
        rgba=rgba,
        quality=quality,
        cells=tuple(cell_receipts),
    )
