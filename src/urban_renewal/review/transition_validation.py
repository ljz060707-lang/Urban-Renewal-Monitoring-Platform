from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PreviewQuality:
    preview_status: str
    review_allowed: bool
    valid_pixel_fraction: float
    display_dynamic_range: float
    mean_luminance: float
    source_read_success: bool
    window_inside_bounds: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def validate_preview(
    rgb: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    source_read_success: bool = True,
    window_inside_bounds: bool = True,
    minimum_dynamic_range: float = 0.08,
    black_threshold: float = 0.03,
    white_threshold: float = 0.97,
) -> PreviewQuality:
    """Validate a display-ready RGB preview without treating invalid display as model failure."""
    array = np.asarray(rgb)
    reasons: list[str] = []
    if array.ndim != 3 or array.shape[-1] != 3 or array.size == 0:
        return PreviewQuality(
            "invalid", False, 0.0, 0.0, 0.0, source_read_success,
            window_inside_bounds, ("invalid_rgb_shape",),
        )
    values = array.astype(np.float64, copy=False)
    if values.max(initial=0.0) > 1.0:
        values /= 255.0
    finite = np.all(np.isfinite(values), axis=-1)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)
    valid_fraction = float(finite.mean()) if finite.size else 0.0
    if valid_fraction > 0:
        valid_values = values[finite]
        luminance = (
            0.2126 * valid_values[:, 0]
            + 0.7152 * valid_values[:, 1]
            + 0.0722 * valid_values[:, 2]
        )
        dynamic_range = float(np.percentile(luminance, 98) - np.percentile(luminance, 2))
        mean_luminance = float(luminance.mean())
    else:
        dynamic_range = 0.0
        mean_luminance = 0.0
    if not source_read_success:
        reasons.append("source_read_failed")
    if not window_inside_bounds:
        reasons.append("window_outside_bounds")
    if valid_fraction <= 0:
        reasons.append("no_valid_pixels")
    if dynamic_range <= minimum_dynamic_range:
        reasons.append("insufficient_display_dynamic_range")
    if mean_luminance <= black_threshold:
        reasons.append("preview_too_dark")
    if mean_luminance >= white_threshold:
        reasons.append("preview_too_bright")
    status = "valid" if not reasons else "invalid"
    return PreviewQuality(
        status,
        status == "valid",
        valid_fraction,
        dynamic_range,
        mean_luminance,
        source_read_success,
        window_inside_bounds,
        tuple(reasons),
    )
