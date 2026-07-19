from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds

from .crs import reproject_raster_to_reference, validate_source_crs
from .manifest import DateRecord


def _gray_preview(path: Path, size: int = 512) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1, out_shape=(min(size, src.height), min(size, src.width))).astype(np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros_like(data)
    low, high = np.nanpercentile(data[finite], [2, 98])
    return np.clip((data - low) / max(high - low, 1e-6), 0, 1).astype(np.float32)


def _overlap_fraction(a: DateRecord, b: DateRecord) -> float:
    with rasterio.open(a.image_path) as src_a, rasterio.open(b.image_path) as src_b:
        crs_a = validate_source_crs(src_a.crs)
        crs_b = validate_source_crs(src_b.crs)
        b_bounds = transform_bounds(crs_b, crs_a, *src_b.bounds, densify_pts=21)
        left = max(src_a.bounds.left, b_bounds[0])
        bottom = max(src_a.bounds.bottom, b_bounds[1])
        right = min(src_a.bounds.right, b_bounds[2])
        top = min(src_a.bounds.top, b_bounds[3])
        intersection = max(0.0, right - left) * max(0.0, top - bottom)
        union = (src_a.bounds.right - src_a.bounds.left) * (src_a.bounds.top - src_a.bounds.bottom)
        return float(intersection / union) if union else 0.0


def diagnose_alignment(
    records: list[DateRecord],
    *,
    base_buffer_m: float = 3.0,
    off_nadir_scale_m_per_degree: float = 0.35,
    max_dynamic_buffer_m: float = 20.0,
) -> pd.DataFrame:
    """Build adjacent-date ground alignment and roof-displacement diagnostics."""
    if len(records) < 2:
        raise ValueError("Alignment diagnostics require at least two dates")
    rows: list[dict] = []
    for first, second in zip(records[:-1], records[1:]):
        with rasterio.open(first.image_path) as src_a, rasterio.open(second.image_path) as src_b:
            validate_source_crs(src_a.crs)
            validate_source_crs(src_b.crs)
            pixel_m = max(abs(src_a.transform.a), abs(src_a.transform.e)) if src_a.crs.is_projected else float("nan")
            compatible = src_a.crs == src_b.crs and src_a.width == src_b.width and src_a.height == src_b.height
        shift_x = shift_y = float("nan")
        notes: list[str] = []
        if compatible:
            a = _gray_preview(first.image_path)
            b = _gray_preview(second.image_path)
            if a.shape == b.shape and float(a.std()) > 1e-6 and float(b.std()) > 1e-6:
                (shift_x, shift_y), _ = cv2.phaseCorrelate(a, b)
            else:
                shift_x = shift_y = 0.0
                notes.append("insufficient stable texture for local shift; reported as 0 in synthetic/flat scene")
        else:
            notes.append("different grids; reproject to the reference grid before local shift estimation")
        off_a = float(first.off_nadir_deg or 0.0)
        off_b = float(second.off_nadir_deg or 0.0)
        recommended = min(max_dynamic_buffer_m, base_buffer_m + off_nadir_scale_m_per_degree * abs(off_a - off_b))
        shift_px = float(np.hypot(shift_x, shift_y))
        overlap = _overlap_fraction(first, second)
        warning = "error" if overlap < 0.7 else "warning" if (not compatible or shift_px > 1.0) else "ok"
        rows.append(
            {
                "date_from": first.date,
                "date_to": second.date,
                "estimated_ground_shift_px": shift_px,
                "estimated_ground_shift_m": shift_px * pixel_m,
                "off_nadir_difference_deg": abs(off_a - off_b),
                "recommended_tracking_buffer_m": recommended,
                "overlap_fraction": overlap,
                "warning_level": warning,
                "notes": "; ".join(notes) or "ground alignment is separate from high-rise roof displacement",
            }
        )
    return pd.DataFrame(rows)


def write_alignment_report(frame: pd.DataFrame, csv_path: str | Path, markdown_path: str | Path) -> None:
    csv_output = Path(csv_path)
    md_output = Path(markdown_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_output, index=False, encoding="utf-8-sig")
    columns = list(frame.columns)
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        table.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    md_output.write_text(
        "# Alignment diagnostics\n\n"
        "Ground alignment and high-rise roof displacement are evaluated separately.\n\n"
        + "\n".join(table)
        + "\n",
        encoding="utf-8",
    )
