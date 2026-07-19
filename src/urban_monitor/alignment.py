from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import rasterio

from .manifest import DateRecord


@dataclass
class RasterSignature:
    date: str
    path: str
    crs: str
    width: int
    height: int
    transform: tuple[float, ...]
    pixel_size_x: float
    pixel_size_y: float
    bounds: tuple[float, float, float, float]
    dtype: str
    count: int
    projected: bool
    matches_reference: bool = True
    mismatch_reason: str = ""


def inspect_raster(record: DateRecord) -> RasterSignature:
    with rasterio.open(record.image_path) as src:
        return RasterSignature(
            date=record.date,
            path=str(record.image_path),
            crs=str(src.crs),
            width=src.width,
            height=src.height,
            transform=tuple(src.transform)[:6],
            pixel_size_x=abs(src.transform.a),
            pixel_size_y=abs(src.transform.e),
            bounds=tuple(src.bounds),
            dtype=str(src.dtypes[0]),
            count=src.count,
            projected=bool(src.crs and src.crs.is_projected),
        )


def validate_alignment(records: list[DateRecord], *, require_projected: bool = True) -> pd.DataFrame:
    if not records:
        raise ValueError("Manifest contains no imagery records")
    signatures = [inspect_raster(r) for r in records]
    ref = signatures[0]
    for sig in signatures:
        reasons: list[str] = []
        if require_projected and not sig.projected:
            reasons.append("CRS is not projected")
        for field in ("crs", "width", "height", "transform", "bounds"):
            if getattr(sig, field) != getattr(ref, field):
                reasons.append(f"{field} differs from reference")
        sig.matches_reference = not reasons
        sig.mismatch_reason = "; ".join(reasons)
    return pd.DataFrame([asdict(s) for s in signatures])
