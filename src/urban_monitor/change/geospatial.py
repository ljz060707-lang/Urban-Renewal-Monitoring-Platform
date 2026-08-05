from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RasterGrid:
    crs: str
    transform: tuple[float, ...]
    width: int
    height: int
    resolution: tuple[float, float]
    nodata: float | None


def grid_from_dataset(dataset) -> RasterGrid:
    return RasterGrid(
        crs=str(dataset.crs) if dataset.crs else "",
        transform=tuple(dataset.transform)[:6],
        width=dataset.width,
        height=dataset.height,
        resolution=(abs(dataset.transform.a), abs(dataset.transform.e)),
        nodata=dataset.nodata,
    )


def same_grid(first, second, *, tolerance: float = 1e-9) -> bool:
    if first.crs != second.crs or first.width != second.width or first.height != second.height:
        return False
    return bool(np.allclose(tuple(first.transform)[:6], tuple(second.transform)[:6], atol=tolerance, rtol=0))


@contextmanager
def aligned_pair(t1_path: str | Path, t2_path: str | Path):
    """Open T1 and a T2 view explicitly aligned to the T1 reference grid."""
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling

    raster_suffixes = {".tif", ".tiff"}
    if Path(t1_path).suffix.casefold() not in raster_suffixes or Path(t2_path).suffix.casefold() not in raster_suffixes:
        # PNG/JPG inputs have no georeference.  They remain pixel-space only;
        # T2 is deterministically resized to T1, never falsely assigned a CRS.
        from PIL import Image
        from rasterio.io import MemoryFile
        from rasterio.transform import Affine
        def read(path):
            value = np.asarray(Image.open(path).convert("RGB"))
            return np.moveaxis(value, -1, 0)
        first, second = read(t1_path), read(t2_path)
        if second.shape[1:] != first.shape[1:]:
            resized = Image.fromarray(np.moveaxis(second, 0, -1)).resize((first.shape[2], first.shape[1]), Image.Resampling.BILINEAR)
            second = np.moveaxis(np.asarray(resized), -1, 0)
        profile = {"driver": "GTiff", "height": first.shape[1], "width": first.shape[2], "count": 3, "dtype": "uint8", "transform": Affine.identity()}
        with MemoryFile() as first_mem, MemoryFile() as second_mem:
            with first_mem.open(**profile) as t1, second_mem.open(**profile) as t2:
                t1.write(first); t2.write(second)
                yield t1, t2, bool(Path(t1_path).suffix.casefold() != Path(t2_path).suffix.casefold() or first.shape != second.shape)
        return
    with rasterio.open(t1_path) as t1, rasterio.open(t2_path) as raw_t2:
        if not t1.crs or not raw_t2.crs:
            raise ValueError("Both GeoTIFF inputs must have a CRS")
        if same_grid(t1, raw_t2):
            yield t1, raw_t2, False
        else:
            with WarpedVRT(
                raw_t2,
                crs=t1.crs,
                transform=t1.transform,
                width=t1.width,
                height=t1.height,
                resampling=Resampling.bilinear,
            ) as aligned:
                yield t1, aligned, True


def tile_windows(width: int, height: int, tile_size: int, overlap: int):
    from rasterio.windows import Window

    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Require tile_size > overlap >= 0")
    stride = tile_size - overlap
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if not xs or xs[-1] != max(0, width - tile_size):
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] != max(0, height - tile_size):
        ys.append(max(0, height - tile_size))
    seen = set()
    for y in ys:
        for x in xs:
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            yield Window(x, y, min(tile_size, width - x), min(tile_size, height - y))


def blending_weight(height: int, width: int, floor: float = 0.05) -> np.ndarray:
    y = np.hanning(max(3, height))[:height]
    x = np.hanning(max(3, width))[:width]
    weight = np.outer(y, x)
    maximum = float(weight.max())
    if maximum > 0:
        weight /= maximum
    return np.maximum(weight, floor).astype(np.float32)


def write_float_raster(path: str | Path, array: np.ndarray, reference, *, nodata: float = -9999.0):
    profile = reference.profile.copy()

    # A source RGB GeoTIFF may carry JPEG/YCBCR/interleave metadata.
    # Those creation options are invalid for the single-band float/mask
    # rasters produced by the change-detection pipeline.
    for _key in (
        "photometric",
        "interleave",
    ):
        profile.pop(_key, None)

    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=nodata,
        compress="deflate",
        predictor=3,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with __import__("rasterio").open(target, "w", **profile) as sink:
        sink.write(np.asarray(array, dtype=np.float32), 1)
    return target


def write_mask_raster(path: str | Path, array: np.ndarray, reference):
    profile = reference.profile.copy()

    # A source RGB GeoTIFF may carry JPEG/YCBCR/interleave metadata.
    # Those creation options are invalid for the single-band float/mask
    # rasters produced by the change-detection pipeline.
    for _key in (
        "photometric",
        "interleave",
    ):
        profile.pop(_key, None)

    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with __import__("rasterio").open(target, "w", **profile) as sink:
        sink.write(np.asarray(array, dtype=np.uint8), 1)
    return target


def resolution_interpretation(resolution_m: float) -> str:
    if resolution_m <= 1.0:
        return "building-object refinement"
    if resolution_m <= 5.0:
        return "building candidates"
    return "building-group / parcel / block level candidate"
