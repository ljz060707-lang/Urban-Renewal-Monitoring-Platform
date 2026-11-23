"""Rasterisation and exact fractions on the frozen reference grid.

The exact-fraction definition is the one used by xian_r5.label_transfer.aggregate:
    Area(Union(geometry) intersect pixel intersect support) / full pixel area
It is evaluated additively per (disjoint) instance polygon instead of per cell, which
is the same number but linear in covered cells rather than in grid cells.
"""
from __future__ import annotations

import math

import numpy as np
import rasterio
import shapely
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import box


def template_grid(template, subdivision=1):
    with rasterio.open(template) as ds:
        affine = ds.transform
        width, height = ds.width, ds.height
        crs = str(ds.crs)
    if subdivision != 1:
        affine = affine * Affine.scale(1.0 / subdivision)
        width *= subdivision
        height *= subdivision
    return dict(affine=affine, width=int(width), height=int(height), crs=crs)


def grid_bounds(grid):
    a = grid['affine']
    return box(a.c, a.f + a.e * grid['height'],
               a.c + a.a * grid['width'], a.f)


def exact_fraction(geoms, grid, support=None, ids=None):
    """Return (fraction float32 HxW, instance id uint32 HxW, assigned area, cells)."""
    affine = grid['affine']
    width, height = grid['width'], grid['height']
    cell_area = abs(affine.a * affine.e)
    frac = np.zeros((height, width), np.float32)
    ident = np.zeros((height, width), np.uint32)
    total = 0.0
    touched = 0
    for k, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue
        if support is not None and not support.covers(g):
            g = g.intersection(support)
            if g.is_empty:
                continue
        inv = ~affine
        minx, miny, maxx, maxy = g.bounds
        px0, py0 = inv * (minx, maxy)
        px1, py1 = inv * (maxx, miny)
        c0 = max(0, int(math.floor(px0)))
        r0 = max(0, int(math.floor(py0)))
        c1 = min(width, int(math.ceil(px1)))
        r1 = min(height, int(math.ceil(py1)))
        if c1 <= c0 or r1 <= r0:
            continue
        cols = np.arange(c0, c1)
        rows = np.arange(r0, r1)
        xs = affine.c + cols * affine.a
        ys = affine.f + rows * affine.e
        X0, Y1 = np.meshgrid(xs, ys)
        cells = shapely.box(X0, Y1 + affine.e, X0 + affine.a, Y1)
        inter = shapely.intersection(g, cells)
        areas = shapely.area(inter)
        hit = areas > 0
        if not hit.any():
            continue
        band = frac[r0:r1, c0:c1]
        band[hit] += (areas[hit] / cell_area).astype(np.float32)
        idb = ident[r0:r1, c0:c1]
        idb[hit] = (k + 1) if ids is None else np.uint32(int(ids[k]))
        total += float(areas[hit].sum())
        touched += int(hit.sum())
    return frac, ident, total, touched


def banded_rasterize(path, grid, geoms, values, dtype='uint32', nodata=0, band_rows=2048,
                     compress='deflate'):
    """Write a raster in row bands so a 1.2 Gpx city grid never needs one array."""
    affine = grid['affine']
    width, height = grid['width'], grid['height']
    from shapely import STRtree
    valid = [(g, int(v)) for g, v in zip(geoms, values) if g is not None and not g.is_empty]
    tree = STRtree([g for g, _ in valid]) if valid else None
    with rasterio.open(path, 'w', driver='GTiff', height=height, width=width, count=1,
                       dtype=dtype, crs=grid['crs'], transform=affine, nodata=nodata,
                       compress=compress, tiled=True, blockxsize=512, blockysize=512,
                       BIGTIFF='YES') as ds:
        for r0 in range(0, height, band_rows):
            r1 = min(height, r0 + band_rows)
            a = affine * Affine.translation(0, r0)
            band_box = box(a.c, a.f + a.e * (r1 - r0), a.c + a.a * width, a.f)
            arr = np.full((r1 - r0, width), nodata, dtype)
            if valid:
                idx = tree.query(band_box)
                gg = [(valid[i][0], valid[i][1]) for i in np.atleast_1d(idx)
                      if valid[i][0].intersects(band_box)]
                if gg:
                    arr = rasterize(gg, out_shape=(r1 - r0, width), transform=a,
                                    dtype=dtype, fill=nodata)
            ds.write(arr, 1, window=rasterio.windows.Window(0, r0, width, r1 - r0))
            del arr


def write_band(path, r0, r1, array, count=1):
    with rasterio.open(path) as ds:
        ds.write(array if array.ndim == 3 else array[None], window=rasterio.windows.Window(0, r0, ds.width, r1 - r0))


def read_band(path, r0, r1):
    with rasterio.open(path) as ds:
        return ds.read(1, window=rasterio.windows.Window(0, r0, ds.width, r1 - r0))
