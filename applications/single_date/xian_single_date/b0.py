"""B0 retained roof baseline, applied with the frozen V1 protocol.

Protocol is taken unchanged from MODEL_ROUTING_MANIFEST.json roof.*:
  architecture  Original RSBuilding FoundationEncoderDecoder Swin-T
  normalization BGR duplication -> original ImageNet RGB preprocessor
  activation    sigmoid on the original A head
  tiling        256 px window, stride 192, separable triangular weight fusion
  decision      probability > 0.5, 8-connectivity components
The only scheduling difference from roof_runtime.probability() is that tiles are
submitted in batches of BATCH.  evidence/BATCHED_SCHEDULER_EQUIVALENCE.json records
the measured per-pixel and per-object effect of that change.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time

import numpy as np
import rasterio
import shapely
import torch
from rasterio.features import shapes
from scipy import ndimage

from . import common

_RT = None


def runtime():
    global _RT
    if _RT is None:
        spec = importlib.util.spec_from_file_location(
            'xian_b0_runtime',
            r'F:\xian\XIAN_R5_S2_SUPERVISED_ROOF_TRAINING_20261019\roof_runtime.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules['xian_b0_runtime'] = mod
        spec.loader.exec_module(mod)
        _RT = mod
    return _RT


def model():
    rt = runtime()
    m, digest = rt.cached_model(str(rt.B0))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return m, digest


def _positions(n):
    return sorted(set(list(range(0, max(n - 256, 0) + 1, common.STRIDE)) + [max(n - 256, 0)]))


def _fusion_weight():
    v = np.minimum(np.minimum(np.arange(256) + 1, 256 - np.arange(256)) / 32, 1).astype('float32')
    return v[:, None] * v[None, :]


def probability(rgb, progress=None):
    """Frozen 256/192 weighted fusion over one block, batched scheduling."""
    rt = runtime()
    m, _ = model()
    h, w = rgb.shape[:2]
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    weight = _fusion_weight()
    acc = np.zeros((h, w), np.float32)
    norm = np.zeros((h, w), np.float32)
    coords = [(y, x) for y in _positions(h) for x in _positions(w)]
    started = time.time()
    with torch.inference_mode():
        for i in range(0, len(coords), common.BATCH):
            chunk = coords[i:i + common.BATCH]
            inp = torch.from_numpy(np.stack(
                [rgb[y:y + 256, x:x + 256].transpose(2, 0, 1) for y, x in chunk]).copy())
            data = rt.prepare_batch(m, inp)
            with torch.autocast('cuda', dtype=torch.float16):
                out = m._forward(data['inputs'], data['data_samples'])
            p = out[:, 0].sigmoid().float().cpu().numpy()
            for j, (y, x) in enumerate(chunk):
                acc[y:y + 256, x:x + 256] += p[j] * weight
                norm[y:y + 256, x:x + 256] += weight
            if progress is not None and i % (common.BATCH * 200) == 0:
                progress(i + len(chunk), len(coords), time.time() - started)
    if not (norm > 0).all():
        raise RuntimeError('Fusion weight normalisation has holes')
    return acc / norm


def block_plan(mosaic):
    """Deterministic core block lattice over the whole registered mosaic."""
    with rasterio.open(mosaic) as ds:
        width, height = ds.width, ds.height
    blocks = []
    cols = int(math.ceil(width / common.CORE))
    rows = int(math.ceil(height / common.CORE))
    for j in range(rows):
        for i in range(cols):
            col0 = i * common.CORE
            row0 = j * common.CORE
            core_w = min(common.CORE, width - col0)
            core_h = min(common.CORE, height - row0)
            rc0 = max(0, col0 - common.GUARD)
            rr0 = max(0, row0 - common.GUARD)
            rc1 = min(width, col0 + core_w + common.GUARD)
            rr1 = min(height, row0 + core_h + common.GUARD)
            blocks.append(dict(
                block_id='b%02d_%02d' % (j, i), index=len(blocks),
                col0=col0, row0=row0, core_w=core_w, core_h=core_h,
                rc0=rc0, rr0=rr0, rc1=rc1, rr1=rr1,
                width=rc1 - rc0, height=rr1 - rr0))
    return blocks


def read_rgb(mosaic, block):
    with rasterio.open(mosaic) as ds:
        arr = ds.read([1, 2, 3],
                      window=rasterio.windows.Window(block['rc0'], block['rr0'],
                                                     block['width'], block['height']))
    return np.ascontiguousarray(arr.transpose(1, 2, 0))


def components(prob, block, affine, to_metric):
    """Threshold, 8-connectivity components, per-object metrics in one vector pass."""
    if affine is None:
        raise ValueError('mosaic affine is required')
    if abs(affine.b) > 1e-12 or abs(affine.d) > 1e-12:
        raise ValueError('rotated source grid needs an explicit rasterio warp')
    mask = prob > common.B0_THRESHOLD
    ids, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return []
    counts = np.bincount(ids.ravel(), minlength=n + 1)
    means = ndimage.mean(prob, ids, index=np.arange(1, n + 1))
    geoms = []
    idxs = []
    from shapely.geometry import shape as shp
    for gj, idx in shapes(ids.astype('int32'), mask=ids > 0, connectivity=8):
        geoms.append(shp(gj))
        idxs.append(int(idx))
    if not geoms:
        return []
    g_all = np.array(geoms, dtype=object)
    cent = shapely.centroid(g_all)
    cx = shapely.get_x(cent)
    cy = shapely.get_y(cent)
    cx0 = block['col0'] - block['rc0']
    cy0 = block['row0'] - block['rr0']
    owned = ((cx >= cx0) & (cx < cx0 + block['core_w']) &
             (cy >= cy0) & (cy < cy0 + block['core_h']))
    if not owned.any():
        return []
    g_all = g_all[owned]
    idx_arr = np.asarray(idxs, dtype='int64')[owned]
    b = shapely.bounds(g_all)
    margin = common.EDGE_MARGIN
    edge = ((b[:, 0] < margin) | (b[:, 1] < margin) |
            (b[:, 2] > block['width'] - margin) | (b[:, 3] > block['height'] - margin))
    # pixel edge (window) -> mosaic pixel -> source CRS degrees, one vectorised call
    scale = np.array([affine.a, affine.e])
    offset = np.array([affine.c, affine.f])
    dx = np.array([float(block['rc0']), float(block['rr0'])])
    g_deg = shapely.transform(g_all, lambda xy: (xy + dx) * scale + offset)
    g_metric = shapely.transform(g_deg, to_metric)
    valid = shapely.is_valid(g_metric)
    if not valid.all():
        g_metric = np.where(valid, g_metric, shapely.make_valid(g_metric))
    rows = []
    for k in range(len(g_metric)):
        rows.append(dict(
            component_index=int(idx_arr[k]),
            pixel_area=int(counts[idx_arr[k]]),
            mean_probability=float(means[idx_arr[k] - 1]),
            native_bounds=[float(v) for v in b[k]],
            geometry=g_metric[k],
            source_edge_partial_risk=bool(edge[k]),
            repaired=bool(not valid[k]),
        ))
    rows.sort(key=lambda r: (r['native_bounds'][1], r['native_bounds'][0], r['component_index']))
    return rows
