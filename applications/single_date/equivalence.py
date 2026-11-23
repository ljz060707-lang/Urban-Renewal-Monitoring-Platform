"""Equivalence evidence: frozen one-tile-at-a-time protocol vs the batched scheduler.

Same checkpoint, same preprocessing, same 256/192 tile lattice, same triangular
fusion weights, same 0.5 threshold, same 8-connectivity.  Only the number of samples
per forward differs.  Two real crops of the registered 2018-04-29 city mosaic.
"""
import json
import sys
import time

sys.path.insert(0, r'F:\20261028西安市区结果\XIAN_R5_SINGLE_DATE_20261028_01\CODE_SNAPSHOT')
sys.path.insert(0, r'F:\Data Set\RSBuilding\repo')

import numpy as np
import rasterio
import shapely
import torch
from scipy.ndimage import label
from rasterio.features import shapes

from xian_single_date import b0, common

MOSAIC = common.EPOCHS[0]['mosaic']
N = 1024
POS = lambda n: sorted(set(list(range(0, max(n - 256, 0) + 1, 192)) + [max(n - 256, 0)]))
T = [(y, x) for y in POS(N) for x in POS(N)]
v = np.minimum(np.minimum(np.arange(256) + 1, 256 - np.arange(256)) / 32, 1).astype('float32')
W = v[:, None] * v[None, :]
model, digest = b0.model()
rt = b0.runtime()


def run(rgb, B):
    h, w = rgb.shape[:2]
    s = np.zeros((h, w), np.float32)
    nrm = np.zeros((h, w), np.float32)
    t = time.time()
    with torch.inference_mode():
        for i in range(0, len(T), B):
            ch = T[i:i + B]
            inp = torch.from_numpy(np.stack([rgb[y:y + 256, x:x + 256].transpose(2, 0, 1)
                                             for y, x in ch]).copy())
            d = rt.prepare_batch(model, inp)
            with torch.autocast('cuda', dtype=torch.float16):
                out = model._forward(d['inputs'], d['data_samples'])
            p = out[:, 0].sigmoid().float().cpu().numpy()
            for j, (y, x) in enumerate(ch):
                s[y:y + 256, x:x + 256] += p[j] * W
                nrm[y:y + 256, x:x + 256] += W
    return s / nrm, time.time() - t


def instances(p):
    ids, n = label(p > 0.5, structure=np.ones((3, 3), int))
    geoms = [shapely.geometry.shape(gj) for gj, _ in shapes(ids.astype('int32'), mask=ids > 0, connectivity=8)]
    return n, geoms


crops = {'urban_center': (50000, 70000), 'north_edge': (2000, 20000)}
out = {'schema': 'XIAN_R5_BATCHED_SCHEDULER_EQUIVALENCE_V1',
       'checkpoint_sha256': digest,
       'protocol': dict(tile=256, stride=192, threshold=0.5, connectivity=8,
                        baseline='one sample per forward (frozen roof_runtime.probability)',
                        candidate='batch of %d samples per forward' % common.BATCH),
       'crops': {}}
for name, (row, col) in crops.items():
    with rasterio.open(MOSAIC) as ds:
        rgb = np.ascontiguousarray(ds.read([1, 2, 3],
                                           window=rasterio.windows.Window(col, row, N, N)).transpose(1, 2, 0))
    p1, t1 = run(rgb, 1)
    p8, t8 = run(rgb, common.BATCH)
    d = np.abs(p1 - p8)
    flips = int(((p1 > .5) != (p8 > .5)).sum())
    n1, g1 = instances(p1)
    n8, g8 = instances(p8)
    a1 = sum(x.area for x in g1)
    a8 = sum(x.area for x in g8)
    out['crops'][name] = dict(
        window=dict(row=row, col=col, width=N, height=N),
        tiles=len(T),
        seconds_baseline=round(t1, 2), seconds_batched=round(t8, 2),
        speedup=round(t1 / t8, 2),
        max_abs_probability_difference=round(float(d.max()), 6),
        mean_abs_probability_difference=round(float(d.mean()), 8),
        p99_abs_probability_difference=round(float(np.percentile(d, 99)), 8),
        threshold_crossing_pixels=flips,
        threshold_crossing_fraction=round(flips / (N * N), 8),
        instances_baseline=n1, instances_batched=n8,
        instance_count_equal=n1 == n8,
        foreground_area_px_baseline=round(a1, 3), foreground_area_px_batched=round(a8, 3),
        foreground_area_relative_difference=round(abs(a1 - a8) / max(a1, 1e-9), 10),
        pixel_iou=round(float(((p1 > .5) & (p8 > .5)).sum() / max(1, ((p1 > .5) | (p8 > .5)).sum())), 8),
    )
    print(name, out['crops'][name])

out['verdict'] = ('ACCEPTED_FOR_SCHEDULING_ONLY' if all(
    c['instance_count_equal'] and c['pixel_iou'] > 0.999 and c['threshold_crossing_fraction'] < 0.001
    for c in out['crops'].values()) else 'REJECTED_SCHEDULING_CHANGE')
out['not_changed'] = ['checkpoint', 'preprocessing', 'tile lattice', 'fusion weights',
                      'threshold', 'connectivity', 'object extraction', 'crs conversion']
json.dump(out, open(r'F:\20261028西安市区结果\XIAN_R5_SINGLE_DATE_20261028_01\evidence\BATCHED_SCHEDULER_EQUIVALENCE.json',
                    'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('verdict', out['verdict'])
