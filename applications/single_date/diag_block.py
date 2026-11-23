"""Diagnostic: why does a 10% >0.5 block yield only 20 owned objects?"""
import sys

sys.path.insert(0, r'F:\20261028西安市区结果\XIAN_R5_SINGLE_DATE_20261028_01\CODE_SNAPSHOT')

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import shapes
from scipy import ndimage

from xian_single_date import b0, common

common.CORE = 1536
common.GUARD = 384
common.BATCH = 8

ep = common.EPOCHS[0]
plan = b0.block_plan(ep['mosaic'])
block = plan[len(plan) // 2 + 20]
rgb = b0.read_rgb(ep['mosaic'], block)
prob = b0.probability(rgb)
mask = prob > 0.5
ids, n = ndimage.label(mask, structure=np.ones((3, 3), int))
counts = np.bincount(ids.ravel())
print('block', block)
print('label count', n, 'foreground px', int(mask.sum()), 'largest', int(counts[1:].max()))
print('size histogram of components:',
      np.percentile(counts[1:], [50, 90, 99]) if n else None)
cx0, cy0 = block['col0'] - block['rc0'], block['row0'] - block['rr0']
kept = 0
kept_area = 0.0
for gj, idx in shapes(ids.astype('int32'), mask=ids > 0, connectivity=8):
    from shapely.geometry import shape as shp
    g = shp(gj)
    cx, cy = g.centroid.coords[0]
    if cx0 <= cx < cx0 + block['core_w'] and cy0 <= cy < cy0 + block['core_h']:
        kept += 1
        kept_area += g.area
print('kept', kept, 'kept pixel area', kept_area, 'px m2 factor',
      kept_area / max(1e-9, sum(shape_area for shape_area in [kept_area])))

Image.fromarray(rgb).save(r'F:\20261028西安市区结果\_diag_rgb.png')
Image.fromarray((np.clip(prob, 0, 1) * 255).astype('uint8')).save(r'F:\20261028西安市区结果\_diag_prob.png')
Image.fromarray((mask.astype('uint8') * 255)).save(r'F:\20261028西安市区结果\_diag_mask.png')
print('wrote diag pngs')
