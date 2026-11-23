"""Smoke test: one central and one edge block of the 2018-04-29 epoch."""
import sys
import time

sys.path.insert(0, r'F:\20261028西安市区结果\XIAN_R5_SINGLE_DATE_20261028_01\CODE_SNAPSHOT')

import rasterio
from pyproj import Transformer
from shapely.ops import transform as shp_transform

from xian_single_date import b0, common, epoch

common.CORE = 1536
common.GUARD = 384
common.BATCH = 8

ep = common.EPOCHS[0]
plan = b0.block_plan(ep['mosaic'])
print('blocks %d' % len(plan))
with rasterio.open(ep['mosaic']) as ds:
    aff = ds.transform
    tr = Transformer.from_crs(ds.crs, common.METRIC_CRS, always_xy=True)

dates = epoch.tile_dates(ep)
targets = [plan[0], plan[len(plan) // 2 + 20], plan[-1]]
for block in targets:
    t = time.time()
    rgb = b0.read_rgb(ep['mosaic'], block)
    prob = b0.probability(rgb)
    t_model = time.time() - t
    t = time.time()
    rows = b0.components(prob, block, aff, lambda g: shp_transform(tr.transform, g))
    t_vec = time.time() - t
    areas = sorted(r['geometry'].area for r in rows) if rows else [0]
    print('%s mean=%.3f frac>.5=%.3f n=%d total_m2=%.0f median=%.1f max=%.0f model=%.1fs vec=%.1fs'
          % (block['block_id'], prob.mean(), float((prob > 0.5).mean()), len(rows),
             sum(areas), areas[len(areas) // 2], areas[-1], t_model, t_vec))
    if rows:
        print('   first date', epoch.date_for_bounds(rows[0]['native_bounds'], dates, ep['tile_origin']))
