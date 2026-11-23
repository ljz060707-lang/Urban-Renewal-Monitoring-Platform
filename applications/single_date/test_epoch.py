"""Two-block end-to-end validation of the frozen epoch path, with timing."""
import sys
import time

sys.path.insert(0, r'F:\20261028西安市区结果\XIAN_R5_SINGLE_DATE_20261028_01\CODE_SNAPSHOT')

import numpy as np
import rasterio
from pyproj import Transformer

from xian_single_date import b0, common, epoch

common.CORE = 4096
common.GUARD = 768

ep = common.EPOCHS[0]
plan = b0.block_plan(ep['mosaic'])
print('blocks %d' % len(plan))
# one urban block and one rural edge block
picks = [plan[len(plan) // 2 + 3], plan[0]]
with rasterio.open(ep['mosaic']) as ds:
    affine = ds.transform
    src_crs = str(ds.crs)
_tr = Transformer.from_crs(src_crs, common.METRIC_CRS, always_xy=True)
to_metric = lambda xy: np.column_stack(_tr.transform(xy[:, 0], xy[:, 1]))
dates = epoch.tile_dates(ep)

for block in picks:
    t = time.time()
    rgb = b0.read_rgb(ep['mosaic'], block)
    t_read = time.time() - t
    t = time.time()
    prob = b0.probability(rgb)
    t_model = time.time() - t
    del rgb
    t = time.time()
    rows = b0.components(prob, block, affine, to_metric)
    t_vec = time.time() - t
    del prob
    areas = sorted(r['geometry'].area for r in rows) if rows else [0.0]
    print('%s shape=%dx%d n=%d total_m2=%.0f median_m2=%.1f max_m2=%.0f | read=%.1fs model=%.1fs vec=%.1fs'
          % (block['block_id'], block['width'], block['height'], len(rows), sum(areas),
             areas[len(areas) // 2], areas[-1], t_read, t_model, t_vec))
    if rows:
        r = rows[0]
        print('   sample area=%.2f m2 prob=%.3f edge=%s date=%s'
              % (r['geometry'].area, r['mean_probability'], r['source_edge_partial_risk'],
                 epoch.date_for_bounds(r['native_bounds'], dates, ep['tile_origin'])))
        print('   bounds', [round(v, 1) for v in r['native_bounds']])
        print('   metric', [round(v, 1) for v in r['geometry'].bounds])
    print('   projected mpx/s = %.2f' % ((block['width'] * block['height']) / t_model / 1e6))
