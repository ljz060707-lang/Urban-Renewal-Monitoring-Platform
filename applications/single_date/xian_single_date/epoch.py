"""One actual observation epoch: blockwise B0 observation, no temporal model, no GT."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from . import b0, common

COLUMNS = ['observation_id', 'epoch_id', 'epoch_label', 'block_id', 'component_index',
           'pixel_area', 'mean_probability', 'area_map_m2', 'native_xmin', 'native_ymin',
           'native_xmax', 'native_ymax', 'source_edge_partial_risk', 'actual_date',
           'date_status', 'covering_tiles', 'geometry']


def tile_dates(epoch):
    """Per-tile recovered capture dates for this GEID task (read-only ledger)."""
    t = pd.read_parquet(epoch['date_table'],
                        columns=['task_name', 'tile_x', 'tile_y', 'actual_capture_date', 'tile_present'])
    t = t[t['task_name'] == epoch['epoch_label']]
    t = t[t['tile_present'] == 1]
    d = {}
    for tx, ty, dt in zip(t['tile_x'].astype('int64'), t['tile_y'].astype('int64'),
                          t['actual_capture_date']):
        d[(int(tx), int(ty))] = str(dt)
    return d


def date_for_bounds(bounds, dates, origin):
    ox, oy = origin
    x0, y0, x1, y1 = bounds
    tx0 = ox + int(math.floor(x0 / 256.0))
    tx1 = ox + int(math.floor(max(x0, x1 - 1e-6) / 256.0))
    ty0 = oy + int(math.floor(y0 / 256.0))
    ty1 = oy + int(math.floor(max(y0, y1 - 1e-6) / 256.0))
    found = set()
    total = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            total += 1
            found.add(dates.get((tx, ty), 'DATE_UNKNOWN'))
    if not found:
        return 'DATE_UNKNOWN', 'NO_TILE_LEDGER_ENTRY', total
    if len(found) == 1:
        only = next(iter(found))
        return only, ('SINGLE_TILE_DATE' if only != 'DATE_UNKNOWN' else 'DATE_UNKNOWN'), total
    return 'MIXED_ACROSS_TILES', 'MIXED_TILE_DATES', total


def run(epoch, state=None, log=print):
    eid = epoch['epoch_id']
    out = common.RUNS / eid
    blocks_dir = out / 'blocks'
    blocks_dir.mkdir(parents=True, exist_ok=True)
    done_file = out / 'BLOCK_STATUS.json'
    done = common.read_json(done_file) if done_file.exists() else {'completed': [], 'failed': {}}
    done.setdefault('completed', [])
    done.setdefault('failed', {})

    with rasterio.open(epoch['mosaic']) as ds:
        affine = ds.transform
        src_crs = str(ds.crs)
    _tr = Transformer.from_crs(src_crs, common.METRIC_CRS, always_xy=True)
    to_metric = lambda xy: np.column_stack(_tr.transform(xy[:, 0], xy[:, 1]))
    dates = tile_dates(epoch)
    plan = b0.block_plan(epoch['mosaic'])
    log('%s: %d core blocks, source %s, dates recovered for %d tiles'
        % (eid, len(plan), src_crs, len(dates)))

    _, model_hash = b0.model()
    started = time.time()
    for block in plan:
        bid = block['block_id']
        target = blocks_dir / (bid + '.parquet')
        if target.exists() and bid in done['completed']:
            continue
        t0 = time.time()
        rgb = b0.read_rgb(epoch['mosaic'], block)
        t_read = time.time() - t0
        t0 = time.time()
        prob = b0.probability(rgb)
        t_model = time.time() - t0
        del rgb
        t0 = time.time()
        rows = b0.components(prob, block, affine, to_metric)
        del prob
        t_vec = time.time() - t0

        records = []
        for r in rows:
            geom = r['geometry']
            actual, dstatus, ntiles = date_for_bounds(r['native_bounds'], dates, epoch['tile_origin'])
            uid = 'OBS_' + common.sha256_bytes(
                ('%s|%s|%d|%s|%s' % (eid, bid, r['component_index'],
                                     [round(v, 3) for v in r['native_bounds']], model_hash)).encode()
            )[:28]
            records.append(dict(
                observation_id=uid, epoch_id=eid, epoch_label=epoch['epoch_label'],
                block_id=bid, component_index=r['component_index'],
                pixel_area=r['pixel_area'], mean_probability=r['mean_probability'],
                area_map_m2=float(geom.area),
                native_xmin=r['native_bounds'][0], native_ymin=r['native_bounds'][1],
                native_xmax=r['native_bounds'][2], native_ymax=r['native_bounds'][3],
                source_edge_partial_risk=bool(r['source_edge_partial_risk']),
                actual_date=actual, date_status=dstatus, covering_tiles=ntiles,
                geometry=geom.wkb,
            ))
        frame = pd.DataFrame(records, columns=COLUMNS)
        tmp = target.with_suffix('.parquet.tmp')
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, target)
        done['completed'].append(bid)
        elapsed = time.time() - started
        remaining = len(plan) - len(done['completed'])
        common.atomic_write_json(done_file, done)
        if state is not None:
            state.update(
                phase='B0_BLOCK_INFERENCE', epoch=eid, block=bid,
                blocks_completed=len(done['completed']), blocks_total=len(plan),
                objects_last_block=int(len(frame.index)),
                last_block_seconds=round(t_read + t_model + t_vec, 1),
                read_s=round(t_read, 1), model_s=round(t_model, 1), vector_s=round(t_vec, 1),
                eta_hours=round(elapsed / max(1, len(done['completed'])) * remaining / 3600, 2),
                elapsed_hours=round(elapsed / 3600, 2),
            )
        log('%s %s objects=%d read=%.0fs model=%.0fs vector=%.0fs eta=%.1fh'
            % (eid, bid, len(records), t_read, t_model, t_vec,
               elapsed / max(1, len(done['completed'])) * remaining / 3600))
    return done
