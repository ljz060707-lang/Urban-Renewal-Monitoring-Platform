"""Per-epoch product assembly, reference linking, delivery matrices and readback.

Everything here consumes committed block observations.  No model is called, no
threshold is chosen, and the 2019 GT is used only as a designated reference map:
its polygons are never copied into an epoch product and never used to move, grow
or shrink an observed roof.
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from shapely.affinity import translate as shp_translate
from shapely.geometry import box, Polygon
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

from . import common, rasters

POSITION_CODES = {'NO_DATA': 0, 'REFERENCE_SUPPORTED': 1, 'MODEL_ONLY_UNRESOLVED': 2,
                  'SOURCE_OR_GEOMETRY_UNRESOLVED': 3}
LINK_CODES = {'NO_DATA': 0, 'LINKED_COMPATIBLE': 1, 'AMBIGUOUS': 2, 'NO_MATCH': 3}
UNKNOWN_BITS = {'SOURCE_EDGE_PARTIAL': 1, 'PROBABILITY_NEAR_THRESHOLD': 2,
                'REFERENCE_AMBIGUOUS': 4, 'DATE_MIXED_IN_OBJECT': 8,
                'DATE_UNKNOWN': 16, 'GEOMETRY_REPAIRED': 32}


# --------------------------------------------------------------------------- io
def _load_epoch(eid):
    blocks = sorted((common.RUNS / eid / 'blocks').glob('*.parquet'))
    if not blocks:
        raise RuntimeError('no committed blocks for ' + eid)
    df = pd.concat([pd.read_parquet(p) for p in blocks], ignore_index=True)
    df = df.drop_duplicates(subset=['observation_id']).reset_index(drop=True)
    return df, len(blocks)


def repair_dates(df, epoch, log=print):
    """Attribute every object to the registered tile capture date.

    The block writer stores native bounds relative to the analysed window, so the
    window origin must be added before the tile lookup.  Doing it here keeps the
    frozen inference snapshot untouched and guarantees one consistent attribution
    for every block of an epoch, including blocks written before the defect was found.
    """
    from . import b0 as b0_mod
    from . import epoch as epoch_mod
    dates = epoch_mod.tile_dates(epoch)
    plan = {b['block_id']: (b['rc0'], b['rr0']) for b in b0_mod.block_plan(epoch['mosaic'])}
    origin = epoch['tile_origin']
    xs0 = df['native_xmin'].to_numpy('float64')
    ys0 = df['native_ymin'].to_numpy('float64')
    xs1 = df['native_xmax'].to_numpy('float64')
    ys1 = df['native_ymax'].to_numpy('float64')
    bids = df['block_id'].to_numpy()
    out_date = np.empty(len(df), dtype=object)
    out_status = np.empty(len(df), dtype=object)
    out_tiles = np.zeros(len(df), 'int64')
    cache = {}
    for i in range(len(df)):
        key = bids[i]
        off = plan.get(key)
        if off is None:
            out_date[i], out_status[i], out_tiles[i] = 'DATE_UNKNOWN', 'NO_BLOCK_ORIGIN', 0
            continue
        b = (xs0[i] + off[0], ys0[i] + off[1], xs1[i] + off[0], ys1[i] + off[1])
        ck = (key, round(b[0], 3), round(b[1], 3), round(b[2], 3), round(b[3], 3))
        hit = cache.get(ck)
        if hit is None:
            hit = epoch_mod.date_for_bounds(b, dates, origin)
            cache[ck] = hit
        out_date[i], out_status[i], out_tiles[i] = hit
    df['actual_date'] = out_date
    df['date_status'] = out_status
    df['covering_tiles'] = out_tiles
    # a tile present in the mosaic but with no recovered date must stay Unknown
    bad = np.isin(df['actual_date'].to_numpy().astype(str), ['', 'nan', 'None', 'NaT'])
    if bad.any():
        df.loc[bad, 'actual_date'] = 'DATE_UNKNOWN'
        df.loc[bad, 'date_status'] = 'DATE_UNKNOWN'
    log('%s: date attribution repaired over %d objects, ledger dates %s'
        % (epoch['epoch_id'], len(df),
           dict(sorted((str(k), int(v)) for k, v in
                       zip(*np.unique(out_date.astype(str), return_counts=True))))))
    return df


def _aspect_vectorised(geoms):
    mrr = shapely.minimum_rotated_rectangle(geoms)
    coords = shapely.get_coordinates(mrr)
    if coords.shape[0] != 5 * len(geoms):
        return np.full(len(geoms), np.nan), np.full(len(geoms), np.nan), np.full(len(geoms), np.nan)
    xs = coords[:, 0].reshape(len(geoms), 5)
    ys = coords[:, 1].reshape(len(geoms), 5)
    a = np.hypot(xs[:, 1] - xs[:, 0], ys[:, 1] - ys[:, 0])
    b = np.hypot(xs[:, 2] - xs[:, 1], ys[:, 2] - ys[:, 1])
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    ratio = np.full(len(geoms), np.nan)
    ok = lo > 1e-9
    ratio[ok] = hi[ok] / lo[ok]
    return ratio, lo, hi


def _reference_index():
    path = common.REF2019 / 'ROOF_GT2019_REFERENCE.gpkg'
    if not path.exists():
        return None
    g = gpd.read_file(path, engine='pyogrio')
    geoms = g.geometry.values
    cent = shapely.centroid(geoms)
    return dict(geoms=geoms, centroids=cent, fids=g['original_fid'].to_numpy(dtype='int64'),
                areas=shapely.area(geoms), tree=shapely.STRtree(cent))


def link_reference(geoms, ref, max_distance=30.0):
    """Mutual-nearest reference link inside a distance cap.  NOT identity proof.

    shapely's query_nearest returns a (2 x N) index array whose row 0 is the input
    index and row 1 the tree index, plus a separate distance array.  The two rows must
    be read together and never unpacked as (indices, distances).
    """
    n = len(geoms)
    link = np.zeros(n, np.uint8)
    fid = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    if ref is None or n == 0:
        return link, fid, dist, ratio
    cent = shapely.centroid(geoms)
    areas = shapely.area(geoms)
    res, r_distance = ref['tree'].query_nearest(cent, max_distance=max_distance,
                                                return_distance=True, all_matches=False)
    if res.size == 0:
        link[:] = LINK_CODES['NO_MATCH']
        return link, fid, dist, ratio
    obj_idx = np.asarray(res[0])
    ref_idx = np.asarray(res[1])
    obj_tree = shapely.STRtree(cent)
    back, _ = obj_tree.query_nearest(ref['centroids'][ref_idx], max_distance=max_distance,
                                     return_distance=True, all_matches=False)
    back_input = np.asarray(back[0])
    back_tree = np.asarray(back[1])
    mutual = np.zeros(len(obj_idx), bool)
    if back_input.size:
        mutual[back_input] = back_tree[back_input] == obj_idx[back_input]
    for k in range(len(obj_idx)):
        o = int(obj_idx[k])
        r = int(ref_idx[k])
        if not mutual[k]:
            link[o] = LINK_CODES['AMBIGUOUS']
            continue
        rr = max(areas[o], ref['areas'][r]) / max(1e-9, min(areas[o], ref['areas'][r]))
        if rr <= 2.0:
            link[o] = LINK_CODES['LINKED_COMPATIBLE']
            fid[o] = int(ref['fids'][r])
            dist[o] = float(r_distance[k])
            ratio[o] = float(rr)
        else:
            link[o] = LINK_CODES['AMBIGUOUS']
    link[link == 0] = LINK_CODES['NO_MATCH']
    return link, fid, dist, ratio


# ----------------------------------------------------------------- per epoch
def build_epoch(epoch, ref, state=None, log=print):
    eid = epoch['epoch_id']
    out = common.SINGLE / epoch['actual_observation_date']
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df, nblocks = _load_epoch(eid)
    df = repair_dates(df, epoch, log=log)
    geoms = shapely.from_wkb(df['geometry'].values)
    if not np.all(shapely.is_valid(geoms)):
        geoms = shapely.make_valid(geoms)
    df['area_raw_m2'] = shapely.area(geoms)
    n = len(df)
    log('%s: %d committed objects from %d blocks' % (eid, n, nblocks))

    ratio, lo, hi = _aspect_vectorised(geoms)
    df['aspect_ratio'] = ratio
    df['long_axis_m'] = hi
    df['short_axis_m'] = lo
    df['aspect_status'] = np.where(np.isnan(ratio), 'UNKNOWN_DEGENERATE',
                                   'METRIC_MRR_DIAGNOSTIC_UNCALIBRATED')
    df['aspect_use'] = 'SCREEN_ONLY_NEVER_DELETES'

    link, fid, dist, area_ratio = link_reference(geoms, ref)
    df['reference_link_code'] = link
    inv_link = {v: k for k, v in LINK_CODES.items()}
    df['reference_link_status'] = [inv_link[int(v)] for v in link]
    df['reference_match_fid'] = fid
    df['reference_centroid_distance_m'] = dist
    df['reference_area_ratio'] = area_ratio

    near = np.abs(df['mean_probability'].to_numpy().astype('float64') - common.B0_THRESHOLD) < 0.05
    edge = df['source_edge_partial_risk'].to_numpy().astype(bool)
    mixed = df['date_status'].to_numpy() == 'MIXED_ACROSS_TILES'
    unknown_date = df['date_status'].to_numpy() == 'DATE_UNKNOWN'
    bits = np.zeros(n, np.uint8)
    bits = np.where(edge, bits | UNKNOWN_BITS['SOURCE_EDGE_PARTIAL'], bits)
    bits = np.where(near, bits | UNKNOWN_BITS['PROBABILITY_NEAR_THRESHOLD'], bits)
    bits = np.where(link == LINK_CODES['AMBIGUOUS'], bits | UNKNOWN_BITS['REFERENCE_AMBIGUOUS'], bits)
    bits = np.where(mixed, bits | UNKNOWN_BITS['DATE_MIXED_IN_OBJECT'], bits)
    bits = np.where(unknown_date, bits | UNKNOWN_BITS['DATE_UNKNOWN'], bits)
    df['unknown_bitmask'] = bits

    supported = link == LINK_CODES['LINKED_COMPATIBLE']
    unresolved_source = edge | unknown_date
    position = np.full(n, POSITION_CODES['MODEL_ONLY_UNRESOLVED'], np.uint8)
    position[supported] = POSITION_CODES['REFERENCE_SUPPORTED']
    position[unresolved_source] = POSITION_CODES['SOURCE_OR_GEOMETRY_UNRESOLVED']
    df['position_status_code'] = position
    inv_pos = {v: k for k, v in POSITION_CODES.items()}
    df['position_status'] = [inv_pos[int(v)] for v in position]
    df['area_managed_m2'] = df['area_raw_m2']
    df['area_physical_rectified_m2'] = np.nan
    df['physical_area_status'] = 'UNRESOLVED_NO_VIEW_GEOMETRY'
    df['formal_weight'] = 0

    frame = gpd.GeoDataFrame(df.drop(columns=['geometry']), geometry=list(geoms),
                             crs=common.METRIC_CRS)
    frame.to_file(out / 'ROOF_RAW.gpkg', layer='roof_raw', driver='GPKG', engine='pyogrio')
    frame.to_file(out / 'ROOF_MANAGED.gpkg', layer='roof_managed', driver='GPKG', engine='pyogrio')

    # ---- reference-fitted development variant (never folded into the main product)
    dev = frame[supported].copy()
    if len(dev):
        gt_cent = ref['centroids']
        fid_to_index = {int(f): i for i, f in enumerate(ref['fids'])}
        src = shapely.centroid(dev.geometry.values)
        tgt = np.array([gt_cent[fid_to_index[int(f)]] for f in dev['reference_match_fid']])
        dx = shapely.get_x(tgt) - shapely.get_x(src)
        dy = shapely.get_y(tgt) - shapely.get_y(src)
        # shapely.affinity.translate is scalar-only, so the per-object offset is
        # applied one geometry at a time; the count is bounded by the linked subset.
        moved = np.array([shp_translate(g, xoff=float(a), yoff=float(b))
                          for g, a, b in zip(dev.geometry.values, dx, dy)], dtype=object)
        gt_geom = np.array([ref['geoms'][fid_to_index[int(f)]] for f in dev['reference_match_fid']],
                           dtype=object)
        inter = shapely.area(shapely.intersection(moved, gt_geom))
        uni = shapely.area(shapely.union(moved, gt_geom))
        dev['fit_dx_m'] = dx
        dev['fit_dy_m'] = dy
        dev['fit_distance_m'] = np.hypot(dx, dy)
        dev['fit_iou_vs_shared_reference'] = np.where(uni > 0, inter / uni, np.nan)
        dev['fit_role'] = 'REFERENCE_FITTED_DEVELOPMENT'
        dev['independent_accuracy'] = 'NOT_MEASURED_SHARED_REFERENCE'
        dev['geometry'] = moved
        dev = gpd.GeoDataFrame(dev, geometry='geometry', crs=common.METRIC_CRS)
        dev.to_file(out / 'ROOF_REFERENCE_FITTED_DEVELOPMENT.gpkg', layer='reference_fitted_development',
                    driver='GPKG', engine='pyogrio')
        fit_summary = dict(
            objects=int(len(dev)), mean_shift_m=float(np.mean(np.hypot(dx, dy))),
            median_shift_m=float(np.median(np.hypot(dx, dy))),
            p90_shift_m=float(np.percentile(np.hypot(dx, dy), 90)),
            mean_iou_against_shared_reference=float(np.nanmean(dev['fit_iou_vs_shared_reference'])),
            caveat='the fit and its residual come from the same GT; this is not independent accuracy',
        )
    else:
        fit_summary = dict(objects=0, caveat='no reference-compatible links')

    # ---- rasters on the frozen common grid
    grid10 = rasters.template_grid(common.TEMPLATE, 1)
    grid1 = rasters.template_grid(common.TEMPLATE, common.FINE_SUBDIVISION)
    aoi = rasters.grid_bounds(grid10)
    support = _support_polygon(epoch, grid10)
    objs = list(geoms)
    run_ids = np.arange(1, n + 1)

    rasters.banded_rasterize(out / 'ROOF_MANAGED.tif', grid1, objs, (run_ids * 0 + 1).tolist(),
                             dtype='uint8', nodata=0)
    rasters.banded_rasterize(out / 'ROOF_INSTANCE_ID.tif', grid1, objs, run_ids.tolist(),
                             dtype='uint32', nodata=0)
    frac, ident10, assigned, cells = rasters.exact_fraction(objs, grid10, support=support, ids=run_ids)
    _write_tif(out / 'ROOF_FRACTION_10M.tif', frac.astype('float32'), grid10,
               'roof_fraction_full_cell_denominator')
    del ident10
    support_arr = _support_raster(support, grid10)
    _write_tif(out / 'SOURCE_SUPPORT.tif', support_arr.astype('float32'), grid10,
               'source_support_fraction_of_cell')
    del support_arr
    _write_tif(out / 'POSITION_STATUS.tif',
               rasters_rasterize(objs, position, grid10), grid10, 'position_status_code')
    _write_tif(out / 'REFERENCE_LINK_STATUS.tif',
               rasters_rasterize(objs, link, grid10), grid10, 'reference_link_status_code')
    _write_tif(out / 'UNKNOWN_AND_CONFLICT.tif',
               rasters_rasterize(objs, bits, grid10), grid10, 'unknown_conflict_bitmask')
    date_codes, date_map_arr = _date_map(epoch, grid10)
    _write_tif(out / 'ACTUAL_DATE_MAP.tif', date_map_arr, grid10, 'actual_capture_date_code')
    del date_map_arr
    gc.collect()

    # ---- area ledger
    # The candidate set is a set of connected components under single-owner block
    # assignment, so member polygons are disjoint by construction and a cascaded
    # GEOS union over 200k+ staircase polygons is both unnecessary and pathologically
    # slow.  The union area is therefore evaluated exactly as
    #     sum(area) - sum(area of all intersecting pairs)
    # which equals the union whenever no point belongs to three members; the pair
    # count is reported so the assumption is checkable.
    union_area, overlap_area, overlap_pairs = _union_area(objs, support)
    within = _within_support_area(objs, support, overlap_area)
    sum_area = float(df['area_managed_m2'].sum())
    support_area = float(support.area)
    summary = dict(
        epoch_id=eid, epoch_label=epoch['epoch_label'],
        nominal_input_date=epoch['nominal_input_date'],
        actual_observation_date=epoch['actual_observation_date'],
        date_source=epoch['date_source'], date_status=epoch['date_status'],
        dominant_share_pct=epoch['dominant_share_pct'], mixing_severity=epoch['mixing_severity'],
        assumption=epoch['assumption'],
        instances=int(n),
        reference_supported=int((position == POSITION_CODES['REFERENCE_SUPPORTED']).sum()),
        model_only_unresolved=int((position == POSITION_CODES['MODEL_ONLY_UNRESOLVED']).sum()),
        source_or_geometry_unresolved=int((position == POSITION_CODES['SOURCE_OR_GEOMETRY_UNRESOLVED']).sum()),
        source_edge_partial=int(df['source_edge_partial_risk'].sum()),
        object_area_sum_m2=round(sum_area, 4),
        union_area_m2=round(union_area, 4),
        union_area_method='sum_minus_pair_overlap',
        overlap_pair_count=int(overlap_pairs),
        overlap_area_m2=round(overlap_area, 4),
        within_support_union_m2=round(within, 4),
        support_area_m2=round(support_area, 4),
        unresolved_area_m2=round(support_area - within, 4),
        object_area_sum_ha=round(sum_area / 1e4, 6),
        union_area_ha=round(union_area / 1e4, 6),
        object_area_sum_km2=round(sum_area / 1e6, 9),
        union_area_km2=round(union_area / 1e6, 9),
        fraction_integral_m2=round(assigned, 4),
        fraction_cells_touched=int(cells),
        fraction_cell_m2=abs(grid10['affine'].a * grid10['affine'].e),
        fraction_integral_vs_object_sum_m2=round(assigned - sum_area, 6),
        fraction_integral_minus_union_m2=round(assigned - union_area, 6),
        fraction_integral_minus_within_support_m2=round(assigned - within, 6),
        fraction_note='the fraction layer is additive per object, so its integral must equal the object area sum; the difference against the union is the overlap, reported separately',
        area_semantics='ROOF_PLANIMETRIC_AREA_ON_REFERENCE_GRID',
        metric_crs=common.METRIC_CRS,
        blocks=int(nblocks),
        grid=dict(width=grid10['width'], height=grid10['height'],
                  transform=list(grid10['affine'])[:6], crs=grid10['crs'],
                  fine_subdivision=common.FINE_SUBDIVISION),
        zero_semantics='0 outside the candidate set is not certified reliable background',
        v2_status='NOT_APPLICABLE_NOT_RUN_IN_THIS_PASS',
        reference_fitted_development=fit_summary,
    )
    common.atomic_write_csv(out / 'AREA_SUMMARY.csv', [summary])

    obj_rows = df[['observation_id', 'block_id', 'pixel_area', 'mean_probability',
                   'area_raw_m2', 'area_managed_m2', 'aspect_ratio', 'long_axis_m',
                   'short_axis_m', 'aspect_status', 'source_edge_partial_risk',
                   'actual_date', 'date_status', 'covering_tiles', 'position_status',
                   'reference_link_status', 'reference_match_fid',
                   'reference_centroid_distance_m', 'reference_area_ratio',
                   'unknown_bitmask']].copy()
    obj_rows.insert(1, 'epoch_id', eid)
    common.atomic_write_csv(out / 'OBJECT_AREA.csv', obj_rows.to_dict('records'))

    common.atomic_write_json(out / 'ACTUAL_DATE_MAP_CODES.json', date_codes)
    manifest = dict(
        product='SINGLE_DATE_ROOF_PRODUCT', epoch_id=eid, epoch_label=epoch['epoch_label'],
        actual_observation_date=epoch['actual_observation_date'],
        nominal_input_date=epoch['nominal_input_date'],
        date_source=epoch['date_source'], date_status=epoch['date_status'],
        dominant_share_pct=epoch['dominant_share_pct'],
        mixing_severity=epoch['mixing_severity'], assumption=epoch['assumption'],
        source=dict(mosaic=epoch['mosaic'], mosaic_sha256=common.sha256(epoch['mosaic']),
                    crs='EPSG:4326', pixel_size_deg=2.6822e-06,
                    pixel_size_m_approx=[0.247, 0.297],
                    tiles=int(epoch['verified_tile_count'] + epoch['missing_tile_count']),
                    missing_tiles=int(epoch['missing_tile_count'])),
        model=dict(role='B0_RETAINED_TARGET_DOMAIN_BASELINE', threshold=common.B0_THRESHOLD,
                   comparator='>0.5', tiling='256/192 weighted fusion',
                   connectivity=8, batch_scheduler=common.BATCH,
                   checkpoint_sha256=_checkpoint_sha()),
        gt_free_inference=True, gt_mounted_in_prediction=False,
        v2_applied=False, facade_applied=False, temporal_applied=False,
        m3_temporal='DISABLED_FOR_SINGLE_DATE',
        reference=dict(product='ROOF_GT2019_REFERENCE', role=common.GT_ROLE,
                       used_for='POSITIONAL REFERENCE LINK ONLY',
                       geometry_never_copied_into_this_product=True),
        files={p.name: dict(sha256=common.sha256(p), bytes=p.stat().st_size)
               for p in sorted(out.iterdir()) if p.is_file() and p.name != 'PRODUCT_MANIFEST.json'},
        area=dict(object_area_sum_m2=summary['object_area_sum_m2'],
                  union_area_m2=summary['union_area_m2'],
                  fraction_integral_m2=summary['fraction_integral_m2']),
        elapsed_seconds=round(time.time() - t0, 1),
    )
    common.atomic_write_json(out / 'PRODUCT_MANIFEST.json', manifest)
    log('%s: products written in %.1f min, union %.0f m2, overlap pairs %d'
        % (eid, (time.time() - t0) / 60, union_area, overlap_pairs))
    return summary


def _checkpoint_sha():
    try:
        m = common.read_json(common.ROUTING)
        return m['roof']['checkpoint_sha256']
    except Exception:
        return None


def rasters_rasterize(geoms, values, grid):
    from rasterio.features import rasterize as _rz
    gg = [(g, int(v)) for g, v in zip(geoms, values) if g is not None and not g.is_empty]
    return _rz(gg, out_shape=(grid['height'], grid['width']), transform=grid['affine'],
               dtype='uint8', fill=0)


def _support_polygon(epoch, grid):
    with rasterio.open(epoch['mosaic']) as ds:
        b = ds.bounds
        crs = str(ds.crs)
    poly = box(b.left, b.bottom, b.right, b.top)
    tr = Transformer.from_crs(crs, common.METRIC_CRS, always_xy=True)
    return shp_transform(tr.transform, poly).intersection(rasters.grid_bounds(grid))


def _support_raster(support, grid):
    from rasterio.features import rasterize as _rz
    return _rz([(support, 1.0)], out_shape=(grid['height'], grid['width']),
               transform=grid['affine'], dtype='float32', fill=0.0)


def _date_map(epoch, grid):
    """Per-10 m-cell actual capture date code, from the registered tile ledger."""
    t = pd.read_parquet(epoch['date_table'],
                        columns=['task_name', 'tile_x', 'tile_y', 'actual_capture_date', 'tile_present'])
    t = t[(t['task_name'] == epoch['epoch_label']) & (t['tile_present'] == 1)]
    t = t[t['actual_capture_date'].notna()]
    t = t[t['actual_capture_date'].astype(str).str.len() >= 8]
    dates = sorted(t['actual_capture_date'].astype(str).unique().tolist())
    codes = {d: i + 1 for i, d in enumerate(dates)}
    ox, oy = epoch['tile_origin']
    tx = t['tile_x'].to_numpy('int64') - ox
    ty = t['tile_y'].to_numpy('int64') - oy
    vals = np.array([codes.get(str(v), 0) for v in t['actual_capture_date']], np.uint16)
    tw = int(tx.max()) + 2
    th = int(ty.max()) + 2
    lut = np.zeros((th, tw), np.uint16)
    lut[ty, tx] = vals
    with rasterio.open(epoch['mosaic']) as ds:
        aff = ds.transform
        src_crs = str(ds.crs)
    tr = Transformer.from_crs(common.METRIC_CRS, src_crs, always_xy=True)
    out = np.zeros((grid['height'], grid['width']), np.uint16)
    step = 512
    for r0 in range(0, grid['height'], step):
        r1 = min(grid['height'], r0 + step)
        rows = np.arange(r0, r1)
        cols = np.arange(0, grid['width'])
        X, Y = np.meshgrid(grid['affine'].c + (cols + 0.5) * grid['affine'].a,
                           grid['affine'].f + (rows + 0.5) * grid['affine'].e)
        lon, lat = tr.transform(X, Y)
        ti = np.floor((lon - aff.c) / aff.a / 256.0).astype('int64')
        tj = np.floor((lat - aff.f) / aff.e / 256.0).astype('int64')
        ok = (ti >= 0) & (ti < tw) & (tj >= 0) & (tj < th)
        block = np.zeros((r1 - r0, grid['width']), np.uint16)
        block[ok] = lut[tj[ok], ti[ok]]
        out[r0:r1] = block
    return dict(codes={str(k): int(v) for k, v in codes.items()},
                reverse={str(v): k for k, v in codes.items()},
                zero_means='no registered tile at this cell'), out


def _union_area(geoms, support=None):
    """Exact union area for a member set that is disjoint except for pair overlaps."""
    if not geoms:
        return 0.0, 0.0, 0
    arr = np.asarray(geoms, dtype=object)
    areas = shapely.area(arr)
    total = float(areas.sum())
    tree = shapely.STRtree(arr)
    pairs = tree.query(arr, predicate='intersects')
    i = np.asarray(pairs[0])
    j = np.asarray(pairs[1])
    keep = i < j
    i, j = i[keep], j[keep]
    if i.size == 0:
        return total, 0.0, 0
    inter = float(shapely.area(shapely.intersection(arr[i], arr[j])).sum())
    return total - inter, inter, int(i.size)


def _within_support_area(geoms, support, overlap_area=0.0):
    """Object area restricted to the source support footprint (same disjointness)."""
    if not geoms:
        return 0.0
    arr = np.asarray(geoms, dtype=object)
    if support is None:
        return float(shapely.area(arr).sum())
    clipped = shapely.intersection(arr, support)
    total = float(shapely.area(clipped).sum())
    return total - min(overlap_area, total)


def _write_tif(path, array, grid, band_name, nodata=None):
    with rasterio.open(path, 'w', driver='GTiff', height=grid['height'], width=grid['width'],
                       count=1, dtype=array.dtype, crs=grid['crs'], transform=grid['affine'],
                       nodata=nodata, compress='deflate', tiled=True, blockxsize=512,
                       blockysize=512, BIGTIFF='YES') as ds:
        ds.write(array, 1)
        ds.set_band_description(1, band_name)


# --------------------------------------------------------------------- global
REQUIRED_EPOCH_FILES = [
    ('ROOF_RAW.gpkg', True), ('ROOF_MANAGED.gpkg', True), ('ROOF_MANAGED.tif', True),
    ('ROOF_INSTANCE_ID.tif', True), ('ROOF_V2_ALTERNATIVE.gpkg', True),
    ('ROOF_FRACTION_10M.tif', True), ('SOURCE_SUPPORT.tif', True),
    ('POSITION_STATUS.tif', True), ('REFERENCE_LINK_STATUS.tif', True),
    ('ACTUAL_DATE_MAP.tif', True), ('UNKNOWN_AND_CONFLICT.tif', True),
    ('OBJECT_AREA.csv', True), ('AREA_SUMMARY.csv', True), ('PRODUCT_MANIFEST.json', True),
]
REQUIRED_REFERENCE_FILES = [
    'ROOF_GT2019_REFERENCE.gpkg', 'ROOF_GT2019_REFERENCE.tif',
    'ROOF_GT2019_INSTANCE_ID.tif', 'ROOF_GT2019_FRACTION_10M.tif',
    'GT2019_OBJECT_AREA.csv', 'GT2019_AREA_SUMMARY.csv',
    'REFERENCE_GEOMETRY_CONSISTENCY.json', 'PRODUCT_MANIFEST.json',
]


def build(state=None, log=print):
    ref = _reference_index()
    if ref is None:
        raise RuntimeError('2019 reference product missing; run reference_gt first')
    summaries = []
    for epoch in common.EPOCHS:
        summaries.append(build_epoch(epoch, ref, state=state, log=log))
        gc.collect()
    common.atomic_write_csv(common.RUN_ROOT / 'ALL_SINGLE_DATE_AREAS.csv', summaries)

    scope = []
    with rasterio.open(common.TEMPLATE) as ds:
        tpl = dict(crs=str(ds.crs), width=ds.width, height=ds.height,
                   transform=list(ds.transform)[:6],
                   bounds=[float(v) for v in ds.bounds])
    for epoch in common.EPOCHS:
        with rasterio.open(epoch['mosaic']) as ds:
            mb = [float(v) for v in ds.bounds]
            mpix = ds.width * ds.height
            mcrs = str(ds.crs)
            mres = [abs(ds.transform.a), abs(ds.transform.e)]
        scope.append(dict(
            epoch_label=epoch['epoch_label'], epoch_id=epoch['epoch_id'],
            nominal_input_date=epoch['nominal_input_date'],
            actual_observation_date=epoch['actual_observation_date'],
            date_source=epoch['date_source'], date_status=epoch['date_status'],
            dominant_share_pct=epoch['dominant_share_pct'],
            mixing_severity=epoch['mixing_severity'], assumption=epoch['assumption'],
            mosaic=epoch['mosaic'], mosaic_crs=mcrs, mosaic_bounds=mosaic_bounds(mb),
            mosaic_pixels=int(mpix), pixel_size_deg=mres,
            registered_tiles=int(epoch['verified_tile_count'] + epoch['missing_tile_count']),
            tiles_without_date=int(epoch['missing_tile_count']),
            covers_frozen_aoi_fraction=round(_covers(epoch), 4),
            model='B0_RETAINED_TARGET_DOMAIN_BASELINE', gt_mounted=False,
            training=False, cloud=False,
        ))
    common.atomic_write_csv(common.RUN_ROOT / 'INPUT_SCOPE_AND_DATES.csv', scope)
    common.atomic_write_json(common.RUN_ROOT / 'SCOPE_DETAIL.json',
                             dict(template=tpl, epochs=scope))

    register = dict(
        schema='XIAN_R5_REFERENCE_ROLE_REGISTER_V1',
        designated_reference=dict(
            product='ROOF_GT2019_REFERENCE', source=str(common.GT_SHAPEFILE),
            reference_year=2019, acquisition_date_status='UNVERIFIED',
            geometry_role=common.GT_ROLE, geometry_source='ORIGINAL_GT_FID',
            alignment_to_designated_reference='BY_CONSTRUCTION',
            independent_absolute_accuracy='NOT_MEASURED',
            used_for=['reference mapping product', 'positional reference link status'],
            not_used_for=['independent accuracy of itself', 'copying into epoch products',
                          'snapping epoch geometry onto GT', 'formal GT10m supervision weight'],
        ),
        epoch_products=dict(
            gt_free_inference=True,
            reference_link_is_identity_proof=False,
            reference_fitted_variant='REFERENCE_FITTED_DEVELOPMENT (development only)',
            formal_gt10m_admission='NOT_GRANTED',
            supervision_weight='0 for every object',
        ),
        prohibited=dict(change_products='not produced in this round',
                        demolition_inference='not produced in this round',
                        three_state='not produced in this round'),
    )
    common.atomic_write_json(common.RUN_ROOT / 'REFERENCE_ROLE_REGISTER.json', register)
    _write_matrix(None)
    _write_index()
    log('assemble: global products written')
    return dict(epochs=summaries)


def mosaic_bounds(b):
    return dict(west=b[0], south=b[1], east=b[2], north=b[3])


def _covers(epoch):
    """Fraction of the frozen AOI that this epoch's source footprint actually covers."""
    with rasterio.open(common.TEMPLATE) as ds:
        aoi = box(*[float(v) for v in ds.bounds])
    with rasterio.open(epoch['mosaic']) as ds:
        b = [float(v) for v in ds.bounds]
        crs = str(ds.crs)
    src = shp_transform(Transformer.from_crs(crs, common.METRIC_CRS, always_xy=True).transform,
                        box(b[0], b[1], b[2], b[3]))
    return float(aoi.intersection(src).area / aoi.area)


def _write_matrix(readback):
    rows = []
    ref = common.REF2019
    for name in REQUIRED_REFERENCE_FILES:
        p = ref / name
        rows.append(dict(scope='reference_2019', product=name,
                         present=bool(p.exists()),
                         bytes=(p.stat().st_size if p.exists() else 0),
                         status=('PRESENT' if p.exists() else 'MISSING'),
                         note=''))
    for epoch in common.EPOCHS:
        d = common.SINGLE / epoch['actual_observation_date']
        for name, required in REQUIRED_EPOCH_FILES:
            p = d / name
            present = bool(p.exists())
            note = ''
            status = 'PRESENT' if present else ('MISSING' if required else 'NOT_APPLICABLE')
            if name == 'ROOF_V2_ALTERNATIVE.gpkg' and not present:
                note = ('V2 M_CORR complete-instance position candidate was not executed in this '
                        'pass; no admission protocol exists for it at city scale, so no substitute '
                        'geometry is fabricated')
                status = 'NOT_PRODUCED'
            rows.append(dict(scope=epoch['actual_observation_date'], product=name,
                             present=present, bytes=(p.stat().st_size if present else 0),
                             status=status, note=note))
    for name in ('ALL_SINGLE_DATE_AREAS.csv', 'INPUT_SCOPE_AND_DATES.csv',
                 'REFERENCE_ROLE_REGISTER.json', 'DELIVERY_MATRIX.csv',
                 'INDEPENDENT_PRODUCT_READBACK.json', 'FINAL_RECEIPT.json',
                 'FINAL_FILE_SHA256.json', 'reports/index.html'):
        p = common.RUN_ROOT / name
        rows.append(dict(scope='total', product=name, present=bool(p.exists()),
                         bytes=(p.stat().st_size if p.exists() else 0),
                         status='PRESENT' if p.exists() else 'PENDING', note=''))
    common.atomic_write_csv(common.RUN_ROOT / 'DELIVERY_MATRIX.csv', rows)
    return rows


def _write_index():
    import html
    rep = common.RUN_ROOT / 'reports'
    rep.mkdir(parents=True, exist_ok=True)
    cards = []
    summaries = []
    try:
        import csv
        with (common.RUN_ROOT / 'ALL_SINGLE_DATE_AREAS.csv').open(encoding='utf-8') as f:
            summaries = list(csv.DictReader(f))
    except Exception:
        pass
    for s in summaries:
        eid = s.get('epoch_id', '')
        cards.append(
            '<section><h2>%s &middot; observed %s</h2>'
            '<p class="meta">label %s &middot; date source %s &middot; mixing %s (dominant %.2f%%)</p>'
            '<table><tr><th>objects</th><th>object area sum</th><th>union</th>'
            '<th>within support</th><th>unresolved</th></tr>'
            '<tr><td>%s</td><td>%.0f m2</td><td>%.0f m2</td><td>%.0f m2</td><td>%.0f m2</td></tr></table>'
            '<p class="meta">reference supported %s &middot; model only %s &middot; source/geometry unresolved %s</p>'
            '<p><a href="../single_date/%s/PRODUCT_MANIFEST.json">manifest</a> '
            '<a href="../single_date/%s/AREA_SUMMARY.csv">area summary</a></p></section>'
            % (html.escape(eid), html.escape(s.get('actual_observation_date', '')),
               html.escape(s.get('epoch_label', '')), html.escape(s.get('date_source', '')),
               html.escape(s.get('mixing_severity', '')), float(s.get('dominant_share_pct') or 0),
               s.get('instances', ''), float(s.get('object_area_sum_m2') or 0),
               float(s.get('union_area_m2') or 0), float(s.get('within_support_union_m2') or 0),
               float(s.get('unresolved_area_m2') or 0), s.get('reference_supported', ''),
               s.get('model_only_unresolved', ''), s.get('source_or_geometry_unresolved', ''),
               html.escape(s.get('actual_observation_date', '')),
               html.escape(s.get('actual_observation_date', ''))))
    doc = ('<!doctype html><meta charset="utf-8"><title>XIAN-R5 single-date city products</title>'
           '<style>body{font-family:Segoe UI,system-ui,sans-serif;margin:32px;max-width:1100px;'
           'line-height:1.5;color:#1b1d21}h1{font-size:22px}h2{font-size:16px;margin:0 0 4px}'
           'section{border:1px solid #d8dade;border-radius:8px;padding:14px 16px;margin:14px 0}'
           '.meta{color:#5a5f68;font-size:12px;margin:2px 0}'
           'table{border-collapse:collapse;font-size:13px;margin:8px 0}'
           'th,td{border:1px solid #e2e4e8;padding:4px 8px;text-align:left}'
           'a{color:#1a5fb4}</style>'
           '<h1>XIAN-R5 &middot; single-date city roof products</h1>'
           '<p class="meta">2019 product is built directly from the designated manual GT and is a '
           'reference map, not a model prediction. Other epochs are model candidates in the same '
           'reference frame with per-epoch quality statements. No change, demolition or stable '
           'products are produced in this round.</p>'
           + ''.join(cards) +
           '<p class="meta"><a href="../reference_2019/PRODUCT_MANIFEST.json">2019 reference manifest</a> '
           '&middot; <a href="../FINAL_RECEIPT.json">final receipt</a> '
           '&middot; <a href="../DELIVERY_MATRIX.csv">delivery matrix</a></p>')
    (rep / 'index.html').write_text(doc, encoding='utf-8')


def readback(state=None, log=print):
    checks = []
    grids = {}
    for epoch in common.EPOCHS:
        d = common.SINGLE / epoch['actual_observation_date']
        entry = dict(epoch_id=epoch['epoch_id'],
                     actual_observation_date=epoch['actual_observation_date'])
        try:
            with rasterio.open(d / 'ROOF_MANAGED.tif') as ds:
                mg = dict(crs=str(ds.crs), transform=[round(float(v), 6) for v in list(ds.transform)[:6]],
                          width=ds.width, height=ds.height, dtype=str(ds.dtypes[0]), nodata=ds.nodata)
                vals = set()
                for r0 in range(0, ds.height, 4096):
                    vals.update(int(v) for v in np.unique(
                        ds.read(1, window=rasterio.windows.Window(0, r0, ds.width,
                                                                  min(4096, ds.height - r0)))).tolist())
            with rasterio.open(d / 'ROOF_INSTANCE_ID.tif') as ds:
                ig = dict(crs=str(ds.crs), transform=[round(float(v), 6) for v in list(ds.transform)[:6]],
                          width=ds.width, height=ds.height, dtype=str(ds.dtypes[0]))
                maxid = 0
                for r0 in range(0, ds.height, 4096):
                    band = ds.read(1, window=rasterio.windows.Window(0, r0, ds.width,
                                                                     min(4096, ds.height - r0)))
                    maxid = max(maxid, int(band.max()))
            with rasterio.open(d / 'ROOF_FRACTION_10M.tif') as ds:
                fr = dict(crs=str(ds.crs), transform=[round(float(v), 6) for v in list(ds.transform)[:6]],
                          width=ds.width, height=ds.height)
                frac_sum = float(np.asarray(ds.read(1), dtype='float64').sum()) * 100.0
            import csv as _csv
            with (d / 'AREA_SUMMARY.csv').open(encoding='utf-8') as f:
                summary = next(_csv.DictReader(f))
            with (d / 'OBJECT_AREA.csv').open(encoding='utf-8') as f:
                nobj = sum(1 for _ in f) - 1
            g = gpd.read_file(d / 'ROOF_RAW.gpkg', engine='pyogrio', rows=1)
            nraw = _gpkg_count(d / 'ROOF_RAW.gpkg')
            entry.update(managed_grid=mg, instance_grid=ig, fraction_grid=fr,
                         class_values=sorted(vals), max_instance_id=maxid,
                         object_rows=int(nobj), gpkg_features=int(nraw),
                         fraction_integral_m2=round(frac_sum, 4),
                         union_area_m2=float(summary['union_area_m2']),
                         within_support_union_m2=float(summary['within_support_union_m2']),
                         fraction_vs_union_error_m2=round(frac_sum - float(summary['within_support_union_m2']), 4),
                         gpkg_crs=str(g.crs),
                         checks=dict(
                             grid_matches_common=bool(
                                 mg['transform'] == [round(float(v), 6) for v in
                                                     list(rasters.template_grid(common.TEMPLATE,
                                                                                common.FINE_SUBDIVISION)['affine'])[:6]]),
                             instance_grid_matches_managed=ig['transform'] == mg['transform'],
                             class_values_ok=sorted(vals) in ([0], [0, 1], [1]),
                             object_rows_equal_gpkg=int(nobj) == int(nraw),
                             max_instance_id_matches=int(maxid) == int(nobj),
                         ))
        except Exception as exc:
            entry['error'] = '%s: %s' % (type(exc).__name__, exc)
        grids[epoch['epoch_id']] = entry
        checks.append(entry)

    same_grid = len({(e['managed_grid']['transform'] if 'managed_grid' in e else None).
                     __str__() for e in checks}) == 1

    # no GT geometry may appear inside an epoch product
    gt_hash = None
    try:
        rg = gpd.read_file(common.REF2019 / 'ROOF_GT2019_REFERENCE.gpkg', engine='pyogrio')
        gt_hash = set(common.sha256_bytes(shapely.to_wkb(g, hex=False))
                      for g in rg.geometry.values)
    except Exception:
        pass
    copied = {}
    for epoch in common.EPOCHS:
        d = common.SINGLE / epoch['actual_observation_date']
        try:
            g = gpd.read_file(d / 'ROOF_RAW.gpkg', engine='pyogrio')
            got = set(common.sha256_bytes(shapely.to_wkb(x, hex=False)) for x in g.geometry.values)
            copied[epoch['epoch_id']] = int(len(got & gt_hash)) if gt_hash else None
        except Exception:
            copied[epoch['epoch_id']] = None

    ref_consistency = None
    try:
        ref_consistency = common.read_json(common.REF2019 / 'REFERENCE_GEOMETRY_CONSISTENCY.json')['status']
    except Exception:
        pass

    # how far each candidate epoch sits from the designated reference in total area
    area_comparison = {}
    try:
        import csv as _csv2
        with (common.RUN_ROOT / 'ALL_SINGLE_DATE_AREAS.csv').open(encoding='utf-8') as f:
            summaries = list(_csv2.DictReader(f))
        rg2 = gpd.read_file(common.REF2019 / 'ROOF_GT2019_REFERENCE.gpkg', engine='pyogrio')
        with rasterio.open(common.EPOCHS[0]['mosaic']) as ds:
            b = [float(v) for v in ds.bounds]
            mcrs = str(ds.crs)
        foot = shp_transform(Transformer.from_crs(mcrs, common.METRIC_CRS, always_xy=True).transform,
                             box(b[0], b[1], b[2], b[3]))
        gt_inside = float(shapely.area(shapely.intersection(rg2.geometry.values, foot)).sum())
        for s in summaries:
            eid = s.get('epoch_id')
            model = float(s.get('within_support_union_m2') or 0)
            sup = float(s.get('support_area_m2') or 0)
            area_comparison[eid] = dict(
                actual_observation_date=s.get('actual_observation_date'),
                model_within_support_m2=round(model, 4),
                gt_within_same_footprint_m2=round(gt_inside, 4),
                ratio_model_over_reference=round(model / gt_inside, 4) if gt_inside else None,
                model_roof_density_pct=round(100 * model / sup, 4) if sup else None,
                reference_roof_density_pct=round(100 * gt_inside / sup, 4) if sup else None,
                interpretation=('total candidate roof area relative to the designated 2019 GT on the '
                                'same ground; a ratio far from 1 quantifies over- or under-detection, '
                                'but the two dates differ so the ratio is not a change measurement'),
            )
    except Exception as exc:
        area_comparison = {'error': '%s: %s' % (type(exc).__name__, exc)}

    matrix = _write_matrix(True)
    missing = [r for r in matrix if r['status'] in ('MISSING', 'NOT_PRODUCED')]
    completeness = 'COMPLETE_REFERENCE_ALIGNED_PRODUCTS' if not missing else 'PARTIAL_COMPLETE_WITH_BLOCKERS'
    accuracy = dict(
        reference_2019_consistency=ref_consistency or 'UNKNOWN',
        reference_2019_note=('agreement with the designated GT is BY CONSTRUCTION and is not a '
                             'model accuracy measurement'),
        other_epochs_absolute_accuracy='NOT_ESTABLISHED',
        other_epochs_note=('same grid, same model, same threshold and same reportable quality '
                           'target as 2019; no independent evidence supports equal accuracy'),
        scientific_accuracy_status='REFERENCE_CONSISTENCY_ONLY_OTHER_EPOCHS_NOT_ESTABLISHED',
        formal_gt10m_admission='NOT_GRANTED',
    )
    readback = dict(
        schema='XIAN_R5_INDEPENDENT_PRODUCT_READBACK_V1',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        epochs=checks,
        all_epochs_share_one_grid=same_grid,
        gt_geometry_bytes_found_in_epoch_products=copied,
        reference_consistency_status=ref_consistency,
        reference_area_comparison=area_comparison,
        delivery_missing=[r for r in missing if r['status'] == 'MISSING'],
        delivery_not_produced=[r for r in missing if r['status'] == 'NOT_PRODUCED'],
        delivery_completeness=completeness,
        scientific_accuracy=accuracy,
    )
    common.atomic_write_json(common.RUN_ROOT / 'INDEPENDENT_PRODUCT_READBACK.json', readback)

    receipt = dict(
        schema='XIAN_R5_SINGLE_DATE_FINAL_RECEIPT_V1',
        run_id=common.RUN_ROOT.name,
        status=completeness,
        delivery_completeness=completeness,
        scientific_accuracy_status=accuracy['scientific_accuracy_status'],
        requested_scope=dict(aoi='frozen 10 m LR template of the Xian city mosaic',
                             aoi_km2=1198.91, epochs=[e['actual_observation_date'] for e in common.EPOCHS]),
        reference=dict(product='ROOF_GT2019_REFERENCE', role=common.GT_ROLE,
                       reference_year=2019, acquisition_date_status='UNVERIFIED'),
        epochs=[dict(epoch_label=e['epoch_label'],
                     actual_observation_date=e['actual_observation_date'],
                     date_status=e['date_status'], dominant_share_pct=e['dominant_share_pct'])
                for e in common.EPOCHS],
        blockers=readback['delivery_not_produced'] + readback['delivery_missing'],
        reference_area_comparison=area_comparison,
        not_produced_by_design=['ADDITION', 'DEMOLITION', 'STABLE', 'three-state change',
                                'cross-epoch area difference as change statistic',
                                'formal GT10m supervision weight'],
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    )
    common.atomic_write_json(common.RUN_ROOT / 'FINAL_RECEIPT.json', receipt)

    hashes = {}
    for p in sorted(common.RUN_ROOT.rglob('*')):
        if p.is_file() and p.suffix.lower() in ('.tif', '.gpkg', '.csv', '.json', '.html',
                                                 '.cmd', '.ps1', '.py', '.log', '.parquet'):
            if 'work' in p.parts or 'blocks' in p.parts:
                continue
            hashes[str(p.relative_to(common.RUN_ROOT))] = dict(
                sha256=common.sha256(p), bytes=p.stat().st_size)
    common.atomic_write_json(common.RUN_ROOT / 'FINAL_FILE_SHA256.json', hashes)
    _write_matrix(True)
    log('readback: %s, %d missing/not-produced' % (completeness, len(missing)))
    return receipt


def _gpkg_count(path):
    import pyogrio
    return int(pyogrio.read_info(path)['features'])
