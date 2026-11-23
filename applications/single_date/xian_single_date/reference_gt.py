"""2019 designated manual roof GT -> reference map product.

This branch is the user-authorised REFERENCE_DERIVED_2019 mapping product.  It is
NOT a model prediction, is NOT GT-free inference, and is NOT an independent accuracy
test: the delivered geometry is the designated GT itself, so comparing the product
back to that same GT can only measure cartographic self-consistency.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from shapely.geometry import box

from . import common, rasters

REFERENCE_YEAR = 2019
ACQUISITION_DATE_STATUS = 'UNVERIFIED_NO_2019_HR_IMAGE_IN_REGISTERED_QUEUE'


def _gt_source_hashes():
    out = {}
    base = common.GT_SHAPEFILE
    for suffix in ('.shp', '.dbf', '.shx', '.prj', '.sbn', '.sbx'):
        p = base.with_suffix(suffix)
        if p.exists():
            out[p.name] = common.sha256(p)
    return out


def build(state=None, log=print):
    t0 = time.time()
    out = common.REF2019
    out.mkdir(parents=True, exist_ok=True)
    log('reference: reading %s' % common.GT_SHAPEFILE)

    gdf = gpd.read_file(common.GT_SHAPEFILE, engine='pyogrio')
    raw_crs = str(gdf.crs)
    n_raw = len(gdf)
    gdf = gdf.reset_index(drop=True)
    gdf['original_fid'] = np.arange(n_raw, dtype='int64')
    if 'Id' not in gdf.columns:
        gdf['Id'] = -1
    if 'Floor' not in gdf.columns:
        gdf['Floor'] = -1
    # 566 source records carry a NULL shape; they are excluded and counted, never
    # silently converted into a zero-area roof.
    null_mask = gdf.geometry.isna().values
    n_null = int(null_mask.sum())
    gdf = gdf.loc[~null_mask].reset_index(drop=True)
    # geometry hash on the ORIGINAL coordinate reference, before any transform
    gdf['geometry_sha256'] = [common.sha256_bytes(shapely.to_wkb(g, hex=False))
                              for g in gdf.geometry.values]
    log('reference: %d features (%d null shapes excluded), source crs %s'
        % (n_raw, n_null, raw_crs))

    metric = gdf.to_crs(common.METRIC_CRS)
    metric['area_full_m2'] = metric.geometry.area.astype('float64')

    grid10 = rasters.template_grid(common.TEMPLATE, 1)
    grid1 = rasters.template_grid(common.TEMPLATE, common.FINE_SUBDIVISION)
    aoi = rasters.grid_bounds(grid10)

    inside = metric.geometry.intersects(aoi).values
    clipped = shapely.intersection(metric.geometry.values, aoi)
    metric['area_aoi_m2'] = shapely.area(clipped)

    keep = inside & (metric['area_aoi_m2'] > 0)
    log('reference: %d features intersect the frozen AOI (%.2f km2)'
        % (int(keep.sum()), aoi.area / 1e6))

    frame = metric.loc[keep, ['original_fid', 'Id', 'Floor', 'geometry_sha256',
                              'area_full_m2', 'area_aoi_m2']].copy()
    frame['geometry'] = [clipped[i] for i in np.where(keep)[0]]
    frame = gpd.GeoDataFrame(frame, geometry='geometry', crs=common.METRIC_CRS)
    frame['geometry_role'] = common.GT_ROLE
    frame['reference_year'] = REFERENCE_YEAR
    frame['geometry_source'] = 'ORIGINAL_GT_FID'
    frame['alignment_to_designated_reference'] = 'BY_CONSTRUCTION'
    frame['independent_absolute_accuracy'] = 'NOT_MEASURED'
    frame['acquisition_date_status'] = ACQUISITION_DATE_STATUS
    frame['bbox_area_m2'] = (frame.geometry.bounds.maxx - frame.geometry.bounds.minx) * \
                            (frame.geometry.bounds.maxy - frame.geometry.bounds.miny)

    gpkg = out / 'ROOF_GT2019_REFERENCE.gpkg'
    if gpkg.exists():
        gpkg.unlink()
    frame.to_file(gpkg, layer='roof_gt2019_reference', driver='GPKG', engine='pyogrio')
    log('reference: wrote %s (%d features)' % (gpkg.name, len(frame)))

    geoms = list(frame.geometry.values)
    fids = frame['original_fid'].to_numpy(dtype='int64')

    # ---- fine-grid class raster and instance raster (original FID preserved)
    cls = out / 'ROOF_GT2019_REFERENCE.tif'
    inst = out / 'ROOF_GT2019_INSTANCE_ID.tif'
    rasters.banded_rasterize(cls, grid1, geoms, [1] * len(geoms), dtype='uint8', nodata=0)
    rasters.banded_rasterize(inst, grid1, geoms, (fids + 1).tolist(), dtype='uint32', nodata=0)
    log('reference: fine grid rasters written (%d x %d)' % (grid1['width'], grid1['height']))

    # ---- exact 10 m fraction from the same projected vectors
    frac, ident10, assigned, cells = rasters.exact_fraction(geoms, grid10, support=aoi, ids=fids + 1)
    fr_path = out / 'ROOF_GT2019_FRACTION_10M.tif'
    _write_single(fr_path, frac.astype('float32'), grid10, 'roof_gt2019_fraction_full_cell_denominator')
    log('reference: exact fraction integral %.3f m2 over %d cells' % (assigned, cells))

    # ---- area tables
    area_rows = []
    for row in frame.itertuples():
        area_rows.append(dict(
            original_fid=int(row.original_fid), Id=int(row.Id), Floor=int(row.Floor),
            area_full_m2=round(float(row.area_full_m2), 4),
            area_aoi_m2=round(float(row.area_aoi_m2), 4),
            area_bbox_m2=round(float(row.bbox_area_m2), 4),
            geometry_sha256=row.geometry_sha256,
            geometry_role=common.GT_ROLE, reference_year=REFERENCE_YEAR,
            geometry_source='ORIGINAL_GT_FID',
            alignment_to_designated_reference='BY_CONSTRUCTION',
            independent_absolute_accuracy='NOT_MEASURED',
        ))
    common.atomic_write_csv(out / 'GT2019_OBJECT_AREA.csv', area_rows)

    union = shapely.union_all(geoms)
    sum_area = float(frame['area_aoi_m2'].sum())
    union_area = float(union.area)
    clipped_full = float(frame['area_full_m2'].sum())
    summary = dict(
        reference_year=REFERENCE_YEAR,
        acquisition_date_status=ACQUISITION_DATE_STATUS,
        geometry_role=common.GT_ROLE,
        geometry_source='ORIGINAL_GT_FID',
        alignment_to_designated_reference='BY_CONSTRUCTION',
        independent_absolute_accuracy='NOT_MEASURED',
        source_features=n_raw,
        features_intersecting_aoi=int(keep.sum()),
        features_outside_aoi=int(n_raw - int(keep.sum())),
        object_area_sum_m2=round(sum_area, 4),
        union_area_m2=round(union_area, 4),
        overlap_area_m2=round(sum_area - union_area, 4),
        full_gt_area_before_aoi_clip_m2=round(clipped_full, 4),
        aoi_area_m2=round(float(aoi.area), 4),
        within_support_union_m2=round(float(union.intersection(aoi).area), 4),
        unresolved_area_m2=round(float(aoi.area - union.area), 4),
        object_area_sum_ha=round(sum_area / 1e4, 6),
        union_area_ha=round(union_area / 1e4, 6),
        object_area_sum_km2=round(sum_area / 1e6, 9),
        union_area_km2=round(union_area / 1e6, 9),
        fraction_integral_m2=round(assigned, 4),
        fraction_cells_touched=int(cells),
        fraction_cell_m2=abs(grid10['affine'].a * grid10['affine'].e),
        fraction_integral_error_m2=round(assigned - union_area, 6),
        metric_crs=common.METRIC_CRS,
        grid_10m=dict(width=grid10['width'], height=grid10['height'],
                      transform=list(grid10['affine'])[:6], crs=grid10['crs']),
        grid_1m=dict(width=grid1['width'], height=grid1['height'],
                     transform=list(grid1['affine'])[:6], crs=grid1['crs']),
    )
    common.atomic_write_csv(out / 'GT2019_AREA_SUMMARY.csv', [summary])

    # ---- consistency evidence
    consistency = _consistency(common.GT_SHAPEFILE, gdf, frame, grid1, cls, inst, fr_path,
                              union, sum_area, assigned, cells, grid10, n_raw, n_null)
    common.atomic_write_json(out / 'REFERENCE_GEOMETRY_CONSISTENCY.json', consistency)

    manifest = dict(
        product='ROOF_GT2019_REFERENCE',
        product_role='DESIGNATED_GT_REFERENCE_DERIVED_MAPPING',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        source=dict(path=str(common.GT_SHAPEFILE), sha256=_gt_source_hashes(),
                    features=n_raw, null_shape_features=n_null, crs=raw_crs,
                    fields=['Id', 'Floor'],
                    id_field_reusable_as_key=False,
                    id_duplicate_count=int(gdf['Id'].duplicated().sum())),
        reference_year=REFERENCE_YEAR,
        acquisition_date_status=ACQUISITION_DATE_STATUS,
        geometry_role=common.GT_ROLE,
        geometry_source='ORIGINAL_GT_FID',
        alignment_to_designated_reference='BY_CONSTRUCTION',
        independent_absolute_accuracy='NOT_MEASURED',
        gt_free_inference=False,
        model_used=False,
        v2_applied=False,
        rgb_snap_applied=False,
        aspect_filter_applied=False,
        features_written=int(len(frame)),
        features_outside_aoi=int(n_raw - int(keep.sum())),
        files={p.name: dict(sha256=common.sha256(p), bytes=p.stat().st_size)
               for p in sorted(out.iterdir()) if p.is_file() and p.name != 'PRODUCT_MANIFEST.json'},
        summary_area=dict(object_area_sum_m2=summary['object_area_sum_m2'],
                          union_area_m2=summary['union_area_m2'],
                          fraction_integral_m2=summary['fraction_integral_m2']),
        note=('The designated GT was used directly as the map.  Its own polygons cannot '
              'be reused to measure the accuracy of this product; identity between the '
              'vector and the raster is a cartographic consistency check only.'),
        elapsed_seconds=round(time.time() - t0, 1),
    )
    common.atomic_write_json(out / 'PRODUCT_MANIFEST.json', manifest)
    log('reference: complete in %.1f min' % ((time.time() - t0) / 60))
    return manifest


def _write_single(path, array, grid, band_name, nodata=None):
    with rasterio.open(path, 'w', driver='GTiff', height=grid['height'], width=grid['width'],
                       count=1, dtype=array.dtype, crs=grid['crs'], transform=grid['affine'],
                       nodata=nodata, compress='deflate', tiled=True,
                       blockxsize=256, blockysize=256, BIGTIFF='YES') as ds:
        ds.write(array, 1)
        ds.set_band_description(1, band_name)
    with rasterio.open(path) as ds:
        back = ds.read(1)
    if back.shape != array.shape:
        raise RuntimeError('raster write/read shape mismatch for ' + path.name)


def _consistency(shapefile, gdf_raw, frame, grid1, cls_path, inst_path, frac_path,
                 union, sum_area, assigned, cells, grid10, n_raw, n_null):
    """Independent re-read and re-rasterisation of the delivered GT product."""
    from affine import Affine
    from rasterio.features import rasterize as _rz

    with rasterio.open(cls_path) as ds:
        cls_meta = dict(crs=str(ds.crs), transform=list(ds.transform)[:6],
                        width=ds.width, height=ds.height, dtype=str(ds.dtypes[0]),
                        nodata=ds.nodata)
    with rasterio.open(inst_path) as ds:
        inst_meta = dict(crs=str(ds.crs), transform=list(ds.transform)[:6],
                         dtype=str(ds.dtypes[0]), nodata=ds.nodata, width=ds.width,
                         height=ds.height)
    with rasterio.open(frac_path) as ds:
        frac_band = ds.read(1)

    # banded scan: never materialise the 1.2 Gpx class or instance arrays
    class_values = set()
    got_ids = set()
    for r0 in range(0, grid1['height'], 4096):
        r1 = min(grid1['height'], r0 + 4096)
        class_values.update(int(v) for v in rasters.read_band(cls_path, r0, r1).ravel())
        got_ids.update(int(v) for v in np.unique(rasters.read_band(inst_path, r0, r1)).tolist())

    # independent re-rasterisation from the delivered GPKG (top 2048 rows)
    gpkg = gpd.read_file(common.REF2019 / 'ROOF_GT2019_REFERENCE.gpkg', engine='pyogrio')
    a = grid1['affine']
    top = box(a.c, a.f + a.e * 2048, a.c + a.a * grid1['width'], a.f)
    sel = gpkg[gpkg.geometry.intersects(top)]
    ref_arr = (_rz([(g, 1) for g in sel.geometry.values], out_shape=(2048, grid1['width']),
                   transform=a, dtype='uint8', fill=0) if len(sel)
               else np.zeros((2048, grid1['width']), 'uint8'))
    delivered = rasters.read_band(cls_path, 0, 2048)
    mismatch = int((ref_arr != delivered).sum())
    del ref_arr, delivered

    expected_ids = set((frame['original_fid'].to_numpy(dtype='int64') + 1).tolist()) | {0}
    absent = sorted(expected_ids - got_ids)
    extra = sorted(got_ids - expected_ids)
    areas = frame['area_aoi_m2'].to_numpy(dtype='float64')
    sub_pixel = int((areas < 1.0).sum())
    fid_to_area = dict(zip((frame['original_fid'].to_numpy(dtype='int64') + 1).tolist(), areas))

    frac_sum = float(np.asarray(frac_band, dtype='float64').sum()) * abs(
        grid10['affine'].a * grid10['affine'].e)

    # the fraction product is additive per instance, so its integral must agree with
    # the OBJECT area sum, not with the union (which removes the overlap)
    object_sum = round(sum_area, 4)
    diff = frac_sum - object_sum
    tolerance = max(1e-3, abs(object_sum) * 1e-7)
    ok = (mismatch == 0 and not extra and abs(diff) <= tolerance and
          class_values <= {0, 1} and set(absent) <= (expected_ids - {0}))

    return dict(
        schema='XIAN_R5_REFERENCE_GEOMETRY_CONSISTENCY_V1',
        source=dict(path=str(shapefile), sha256=_gt_source_hashes(),
                    features=int(n_raw), null_shape_features=int(n_null),
                    features_with_geometry=int(len(gdf_raw))),
        crs=dict(source=str(gdf_raw.crs), product=common.METRIC_CRS, grid=cls_meta['crs']),
        fid_preservation=dict(
            preserved=True,
            rule='original_fid = source record order; Id is duplicated in the source and is never used as a key',
            id_duplicate_count=int(gdf_raw['Id'].duplicated().sum()),
            fid_min=int(frame['original_fid'].min()), fid_max=int(frame['original_fid'].max()),
            written=int(len(frame)),
            fid_sha256=common.sha256_bytes(','.join(map(str, frame['original_fid'].tolist())).encode()),
        ),
        geometry_hash_rule='sha256 of WKB in the source CRS, before any transform',
        transforms_applied=['EPSG:4326 -> EPSG:32649', 'intersection with frozen 10 m template AOI'],
        transforms_forbidden_and_not_applied=['V2 translation', 'RGB roof snapping',
                                              'aspect-ratio deletion', 'buffering', 'simplification'],
        raster=dict(cls=cls_meta, instance=inst_meta,
                    class_values=sorted(class_values),
                    instance_ids_written=len(expected_ids) - 1,
                    instance_ids_without_any_pixel=len(absent),
                    instance_ids_unexpected=extra[:10],
                    sub_pixel_polygons=sub_pixel,
                    absent_ids_with_area_ge_1m2=len([f for f in absent
                                                     if fid_to_area.get(f, 0) >= 1.0]),
                    instance_id_rule='uint32 = original_fid + 1; 0 = outside the GT set'),
        independent_re_rasterisation=dict(rows=2048, mismatching_pixels=mismatch,
                                          method='rasterio.features.rasterize on the delivered GPKG'),
        area=dict(object_sum_m2=object_sum, union_m2=round(union.area, 4),
                  overlap_m2=round(object_sum - union.area, 4),
                  fraction_integral_m2=round(assigned, 4),
                  fraction_integral_from_raster_m2=round(frac_sum, 4),
                  difference_vs_object_sum_m2=round(diff, 6),
                  tolerance_m2=round(tolerance, 6),
                  cells_touched=int(cells)),
        background_semantics=dict(
            zero_means='outside the designated GT reference set inside the frozen AOI',
            zero_is_not='certified reliable background',
        ),
        status='PASS' if ok else 'REVIEW',
        status_reason=('fraction integral equals the additive object area; the union is smaller '
                       'by the overlap, which is reported separately')
        if ok else 'one or more checks outside tolerance',
    )
