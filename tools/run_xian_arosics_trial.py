"""Isolated AROSICS: synthetic calibration before bounded real diagnostics.

Bad-data masks reject tie centers, not all contaminated pixels in FFT windows.
Real windows in this trial are therefore diagnostics, NEVER ground-only ties.
"""
import argparse
import hashlib
import inspect
import json
import sys
import traceback
import time
from pathlib import Path
import numpy as np
from osgeo import osr, gdal
from scipy.ndimage import shift, gaussian_filter
from geoarray import GeoArray
import arosics
from arosics import COREG, COREG_LOCAL
from arosics.Tie_Point_Grid import Tie_Point_Grid

OUT=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment/XIAN_R5_ALIGNMENT_FEATURE_RECOVERY_20261013')
GT=(308460,10,0,3794750,0,-10)
SRS=osr.SpatialReference(); SRS.ImportFromEPSG(32649); WKT=SRS.ExportToWkt()


def save(name,data):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)


def scalar(v):
    return None if v is None or not np.isfinite(v) else float(v)


def result(cr):
    return dict(success=bool(cr.success), x_shift_px=scalar(cr.x_shift_px), y_shift_px=scalar(cr.y_shift_px),
        dx_map_m=scalar(cr.x_shift_map), dy_map_m=scalar(cr.y_shift_map),
        reliability_percent=scalar(cr.shift_reliability), ssim_original=scalar(cr.ssim_orig),
        ssim_deshifted=scalar(cr.ssim_deshifted), errors=[str(e) for e in cr.tracked_errors])


def synthetic():
    names=['SYNTHETIC_forward_DESHIFTED.tif','SYNTHETIC_reverse_DESHIFTED.tif',
           'AROSICS_SYNTHETIC_TIE_POINT_GRID.csv','AROSICS_SYNTHETIC_TEST.json']
    if any((OUT/name).exists() for name in names):raise ValueError('Fresh append-only synthetic outputs required')
    rng=np.random.default_rng(130)
    ref=(50+gaussian_filter(rng.normal(0,30,(256,256)),1.2)).astype('float32')
    moving=shift(ref,(-.75,1.25),order=3,mode='constant',cval=0)
    a,b=GeoArray(ref,geotransform=GT,projection=WKT,nodata=0),GeoArray(moving,geotransform=GT,projection=WKT,nodata=0)
    rows=[]
    for label,left,right,expected in [('forward',a,b,(-12.5,-7.5)),('reverse',b,a,(12.5,7.5))]:
        cr=COREG(left,right,ws=(128,128),max_shift=3,CPUs=1,q=True,
                 path_out=str(OUT/('SYNTHETIC_'+label+'_DESHIFTED.tif')),fmt_out='GTIFF')
        cr.calculate_spatial_shifts(); row=result(cr)
        row.update(test=label,expected_correction_m=list(expected))
        error=float(np.linalg.norm(np.array([cr.x_shift_map,cr.y_shift_map])-expected))
        row['vector_error_m']=error
        row['two_metre_numeric_calibration_pass']=bool(cr.success and error<=2)
        row['direction_pass']=bool(cr.success and np.sign(cr.x_shift_map)==np.sign(expected[0])
                                   and np.sign(cr.y_shift_map)==np.sign(expected[1]))
        if not row['direction_pass']:
            raise AssertionError('Synthetic direction calibration failed: '+str(row))
        cr.correct_shifts()
        ds=gdal.Open(str(OUT/('SYNTHETIC_'+label+'_DESHIFTED.tif')))
        row['output_geotransform']=list(ds.GetGeoTransform()); row['output_epsg']=osr.SpatialReference(wkt=ds.GetProjection()).GetAuthorityCode(None)
        row['output_valid_pixels']=int(np.count_nonzero(ds.GetRasterBand(1).ReadAsArray()!=0)); ds=None
        if row['output_epsg']!='32649' or row['output_valid_pixels']<=0:
            raise AssertionError('Synthetic CRS/valid-area failure')
        rows.append(row)
    bad=np.zeros((256,256),bool);bad[80:145,80:145]=True
    local=COREG_LOCAL(a,b,grid_res=48,window_size=(64,64),max_shift=3,CPUs=1,q=True,
          progress=False,mask_baddata_ref=bad,mask_baddata_tgt=bad,max_points=16,
          tieP_filter_level=3,min_reliability=60,tieP_random_state=130,
          rs_random_state=130,path_out=str(OUT/'SYNTHETIC_LOCAL_NOT_APPLIED.tif'),fmt_out='GTIFF')
    local.calculate_spatial_shifts()
    table=local.CoRegPoints_table
    table.drop(columns='geometry',errors='ignore').to_csv(OUT/'AROSICS_SYNTHETIC_TIE_POINT_GRID.csv',index=False)
    within_mask=[]
    for _,r in table.iterrows():
        c=int((r.X_MAP-GT[0])/10);y=int((GT[3]-r.Y_MAP)/10)
        if 0<=c<256 and 0<=y<256 and bad[y,c]:within_mask.append(int(r.POINT_ID))
    if within_mask:raise AssertionError('Masked centers entered tie grid')
    inspected={name:dict(path=inspect.getfile(obj),sha256=hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest())
          for name,obj in [('COREG',COREG),('COREG_LOCAL',COREG_LOCAL),('Tie_Point_Grid',Tie_Point_Grid)]}
    save('AROSICS_SYNTHETIC_TEST.json',dict(status='RUNTIME_DIRECTION_PASS_PRECISION_LIMIT_RECORDED',arosics_version=arosics.__version__,
          runtime_direction_gate_pass=True, two_metre_numeric_calibration_pass=all(r['two_metre_numeric_calibration_pass'] for r in rows),
          python=sys.version, gdal=gdal.VersionInfo(), rows=rows,seed=130,
          subpixel_shift_not_rounded=True, crs='EPSG:32649', tie_grid_rows=len(table),
          masked_centers_in_table=len(within_mask),filter_level=3,tie_columns=list(table.columns),
          local_warp_not_applied=True, source_inspection=inspected,
          badmask_semantics='CENTER_EXCLUSION_NOT_CORRELATION_WINDOW_PIXEL_EXCLUSION'))
    print('Synthetic runtime/sign completed; numeric precision separately recorded',rows,flush=True)


def real():
    if (OUT/'AROSICS_REAL_CROSS_RESOLUTION_TRIAL.json').exists():
        raise ValueError('Fresh append-only real trial required')
    syn=json.loads((OUT/'AROSICS_SYNTHETIC_TEST.json').read_text(encoding='utf-8'))
    if not syn['runtime_direction_gate_pass']:raise ValueError('Synthetic direction/runtime gate not passed')
    z=np.load(OUT/'MASK_DIAGNOSTIC_DATA.npz')
    # AROSICS supports cross-GSD images; use genuinely different 1m/10m grids.
    # 1m is only source display resampling; native GE resolution ledger retained.
    native_ds=gdal.Open(str(OUT/'GE_NATIVE_CONTEXT_DISPLAY_1M.tif'))
    high=native_ds.ReadAsArray().transpose(1,2,0)
    gt_high=native_ds.GetGeoTransform();native_ds=None
    ref=(z['ref'].astype('float32')*np.array([.299,.587,.114])).sum(2).astype('float32')
    moving=(high.astype('float32')*np.array([.299,.587,.114])).sum(2).astype('float32')
    a=GeoArray(ref,geotransform=GT,projection=WKT,nodata=0)
    b=GeoArray(moving,geotransform=gt_high,projection=WKT,nodata=0)
    plan=json.loads((OUT/'FROZEN_GROUND_FEATURE_PLAN.json').read_text(encoding='utf-8'))
    rows=[]
    for f in plan['features']:
        c,r=f['patch_anchor_col_row']
        center=(GT[0]+(c+.5)*10,GT[3]-(r+.5)*10)
        try:
            cr=COREG(a,b,wp=center,ws=(24,24),max_shift=3,CPUs=1,q=True,
                     mask_baddata_ref=z['touched'], mask_baddata_tgt=np.repeat(np.repeat(z['touched'],10,0),10,1))
            cr.calculate_spatial_shifts(); row=result(cr)
            row['effective_window_dims_px']=list(cr.matchBox.imDimsYX) if cr.matchBox is not None else None
        except Exception as e:
            row=dict(success=False,errors=[type(e).__name__+': '+str(e)])
        row.update(feature_id=f['feature_id'], source_region=f['source_region'],
                   algorithm='AROSICS_COREG_CROSS_GSD', evidence_role='CONTAMINATED_WINDOW_FEASIBILITY_DIAGNOSTIC',
                   ground_only_verified=False,accepted=False,applied=False,
                   reason='MASK_ONLY_EXCLUDES_CENTER; WINDOW_CONTAINS_ROOFS_TREES_CARS_OR_SHADOWS')
        rows.append(row); print(f['feature_id'],row,flush=True)
    save('AROSICS_REAL_CROSS_RESOLUTION_TRIAL.json',dict(rows=rows,region_denominator=16,
         native_reference_m=10,source_input_display_m=1, native_source_m='VARIABLE_GE_NATIVE_DEGREES',
         outputs_not_localization_bounds=True, COREG_LOCAL_real_not_run_reason=
         'POINT_GRID_CENTERS_CANNOT_ENFORCE_GROUND_ONLY_MATCH_WINDOWS',
         synthetic_numeric_precision_pass=syn['two_metre_numeric_calibration_pass'],
         qualified_regions=0, applied_regions=0, absolute_position_accuracy='UNKNOWN'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['synthetic','real']);args=p.parse_args()
    try:globals()[args.stage]()
    except Exception:
        text=traceback.format_exc();print(text,flush=True)
        with (OUT/('AROSICS_'+args.stage.upper()+'_FAILURE_'+str(time.time_ns())+'.txt')).open('x',encoding='utf-8') as f:f.write(text)
        raise
