"""Read-back QA of a completed fraction shadow; no calibration/GT certification."""
from pathlib import Path
import argparse,json,zipfile,hashlib,sys
import numpy as np
import rasterio,fiona,shapely
from shapely.geometry import shape,box
from shapely.affinity import translate
from run_xian_source_partition_alignment import sha,save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    p=parser.parse_args().run.resolve()
    expected_parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment').resolve()
    if p.parent!=expected_parent or (p/'INDEPENDENT_ENGINEERING_READBACK_QA.json').exists():
        raise ValueError('Existing isolated shadow required; append-only QA')
    plan=json.loads((p/'FROZEN_FRACTION_PLAN.json').read_text(encoding='utf-8'))
    summary=json.loads((p/'FRACTION_DIAGNOSTICS.json').read_text(encoding='utf-8'))
    records={};arrays={}
    for f in sorted(p.glob('*.tif')):
        with rasterio.open(f) as s:
            assert s.shape==(256,256) and str(s.crs)==plan['crs']
            assert list(s.transform)[:6]==plan['grid_affine']
            assert list(s.transform.to_gdal())==[308460,10,0,3794750,0,-10]
            assert s.tags()['TRAINING_USE_PROHIBITED']=='YES'
            assert s.tags()['FORMAL_CORRECTION_ACCEPTED']=='NO'
            arrays[f.name]=s.read()
            records[f.name]={'bands':s.count,'shape':list(s.shape),'dtype':list(s.dtypes),
                'nodata':s.nodata,'grid_tags_verified':True}
    assert len(arrays)==10
    raw=arrays['ROOF_FRACTION_RAW.tif'][0].astype(np.float64)
    candidate=arrays['ROOF_FRACTION_CORRECTED_CANDIDATE.tif'][0].astype(np.float64)
    diff=arrays['FRACTION_DIFFERENCE.tif'][0].astype(np.float64)
    scope=arrays['SOURCE_SCOPE_FRACTION.tif'];quality=arrays['QUALITY.tif']
    assert np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all()
    available=candidate!=-9999
    assert ((candidate[available]>=0)&(candidate[available]<=1)).all()
    assert (candidate[available]<=scope[1,available]+1e-6).all()
    assert np.all(diff[~available]==-9999)
    assert int((~available).sum())==summary['candidate_unknown_pixels']
    full=available&(scope[0]>=1-1e-6)&(scope[1]>=1-1e-6)
    core=full&~quality[5].astype(bool)&~quality[6].astype(bool)&~quality[7].astype(bool)
    assert int(full.sum())==summary['full_source_scope']['denominator']
    assert int(core.sum())==summary['excluding_boundary_seam_conflict']['denominator']
    difference_error=float(np.max(np.abs(diff[available]-(candidate-raw)[available])))
    assert difference_error<=1e-6
    assert not quality[3].any() and not quality[4].any() and not quality[9].any()
    assert not (quality[0]==0).any()
    assert not arrays['TRAINING_WEIGHT_ZERO.tif'].any()
    probe=np.full(raw.shape,255,np.uint8)
    probe[full]=(raw[full]>.43).astype(np.uint8)+2*(candidate[full]>.43).astype(np.uint8)
    assert np.array_equal(probe,arrays['THRESHOLD_043_FLIP_PROBE.tif'][0])
    flips=full&np.isin(probe,[1,2])
    assert int(flips.sum())==summary['full_source_scope']['strict_0_43_flip_pixels']
    assert int((flips&core).sum())==summary['excluding_boundary_seam_conflict']['strict_0_43_flip_pixels']
    binary_member=plan['old_binary_member']
    from rasterio.io import MemoryFile
    with zipfile.ZipFile('F:/xian/10m结果.zip') as z:
        names=[n for n in z.namelist() if n.endswith('/2019_pseudo_label_R90_tau43.tif')]
        assert len(names)==1
        data=z.read(names[0]);assert hashlib.sha256(data).hexdigest()==binary_member['sha256']
    with MemoryFile(data) as m,m.open() as s:old=s.read(1)
    mismatch=arrays['OLD_BINARY_COMPARISON_DIAGNOSTIC.tif']
    assert np.array_equal(mismatch[0],((raw>.43)!=old).astype(np.uint8))
    assert np.array_equal(mismatch[1,full],((candidate[full]>.43)!=old[full]).astype(np.uint8))
    assert (mismatch[1,~full]==255).all()
    components=[]
    with fiona.open(p/'SOURCE_MASK_COMPONENTS_SHADOW.gpkg') as s:
        assert s.crs.to_epsg()==32649
        for f in s:
            assert f.properties['physical_id_status']=='NOT_ESTABLISHED'
            components.append(shape(f.geometry))
    assert len(components)==summary['native_component_count']==1662
    roi=box(*plan['roi_bounds'])
    expected_raw=shapely.union_all(components).intersection(roi).area
    field=json.loads((p.parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))
    candidate_pieces=[];shifted_scope=np.zeros(raw.shape,bool)
    for cell in field['cells']:
        if cell['candidate_correction_m'] is None:continue
        support=shape(cell['support_geometry']);dx,dy=cell['candidate_correction_m']
        valid=translate(support,xoff=dx,yoff=dy).intersection(support).intersection(roi)
        for g in components:
            fragment=g.intersection(support)
            if fragment.area<=1e-6:continue
            moved=translate(fragment,xoff=dx,yoff=dy).intersection(valid)
            if moved.area>1e-6:candidate_pieces.append(moved)
        if abs(dx)+abs(dy)>0:
            rr,cc=(int(v[1:]) for v in cell['region_id'].split('_')[1:])
            shifted_scope[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True
    expected_candidate=shapely.union_all(candidate_pieces).area
    exported_area_errors={'raw_m2':abs(raw.sum()*100-expected_raw),
        'candidate_m2':abs(candidate[available].sum()*100-expected_candidate)}
    assert max(exported_area_errors.values())<.05
    groups={}
    for name,mask in [('nonzero_shift_only',full&shifted_scope),
                      ('nonzero_shift_core_only',core&shifted_scope),('zero_shift_only',full&~shifted_scope)]:
        delta=np.abs(diff[mask])
        groups[name]={'denominator':int(mask.sum()),'strict_043_flips':int((flips&mask).sum()),
            'numerically_unchanged':int((delta<=1e-6).sum()),
            'abs_difference_le_010':int((delta<=.10).sum()),
            'abs_difference_gt_010':int((delta>.10).sum()),
            'mean_abs_difference':float(delta.mean()) if len(delta) else None}
    for path,expected in {**plan['protected_sha256'],**plan['source_hashes']}.items():
        assert sha(path)==expected,path
    result={'stage':'INDEPENDENT_ENGINEERING_READBACK_NOT_SPATIAL_ACCURACY_VALIDATION',
        'raster_records':records,'raster_count':len(arrays),'component_count':len(components),
        'threshold_probe_recomputation_mismatches':0,'difference_recomputation_max_error':difference_error,
        'exported_float32_area_error_m2':exported_area_errors,'float32_serialization_area_tolerance_m2':.05,
        'affine_to_gdal_order_verified':True,'full_comparison_pixels':int(full.sum()),
        'core_comparison_pixels':int(core.sum()),'separated_shift_cohorts':groups,
        'protected_hashes_verified':len(plan['protected_sha256'])+len(plan['source_hashes']),
        'training_eligible_pixels':0,'training_weights_all_zero':True,
        'independent_ground_alignment_or_timing_validation_performed':False,
        'zero_component_count_is_not_verified_background':True}
    save_json(p/'INDEPENDENT_ENGINEERING_READBACK_QA.json',result)
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':main()
