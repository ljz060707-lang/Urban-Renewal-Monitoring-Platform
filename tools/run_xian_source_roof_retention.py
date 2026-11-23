"""Commit source-only semantic retention before any manual-roof shape comparison."""
from pathlib import Path
import argparse,json,sys
import numpy as np,fiona
from affine import Affine
from pyproj import Transformer
from shapely.geometry import shape,Polygon,box,mapping
from shapely.ops import transform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.source_mask_retention import retain_source_masks
from xian_r5.roof_reference_candidates import roof_reference_candidates,RoofMatchPolicy
from xian_r5.fraction_shadow import _areas_and_counts
from xian_r5.label_transfer import Grid
from xian_r5.alignment.source_template_alignment import _sample_target


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);out=parser.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment');prior=parent/'SINGLE2019_SAM_ROOF_PILOT_20261012_16'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh source-retention output required')
    plan0=json.loads((prior/'FROZEN_SAM_VECTOR_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((prior/'SAM_VECTOR_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(plan0['protected_sha256']);protected.update(plan0['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(prior/'SAM_VECTOR_RECEIPT.json')]=sha(prior/'SAM_VECTOR_RECEIPT.json')
    annex=json.loads((prior/'SOURCE_MASK_PROGRESS_ANNEX_RECEIPT.json').read_text(encoding='utf-8'))
    protected.update({str(prior/n):h for n,h in annex['new_annex_sha256'].items()})
    protected[str(prior/'SOURCE_MASK_PROGRESS_ANNEX_RECEIPT.json')]=sha(prior/'SOURCE_MASK_PROGRESS_ANNEX_RECEIPT.json')
    choices_path=prior/'FROZEN_SOURCE_MASK_RETENTION_CHOICES.json';protected[str(choices_path)]=sha(choices_path)
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected source drift: '+path)
    policy=RoofMatchPolicy();code=[ROOT/'src/xian_r5/source_mask_retention.py',ROOT/'tests/test_xian_source_mask_retention.py',Path(__file__).resolve()]
    out.mkdir();save_json(out/'FROZEN_RETENTION_AND_SHAPE_PLAN.json',{'run':out.name,
        'selection':'FROZEN_SOURCE_VISUAL_CHOICES_NOT_SAM_SCORE_OR_GT_MATCH_OUTCOME',
        'source_geometry_positions_frozen_before_reference_geometry_load':True,
        'shape_roles':'VISIBLE_EXTENT_CANDIDATE_ONLY; exclude surfaces, partials, risks, unresolved',
        'shape_comparison_policy':policy.__dict__,'reference_to_Sentinel_binding':'NOT_ESTABLISHED',
        'source_observed_date':'2018-11-01','experiment_year':2019,
        'candidate_is_not_complete_physical_roof_or_GT':True,'new_registration_estimation_or_application':False,
        'protected_sha256':protected,'new_code_sha256':{str(f):sha(f) for f in code}})
    candidates=json.loads((prior/'SAM_SOURCE_GEOMETRY_CANDIDATES.json').read_text(encoding='utf-8'))['records']
    choices=json.loads(choices_path.read_text(encoding='utf-8'))['choices']
    retained=retain_source_masks(candidates,choices)
    save_json(out/'SOURCE_ROOF_RETENTION.json',{k:v for k,v in retained.items() if k!='geometries'})
    schema={'geometry':'MultiPolygon','properties':{'candidate_id':'str','source_role':'str','shape_eligible':'int',
        'physical_id':'str','formal_gt':'int','complete_verified':'int'}}
    with fiona.open(out/'RETAINED_SOURCE_ROOF_CANDIDATES.gpkg','w',driver='GPKG',layer='source_only_retained',crs='EPSG:32649',schema=schema) as layer:
        for row in retained['records']:
            key=row['retained_candidate_id']
            if key is None:continue
            layer.write({'geometry':mapping(retained['geometries'][key]),'properties':{'candidate_id':key,
                'source_role':row['source_role'],'shape_eligible':int(row['shape_diagnostic_eligible']),
                'physical_id':None,'formal_gt':0,'complete_verified':0}})
    with fiona.open(out/'RETAINED_SOURCE_ROOF_CANDIDATES.gpkg') as layer:
        if len(layer)!=retained['retained_masks']:raise RuntimeError('Retained candidate count drift')
        for f in layer:
            if not shape(f.geometry).equals_exact(retained['geometries'][f.properties['candidate_id']],0):
                raise RuntimeError('Retained geometry changed on disk')
    # Only after the semantic choices and their unchanged geometries are saved,
    # load reference geometry. The reference cannot alter a retained candidate.
    eligible={r['retained_candidate_id']:retained['geometries'][r['retained_candidate_id']] for r in retained['records'] if r['shape_diagnostic_eligible']}
    field=json.loads((parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))['cells']
    regions={c['region_id']:shape(c['support_geometry']) for c in field}
    inventory=json.loads((parent/'GT10M_VALIDATION_PREFLIGHT_20261012_05/REFERENCE_EVIDENCE_CENSUS.json').read_text(encoding='utf-8'))
    fids={r['source_fid'] for r in inventory['records']};refs={}
    reference=Path('F:/xian/西安建筑轮廓数据/西安.shp');to_ll=Transformer.from_crs(32649,4326,always_xy=True);to_metric=Transformer.from_crs(4326,32649,always_xy=True)
    corners=[(308460,3792190),(311020,3792190),(311020,3794750),(308460,3794750)]
    ring=[to_ll.transform(a[0]+(b[0]-a[0])*n/64,a[1]+(b[1]-a[1])*n/64)
        for a,b in zip(corners,corners[1:]+corners[:1]) for n in range(64)]
    with fiona.open(reference) as layer:
        if layer.crs.to_epsg()!=4326:raise ValueError('Reference frame changed')
        for f in layer.filter(bbox=Polygon(ring).bounds):
            if str(f.id) in fids:refs[str(f.id)]=transform(to_metric.transform,shape(f.geometry))
    if set(refs)!=fids:raise ValueError('Frozen reference ID inventory differs')
    compared=roof_reference_candidates(eligible,refs,regions,policy=policy)
    save_json(out/'ROOF_REFERENCE_SHAPE_DIAGNOSTIC.json',compared)
    save_json(out/'REFERENCE_DEVELOPMENT_EXPOSURE.json',{'source':str(reference),'sha256':sha(reference),
        'source_fids_loaded':sorted(refs),'all_fids_already_development_exposed_in_stage08':True,
        'source_masks_selected_before_this_reference_load':True,'external_unseen_validation':False})
    grid=Grid(Affine(10,0,308460,0,-10,3794750),256,256,'EPSG:32649')
    cells=np.array([grid.cell(c,r) for r in range(256) for c in range(256)],dtype=object)
    area,count,conflict=_areas_and_counts(eligible.values(),cells)
    sample=(area.reshape(256,256)/100>=.8)&~conflict.reshape(256,256)
    yy,xx=np.indices((256,256));fit=(yy//16+xx//16)%2==0;population=[]
    for c in field:
        rr,cc=(int(v[1:]) for v in c['region_id'].split('_')[1:]);scope=np.zeros((256,256),bool)
        scope[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True;numbers={}
        for name,fold in [('FIT',fit),('CHECK',~fit)]:
            common=sample&scope&fold;target=scope&fold
            for dx in range(-3,4):
                for dy in range(-3,4):common&=_sample_target(target[None],dx,dy)[0]
            numbers[name]=int(common.sum())
        population.append({'region_id':c['region_id'],'source_candidate_roof_sample_pixels':int((sample&scope).sum()),
            'fixed_fit_pixels':numbers['FIT'],'fixed_check_pixels':numbers['CHECK'],
            'can_run_fixed_checker':numbers['FIT']>=100 and numbers['CHECK']>=100,'alignment_fit_run':False})
    save_json(out/'RETAINED_ROOF_POPULATION_PREFLIGHT.json',{'records':population,'threshold':.8,
        'minimum_fit':100,'minimum_check':100,'radius_px':3,'no_target_RGB_or_predictions_read':True})
    summary={'run':out.name,'source_groups_disposed':len(retained['records']),'source_masks_retained':retained['retained_masks'],
        'source_role_counts':retained['role_counts'],'shape_comparison_source_candidates':len(eligible),
        'reference_roofs_loaded':len(refs),'area_compatible_shape_pairs':compared['candidate_pair_count'],
        'mutual_unique_shape_hypotheses':len(compared['hypotheses']),
        'roof_to_manual_reference_centroid_difference_is_registration_vector':False,
        'source_positions_or_original_masks_changed':0,'cells_can_run_fixed_checker':sum(r['can_run_fixed_checker'] for r in population),
        'new_registration_vectors_or_applications':0,'new_training_labels':0,'formal_gt_eligible':0,
        'clean_physical_roof_instances_verified':0,'physical_ids':0,'ground_footprints':0,
        'reference_to_Sentinel_binding':'NOT_ESTABLISHED','absolute_position_accuracy':'UNKNOWN'}
    save_json(out/'SOURCE_ROOF_RETENTION_SUMMARY.json',summary)
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path)
    save_json(out/'SOURCE_ROOF_RETENTION_RECEIPT.json',{'summary':summary,'new_code_sha256':{str(f):sha(f) for f in code},
        'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()},'whole_window_governance_completed':False})
    print(json.dumps(summary))


if __name__=='__main__':main()
