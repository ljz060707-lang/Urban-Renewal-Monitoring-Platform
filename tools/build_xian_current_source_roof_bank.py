"""Versioned source-roof bank, retaining old attempts and uncertain roles."""
from pathlib import Path
import argparse,json,sys
import numpy as np,fiona
from affine import Affine
from pyproj import Transformer
from shapely.geometry import shape,mapping,Polygon
from shapely.ops import transform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.source_mask_retention import retain_source_masks
from xian_r5.roof_group_correspondence import roof_group_correspondence
from xian_r5.roof_reference_candidates import roof_reference_candidates
from xian_r5.fraction_shadow import _areas_and_counts
from xian_r5.label_transfer import Grid
from xian_r5.alignment.source_template_alignment import _sample_target


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);out=parser.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    refined=parent/'SINGLE2019_SAM_SOURCE_REFINE_20261012_18';base=parent/'SINGLE2019_SAM_ROOF_PILOT_20261012_16'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh current source-roof bank required')
    plan0=json.loads((refined/'FROZEN_REFINEMENT_VECTOR_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((refined/'SAM_REFINEMENT_VECTOR_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(plan0['protected_sha256']);protected.update(plan0['new_code_sha256'])
    protected.update({str(refined/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(refined/'SAM_REFINEMENT_VECTOR_RECEIPT.json')]=sha(refined/'SAM_REFINEMENT_VECTOR_RECEIPT.json')
    disposition_path=refined/'FROZEN_REFINED_SOURCE_DISPOSITION.json';protected[str(disposition_path)]=sha(disposition_path)
    groups_root=parent/'SINGLE2019_SOURCE_ROOF_RETENTION_20261012_17'
    group_receipt=json.loads((groups_root/'ROOF_REFERENCE_GROUP_RECEIPT.json').read_text(encoding='utf-8'))
    protected.update(group_receipt['new_code_sha256']);protected.update({str(groups_root/n):h for n,h in group_receipt['output_sha256'].items()})
    protected[str(groups_root/'ROOF_REFERENCE_GROUP_RECEIPT.json')]=sha(groups_root/'ROOF_REFERENCE_GROUP_RECEIPT.json')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen source candidate drift: '+path)
    dispositions=json.loads(disposition_path.read_text(encoding='utf-8'))['records']
    choices=json.loads((base/'FROZEN_SOURCE_MASK_RETENTION_CHOICES.json').read_text(encoding='utf-8'))['choices']
    candidates=json.loads((base/'SAM_SOURCE_GEOMETRY_CANDIDATES.json').read_text(encoding='utf-8'))['records']
    new=json.loads((refined/'SAM_REFINED_SOURCE_GEOMETRY_CANDIDATES.json').read_text(encoding='utf-8'))['records'];lineage=[]
    for d in dispositions:
        if d['source_role']=='UNRESOLVED':continue
        if d['reference_or_target_position_used'] or d['roof_purity_independently_verified']:raise ValueError('Source-only candidate role breach')
        old=d['base_group_id'];new_group=d['refined_group_id']
        if len([c for c in choices if c['group_id']==old])!=1:raise ValueError('Base group ambiguity')
        choices=[c for c in choices if c['group_id']!=old]+[{'group_id':new_group,'mask_index':d['retained_mask_index'],
            'source_role':d['source_role'],'source_visual_basis':d['source_visual_basis'],'reference_or_target_position_used':False}]
        candidates=[c for c in candidates if c['group_id']!=old]+[c for c in new if c['group_id']==new_group]
        lineage.append({'base_source_group':old,'active_source_group':new_group,
            'source_only_reason':d['source_visual_basis'],'old_candidates_and_retention_files_preserved':True})
    code=[Path(__file__).resolve()]
    out.mkdir();save_json(out/'FROZEN_CURRENT_ROOF_BANK_PLAN.json',{'scope':'CURRENT_256x256_NOMINAL_2019',
        'source_choices_frozen_before_reference_load':choices,'version_lineage':lineage,
        'old_masks_vectors_and_deliveries_unchanged':True,'formal_ground_accuracy':False,
        'protected_sha256':protected,'new_code_sha256':{str(f):sha(f) for f in code}})
    bank=retain_source_masks(candidates,choices)
    save_json(out/'CURRENT_SOURCE_ROOF_BANK.json',{k:v for k,v in bank.items() if k!='geometries'})
    schema={'geometry':'MultiPolygon','properties':{'candidate_id':'str','source_role':'str','shape_eligible':'int','physical_id':'str','formal_gt':'int'}}
    with fiona.open(out/'CURRENT_SOURCE_ROOF_BANK.gpkg','w',driver='GPKG',layer='retained_source_model_candidates',crs='EPSG:32649',schema=schema) as s:
        for r in bank['records']:
            key=r['retained_candidate_id']
            if key is not None:s.write({'geometry':mapping(bank['geometries'][key]),'properties':{'candidate_id':key,
                'source_role':r['source_role'],'shape_eligible':int(r['shape_diagnostic_eligible']),'physical_id':None,'formal_gt':0}})
    with fiona.open(out/'CURRENT_SOURCE_ROOF_BANK.gpkg') as s:
        if len(s)!=bank['retained_masks']:raise RuntimeError('Bank vector count changed')
        for f in s:
            if not shape(f.geometry).equals_exact(bank['geometries'][f.properties['candidate_id']],0):raise RuntimeError('Bank source geometry changed')
    eligible={r['retained_candidate_id']:bank['geometries'][r['retained_candidate_id']] for r in bank['records'] if r['shape_diagnostic_eligible']}
    field=json.loads((parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))['cells']
    regions={c['region_id']:shape(c['support_geometry']) for c in field}
    fids=set(json.loads((groups_root/'REFERENCE_DEVELOPMENT_EXPOSURE.json').read_text(encoding='utf-8'))['source_fids_loaded'])
    ll=Transformer.from_crs(32649,4326,always_xy=True);metric=Transformer.from_crs(4326,32649,always_xy=True)
    corners=[(308460,3792190),(311020,3792190),(311020,3794750),(308460,3794750)]
    ring=[ll.transform(a[0]+(b[0]-a[0])*n/64,a[1]+(b[1]-a[1])*n/64) for a,b in zip(corners,corners[1:]+corners[:1]) for n in range(64)]
    refs={}
    with fiona.open('F:/xian/西安建筑轮廓数据/西安.shp') as s:
        for f in s.filter(bbox=Polygon(ring).bounds):
            if str(f.id) in fids:refs[str(f.id)]=transform(metric.transform,shape(f.geometry))
    if set(refs)!=fids:raise ValueError('Reference ID exposure drift')
    single=roof_reference_candidates(eligible,refs,regions);grouped=roof_group_correspondence(eligible,refs,regions)
    save_json(out/'CURRENT_SINGLE_REFERENCE_SHAPE_DIAGNOSTIC.json',single)
    save_json(out/'CURRENT_GROUP_REFERENCE_SHAPE_DIAGNOSTIC.json',grouped)
    grid=Grid(Affine(10,0,308460,0,-10,3794750),256,256,'EPSG:32649')
    pixel_cells=np.array([grid.cell(c,r) for r in range(256) for c in range(256)],dtype=object)
    area,count,conflict=_areas_and_counts(eligible.values(),pixel_cells)
    mask=(area.reshape(256,256)/100>=.8)&~conflict.reshape(256,256)
    yy,xx=np.indices((256,256));fit=(yy//16+xx//16)%2==0;pop=[]
    for cell in field:
        rr,cc=(int(v[1:]) for v in cell['region_id'].split('_')[1:]);scope=np.zeros((256,256),bool)
        scope[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True;numbers=[]
        for fold in (fit,~fit):
            common=mask&scope&fold;target=scope&fold
            for dx in range(-3,4):
                for dy in range(-3,4):common&=_sample_target(target[None],dx,dy)[0]
            numbers.append(int(common.sum()))
        pop.append({'region_id':cell['region_id'],'candidate_roof_pixels_80percent':int((mask&scope).sum()),
            'fixed_fit_pixels':numbers[0],'fixed_check_pixels':numbers[1],
            'checker_population_sufficient':min(numbers)>=100,'alignment_fit_run':False})
    save_json(out/'CURRENT_ROOF_POPULATION_PREFLIGHT.json',{'records':pop,'minimum_fit':100,'minimum_check':100,
        'radius_px':3,'new_registration_vectors_or_apps':0})
    summary={'source_groups_disposed':len(bank['records']),'retained_source_masks':bank['retained_masks'],
        'role_counts':bank['role_counts'],'shape_source_candidates':len(eligible),
        'single_reference_hypotheses':len(single['hypotheses']),'group_reference_hypotheses':grouped['hypotheses'],
        'stable_independent_physical_pairs_verified':0,'source_roof_pixels_80percent':int(mask.sum()),
        'cells_with_sufficient_fixed_population':sum(r['checker_population_sufficient'] for r in pop),
        'old_vectors_or_masks_changed':0,'new_registration_vectors_or_apps':0,'new_training_labels':0,
        'formal_gt':0,'reference_to_Sentinel_binding':'NOT_ESTABLISHED','absolute_position_accuracy':'UNKNOWN'}
    save_json(out/'CURRENT_SOURCE_ROOF_BANK_SUMMARY.json',summary)
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path)
    save_json(out/'CURRENT_SOURCE_ROOF_BANK_RECEIPT.json',{'summary':summary,'new_code_sha256':{str(f):sha(f) for f in code},
        'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()},'whole_window_governance_completed':False})
    print(json.dumps(summary))


if __name__=='__main__':main()
