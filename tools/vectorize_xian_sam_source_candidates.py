"""Preserve unselected source SAM alternatives, with lossless pixel provenance."""
from pathlib import Path
import argparse,json,sys
import fiona,numpy as np
from shapely.geometry import shape,box,mapping
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.sam_source_candidates import sam_source_candidate


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);root=p.parse_args().run.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if root.parent!=parent.resolve() or (root/'SAM_VECTOR_RECEIPT.json').exists():raise ValueError('Append-only completed SAM pilot required')
    plan=json.loads((root/'FROZEN_SAM_PILOT_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((root/'SAM_INFERENCE_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(plan['protected_sha256']);protected.update(plan['new_code_sha256'])
    protected.update({str(root/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(root/'SAM_INFERENCE_RECEIPT.json')]=sha(root/'SAM_INFERENCE_RECEIPT.json')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen source/SAM result drift: '+path)
    prior=parent/'SINGLE2019_ROOF_OBSERVATIONS_20261012_15'
    packets={r['source_parent_component']:r for r in json.loads((prior/'RAW_SOURCE_PACKETS.json').read_text(encoding='utf-8'))['records']}
    prompts=json.loads((root/'SAM_SOURCE_ONLY_PROMPTS.json').read_text(encoding='utf-8'))['records']
    prompt_by_id={r['group_id']:r for r in prompts}
    infer=json.loads((root/'SAM_INFERENCE_SUMMARY.json').read_text(encoding='utf-8'))
    code=[ROOT/'src/xian_r5/sam_source_candidates.py',ROOT/'tests/test_xian_sam_source_candidates.py',Path(__file__).resolve()]
    save_json(root/'FROZEN_SAM_VECTOR_PLAN.json',{'operation':'NATIVE_PIXEL_EDGE_TO_SOURCE_GEOGRAPHIC_GRID_TO_EPSG32649',
        'source_position_may_not_snap_to_reference':True,'no_geometry_repair_or_mask_modification':True,
        'automatic_selection':False,'formal_GT':False,'protected_sha256':protected,
        'new_code_sha256':{str(f):sha(f) for f in code}})
    region=box(308460,3792190,311020,3794750);records=[];geometries={};failures=[]
    for source in infer['source_records']:
        if sha(root/source['npz'])!=source['npz_sha256']:raise ValueError('SAM mask hash changed')
        prompt=prompt_by_id[source['group_id']];packet=packets[source['source_parent_component']]
        with np.load(root/source['npz'],allow_pickle=False) as values:
            masks=values['masks'];scores=values['predicted_mask_quality']
        if masks.shape!=(3,*packet['local_image_shape']):raise ValueError('Mask coordinate frame mismatch')
        for index in range(3):
            try:record,g=sam_source_candidate(masks[index],packet,prompt,index,float(scores[index]),plan['checkpoint_sha256'])
            except ValueError as exc:
                failures.append({'candidate_id':prompt['group_id']+'_MASK_'+str(index),'reason':str(exc),
                    'raw_mask_retained':True,'source_geometry':None,'formal_disposition':None});continue
            record['source_npz']=str(root/source['npz']);record['source_npz_sha256']=source['npz_sha256']
            if g is not None:
                if not region.covers(g):raise ValueError('Source mask leaves fixed training ROI; no clipping')
                record['within_fixed_source_ROI']=True;geometries[record['candidate_id']]=g
            records.append(record)
    save_json(root/'SAM_SOURCE_GEOMETRY_CANDIDATES.json',{'records':records,'unexported_raw_candidates':failures,
        'all_masks_are_unselected':True,'candidates_are_not_complete_roof_or_GT':True})
    if geometries:
        schema={'geometry':'MultiPolygon','properties':{'candidate_id':'str','parent_component':'str',
            'mask_index':'int','area_m2':'float','model_score':'float','physical_id':'str','selected':'int','formal_gt':'int'}}
        with fiona.open(root/'SAM_SOURCE_CANDIDATES.gpkg','w',driver='GPKG',layer='unselected_source_masks',crs='EPSG:32649',schema=schema) as layer:
            for r in records:
                if r['candidate_id'] not in geometries:continue
                layer.write({'geometry':mapping(geometries[r['candidate_id']]),'properties':{
                    'candidate_id':r['candidate_id'],'parent_component':r['source_parent_component'],
                    'mask_index':r['mask_index'],'area_m2':r['area_m2'],'model_score':r['predicted_mask_quality_diagnostic_only'],
                    'physical_id':None,'selected':0,'formal_gt':0}})
        with fiona.open(root/'SAM_SOURCE_CANDIDATES.gpkg') as s:
            if len(s)!=len(geometries) or s.crs.to_epsg()!=32649:raise RuntimeError('Vector count/frame readback mismatch')
            for f in s:
                if not shape(f.geometry).equals_exact(geometries[f.properties['candidate_id']],0):raise RuntimeError('Source candidate serialized differently')
    summary={'source_candidate_masks':30,'georeferenced_unselected_candidates':len(geometries),
        'unexported_raw_masks_without_geometry_repair':len(failures),
        'mask_prompt_violation_candidates':sum(r.get('positive_prompt_pixels_missing',0)>0 or r.get('negative_prompt_pixels_included',0)>0 for r in records),
        'candidate_masks_selected':0,'clean_roofs_verified':0,'new_displacements':0,'new_training_labels':0,
        'ground_footprints':0,'physical_ids':0,'formal_reviews':0,'absolute_position_accuracy':'UNKNOWN'}
    save_json(root/'SAM_SOURCE_GEOMETRY_SUMMARY.json',summary)
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path)
    save_json(root/'SAM_VECTOR_RECEIPT.json',{'summary':summary,'new_code_sha256':{str(f):sha(f) for f in code},
        'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'output_sha256':{f.name:sha(f) for f in root.iterdir() if f.is_file()},'whole_window_governance_completed':False})
    print(json.dumps(summary))


if __name__=='__main__':main()
