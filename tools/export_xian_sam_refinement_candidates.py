"""Dynamic-count source mask exporter for the independently frozen refinement."""
from pathlib import Path
import argparse,json,sys
import numpy as np,fiona
from shapely.geometry import shape,mapping,box
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.sam_source_candidates import sam_source_candidate


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);root=p.parse_args().run.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if root.parent!=parent.resolve() or (root/'SAM_REFINEMENT_VECTOR_RECEIPT.json').exists():raise ValueError('Append-only completed refinement required')
    plan=json.loads((root/'FROZEN_SAM_PILOT_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((root/'SAM_INFERENCE_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(plan['protected_sha256']);protected.update(plan['new_code_sha256'])
    protected.update({str(root/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(root/'SAM_INFERENCE_RECEIPT.json')]=sha(root/'SAM_INFERENCE_RECEIPT.json')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen source changed: '+path)
    code=[ROOT/'src/xian_r5/sam_source_candidates.py',ROOT/'tests/test_xian_sam_source_candidates.py',Path(__file__).resolve()]
    save_json(root/'FROZEN_REFINEMENT_VECTOR_PLAN.json',{'no_mask_repair_smoothing_GT_snapping_or_selection':True,
        'protected_sha256':protected,'new_code_sha256':{str(f):sha(f) for f in code}})
    packets={r['source_parent_component']:r for r in json.loads((parent/'SINGLE2019_ROOF_OBSERVATIONS_20261012_15/RAW_SOURCE_PACKETS.json').read_text(encoding='utf-8'))['records']}
    prompts={r['group_id']:r for r in json.loads((root/'SAM_SOURCE_ONLY_PROMPTS.json').read_text(encoding='utf-8'))['records']}
    infer=json.loads((root/'SAM_INFERENCE_SUMMARY.json').read_text(encoding='utf-8'))
    records=[];geometries={};failures=[];roi=box(308460,3792190,311020,3794750)
    for group in infer['source_records']:
        with np.load(root/group['npz'],allow_pickle=False) as array:masks=array['masks'];scores=array['predicted_mask_quality']
        for index,mask in enumerate(masks):
            try:r,g=sam_source_candidate(mask,packets[group['source_parent_component']],prompts[group['group_id']],index,float(scores[index]),plan['checkpoint_sha256'])
            except ValueError as exc:
                failures.append({'candidate_id':group['group_id']+'_MASK_'+str(index),'reason':str(exc),'raw_mask_retained':True});continue
            r['source_npz']=str(root/group['npz']);r['source_npz_sha256']=group['npz_sha256']
            if g is not None:
                if not roi.covers(g):raise ValueError('Refinement leaves fixed source ROI')
                geometries[r['candidate_id']]=g
            records.append(r)
    save_json(root/'SAM_REFINED_SOURCE_GEOMETRY_CANDIDATES.json',{'records':records,'unexported_raw_masks':failures,
        'original_masks_replaced':False,'selected_mask_candidates':0})
    if geometries:
        schema={'geometry':'MultiPolygon','properties':{'candidate_id':'str','area_m2':'float','mask_index':'int','physical_id':'str','formal_gt':'int'}}
        with fiona.open(root/'SAM_REFINED_SOURCE_CANDIDATES.gpkg','w',driver='GPKG',layer='unselected_refined_source_masks',crs='EPSG:32649',schema=schema) as s:
            for r in records:
                if r['candidate_id'] in geometries:s.write({'geometry':mapping(geometries[r['candidate_id']]),'properties':{
                    'candidate_id':r['candidate_id'],'area_m2':r['area_m2'],'mask_index':r['mask_index'],'physical_id':None,'formal_gt':0}})
        with fiona.open(root/'SAM_REFINED_SOURCE_CANDIDATES.gpkg') as s:
            if len(s)!=len(geometries):raise RuntimeError('Refinement vector count drift')
            for f in s:
                if not shape(f.geometry).equals_exact(geometries[f.properties['candidate_id']],0):raise RuntimeError('Refined source geometry serialization drift')
    summary={'source_groups_inferred':len(infer['source_records']),'source_raw_masks':sum(len(r['mask_pixels']) for r in infer['source_records']),
        'georeferenced_unselected_masks':len(geometries),'geometry_failures_without_repair':len(failures),
        'prompt_violation_masks':sum(r.get('positive_prompt_pixels_missing',0)>0 or r.get('negative_prompt_pixels_included',0)>0 for r in records),
        'original_masks_replaced':0,'registration_vectors_estimated_or_applied':0,'new_labels':0,'formal_gt':0}
    save_json(root/'SAM_REFINEMENT_VECTOR_SUMMARY.json',summary)
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path)
    save_json(root/'SAM_REFINEMENT_VECTOR_RECEIPT.json',{'summary':summary,'new_code_sha256':{str(f):sha(f) for f in code},
        'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'output_sha256':{f.name:sha(f) for f in root.iterdir() if f.is_file()},'whole_window_governance_completed':False})
    print(json.dumps(summary))


if __name__=='__main__':main()
