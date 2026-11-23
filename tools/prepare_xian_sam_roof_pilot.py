"""Freeze an isolated official-SAM source-only pilot before inference."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.sam_source_prompts import source_segmentation_prompts


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment');prior=parent/'SINGLE2019_ROOF_OBSERVATIONS_20261012_15'
    runtime=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/22_roof_candidate_segmentation/SAM_VITB_RUNTIME_20261012_02')
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh SAM pilot output required')
    old_plan=json.loads((prior/'FROZEN_OBSERVATION_BINDING_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((prior/'MODEL_OBSERVATION_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(old_plan['protected_sha256']);protected.update(old_plan['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(prior/'MODEL_OBSERVATION_RECEIPT.json')]=sha(prior/'MODEL_OBSERVATION_RECEIPT.json')
    definitions_path=prior/'SOURCE_ONLY_SAM_GROUP_DEFINITIONS.json';protected[str(definitions_path)]=sha(definitions_path)
    runtime_receipt=json.loads((runtime/'OFFICIAL_SAM_RUNTIME_RECEIPT.json').read_text(encoding='utf-8'))
    protected[str(runtime/'sam_vit_b_01ec64.pth')]=runtime_receipt['weights']['sha256']
    protected.update({str(runtime/'sam_source'/name):h for name,h in runtime_receipt['source_files_sha256'].items()})
    protected[str(runtime/'OFFICIAL_SAM_RUNTIME_RECEIPT.json')]=sha(runtime/'OFFICIAL_SAM_RUNTIME_RECEIPT.json')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen source/runtime changed: '+path)
    records=json.loads((prior/'SOURCE_ROOF_OBSERVATIONS.json').read_text(encoding='utf-8'))['records']
    packets={r['source_parent_component']:r for r in json.loads((prior/'RAW_SOURCE_PACKETS.json').read_text(encoding='utf-8'))['records']}
    definitions=json.loads(definitions_path.read_text(encoding='utf-8'))
    if definitions['target_or_GT_position_used']:raise ValueError('Source-only SAM scope violated')
    prompts=source_segmentation_prompts(records,packets,definitions['groups'])
    code=[ROOT/'src/xian_r5/sam_source_prompts.py',ROOT/'tests/test_xian_sam_source_prompts.py',
        Path(__file__).resolve(),ROOT/'tools/infer_xian_sam_roof_pilot_cpu.py']
    plan={'run':out.name,'backend':'OFFICIAL_SAM_VIT_B_NOT_RBF_SAM','sam_commit':runtime_receipt['commit'],
        'runtime':str(runtime),'checkpoint':str(runtime/'sam_vit_b_01ec64.pth'),
        'checkpoint_sha256':runtime_receipt['weights']['sha256'],'device':'cpu','torch_threads':2,
        'model_training':False,'conda_environment_modified':False,'source_packet_count':len({r['source_parent_component'] for r in prompts}),
        'source_prompt_groups':len(prompts),'multimask_output':True,'all_3_masks_saved':True,
        'automatic_mask_selection_or_formal_acceptance':False,
        'predicted_mask_quality_is_metric_accuracy_or_uncertainty':False,
        'source_observed_date':'2018-11-01','experiment_year':2019,'GT_or_target_positions_used':False,
        'prompt_role':'SOURCE_SEGMENTATION_NOT_REGISTRATION_TIES','ground_ties_or_human_reviews_generated':False,
        'roof_purity_or_complete_instance_verified':False,'RBF_SAM_status':'NOT_RUN_OFFICIAL_RBF_CHECKPOINT_UNAVAILABLE',
        'protected_sha256':protected,'new_code_sha256':{str(f):sha(f) for f in code}}
    out.mkdir();save_json(out/'FROZEN_SAM_PILOT_PLAN.json',plan)
    save_json(out/'SAM_SOURCE_ONLY_PROMPTS.json',{'records':prompts,'source_annotation_position_frozen_before_inference':True})
    print(json.dumps({'run':out.name,'source_groups':len(prompts),'backend':'SAM_VIT_B','device':'cpu','inference':'NOT_YET_RUN'}))


if __name__=='__main__':main()
