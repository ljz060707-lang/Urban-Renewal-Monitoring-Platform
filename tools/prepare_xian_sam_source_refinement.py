"""Freeze two source-image prompt corrections, no GT-guided refitting."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.source_prompt_refinement import refine_source_prompt


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    prior=parent/'SINGLE2019_SOURCE_ROOF_RETENTION_20261012_17';sam=parent/'SINGLE2019_SAM_ROOF_PILOT_20261012_16'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh source-only refinement output required')
    old_plan=json.loads((prior/'FROZEN_RETENTION_AND_SHAPE_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((prior/'SOURCE_ROOF_RETENTION_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(old_plan['protected_sha256']);protected.update(old_plan['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(prior/'SOURCE_ROOF_RETENTION_RECEIPT.json')]=sha(prior/'SOURCE_ROOF_RETENTION_RECEIPT.json')
    refinements_path=sam/'FROZEN_SOURCE_ONLY_PROMPT_REFINEMENTS.json';protected[str(refinements_path)]=sha(refinements_path)
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen source/retention changed: '+path)
    base_plan=json.loads((sam/'FROZEN_SAM_PILOT_PLAN.json').read_text(encoding='utf-8'))
    base_prompts={r['group_id']:r for r in json.loads((sam/'SAM_SOURCE_ONLY_PROMPTS.json').read_text(encoding='utf-8'))['records']}
    surfaces=json.loads((parent/'SINGLE2019_ROOF_OBSERVATIONS_20261012_15/SOURCE_ROOF_OBSERVATIONS.json').read_text(encoding='utf-8'))['records']
    refinements=json.loads(refinements_path.read_text(encoding='utf-8'))['refinements']
    prompts=[refine_source_prompt(base_prompts[r['base_group_id']],r,surfaces) for r in refinements]
    code=[ROOT/'src/xian_r5/source_prompt_refinement.py',ROOT/'tests/test_xian_source_prompt_refinement.py',
        Path(__file__).resolve(),ROOT/'tools/infer_xian_sam_roof_pilot_cpu.py']
    plan={k:base_plan[k] for k in ('backend','sam_commit','runtime','checkpoint','checkpoint_sha256','device','torch_threads',
        'model_training','conda_environment_modified','multimask_output','all_3_masks_saved','automatic_mask_selection_or_formal_acceptance',
        'predicted_mask_quality_is_metric_accuracy_or_uncertainty','source_observed_date','experiment_year','GT_or_target_positions_used',
        'prompt_role','ground_ties_or_human_reviews_generated','roof_purity_or_complete_instance_verified','RBF_SAM_status')}
    plan.update(run=out.name,source_packet_count=1,source_prompt_groups=2,
        source_only_refinement_basis='FROZEN_STAGE16_VISUAL_DEFECTS; NOT_STAGE17_MATCH_RESULTS',
        old_masks_and_vectors_unchanged=True,parameter_sweep=False,
        protected_sha256=protected,new_code_sha256={str(f):sha(f) for f in code})
    out.mkdir();save_json(out/'FROZEN_SAM_PILOT_PLAN.json',plan)
    save_json(out/'SAM_SOURCE_ONLY_PROMPTS.json',{'records':prompts,'source_annotation_position_frozen_before_inference':True,
        'refinement_definitions':refinements,'source_prompt_baseline':str(sam/'SAM_SOURCE_ONLY_PROMPTS.json')})
    print(json.dumps({'run':out.name,'source_refined_groups':len(prompts),'old_masks_changed':0,'inference':'NOT_YET_RUN'}))


if __name__=='__main__':main()
