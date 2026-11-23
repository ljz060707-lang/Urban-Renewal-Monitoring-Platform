"""Continue the explicit bounded job after its independently frozen S phase.

This is one batch pipeline, not a new scheduler/database. No provider control,
new instances, protected historical tests, or old acceptance-count mutation.
"""
from datetime import datetime
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for c in iter(lambda:s.read(4194304),b''):h.update(c)
    return h.hexdigest()
def atomic(path,value):
    t=Path(str(path)+'.tmp')
    with t.open('w') as f:json.dump(value,f,indent=2,allow_nan=False)
    t.replace(path)


def clean_inference_view(root):
    """Hardlink allowlisted inputs only: no map GT, train labels, or GT bbox."""
    payload=root/'payload';view=root/'GT_FREE_VIEW';view.mkdir(exist_ok=False)
    def link(src,relative):
        dst=view/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        dst.hardlink_to(src)
    for directory in ('code','weights'):
        for src in (payload/directory).rglob('*'):
            if src.is_file() and '__pycache__' not in src.parts and (src.suffix=='.py' or src.name in ('LICENSE','LICENSE.txt') or directory=='weights'):
                link(src,src.relative_to(payload))
    for name in ('B0_model.py','NATIVE_GT_FREE_INFERENCE_INPUTS.json'):
        link(payload/'config'/name,Path('config')/name)
    records=read(payload/'config/NATIVE_GT_FREE_INFERENCE_INPUTS.json')['records']
    for row in records:
        for key in ('native_image','feature_input'):link(payload/row[key],Path(row[key]))
    return view


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    frozen=read(a.manifest);protocol=read(root/'incoming/PHASED_TRAINING_PROTOCOL.json')
    for relative,expected in frozen['input_sha256'].items():
        if sha(root/relative)!=expected:raise ValueError('Frozen followup input changed: '+relative)
    training_deadline=datetime.fromisoformat(protocol['clock']['hard_training_deadline']).timestamp()
    total_deadline=datetime.fromisoformat(protocol['clock']['hard_total_deadline']).timestamp()
    status=dict(S='RUNNING_SEPARATE_FROZEN_PROCESS',A='NOT_RUN',JOINT='NOT_RUN',
                native_inference='NOT_RUN',geo_evaluation='NOT_RUN',public_evaluation='NOT_RUN',production_admission=False)
    atomic(root/'FOLLOWUP_PROGRESS.json',status)
    # Never consume a checkpoint still being updated by another phase.
    while not (root/'S_PHASE_EXIT.json').exists():
        if time.time()>=training_deadline:
            status['reason']='NO_COMPLETED_S_HANDOFF_BEFORE_TRAINING_DEADLINE'
            atomic(root/'FOLLOWUP_EXIT.json',status);return
        time.sleep(15)
    sr=read(root/'S_PHASE_EXIT.json');status['S']=sr['result']
    if not sr.get('frozen_input_hashes_unchanged') or sr['result'].get('status') in ('FAILED_NOT_CONVERGED','CHECKPOINT_SAVE_FAILED'):
        status['reason']='S_HANDOFF_FAILED_NO_JOINT_OR_RELEASE_ALLOWED'
        atomic(root/'FOLLOWUP_EXIT.json',status);return
    sys.path[:0]=[str(root/'payload/code/src'),str(root/'payload/code/tools')]
    import torch
    from xian_cloud_a_train import run_a
    from xian_cloud_joint_phase import run_joint
    try:
        if time.time()>=training_deadline:raise TimeoutError('Training budget reached')
        status['A']=run_a(root/'payload',root/'A_TRAINING_01',protocol['A'])
        atomic(root/'FOLLOWUP_PROGRESS.json',status)
        s_checkpoint=sr['result'].get('checkpoint')
        # Preserve a trained research candidate even if the guarded best is
        # unchanged B0; this is never an implicit production promotion.
        if sr['result'].get('best_is_step0'):
            s_checkpoint=sr['result'].get('latest_checkpoint')
        a_checkpoint=status['A'].get('best_checkpoint')
        if not s_checkpoint or not a_checkpoint:raise ValueError('Missing actual trained checkpoint')
        if sr['result']['training_executed'] and status['A']['TRAINING_EXECUTED'] and time.time()<training_deadline:
            status['JOINT']=run_joint(root/'payload',root/'JOINT_TRAINING_01',protocol['JOINT'],s_checkpoint,a_checkpoint)
            atomic(root/'FOLLOWUP_PROGRESS.json',status)
        release=root/'RELEASE';release.mkdir(exist_ok=False)
        sp=torch.load(s_checkpoint,map_location='cpu',weights_only=False)['model']
        ap=torch.load(a_checkpoint,map_location='cpu',weights_only=False)['model']
        torch.save(sp,release/'S_parent_state.pth');torch.save(ap,release/'A_parent_state.pth')
        if isinstance(status['JOINT'],dict) and status['JOINT'].get('TRAINING_EXECUTED'):
            jp=status['JOINT'].get('best_checkpoint') or status['JOINT']['latest_checkpoint']
            state=torch.load(jp,map_location='cpu',weights_only=False)
            torch.save(state['S'],release/'S_state.pth');torch.save(state['A'],release/'A_state.pth')
            release_scope='JOINT_RESEARCH_CANDIDATE_NOT_PRODUCTION_ADMITTED'
        else:
            torch.save(sp,release/'S_state.pth');torch.save(ap,release/'A_state.pth')
            release_scope='S_AND_A_INDEPENDENT_TRAINED_PARENTS_NOT_JOINT_MODEL'
        del sp,ap;gc.collect();torch.cuda.empty_cache()
        atomic(release/'MODEL_FREEZE_MANIFEST.json',dict(frozen_before_FINAL_test=True,
            source_S_checkpoint=str(s_checkpoint),source_A_checkpoint=str(a_checkpoint),
            release_scope=release_scope,files={p.name:sha(p) for p in release.glob('*.pth')},
            GT_required_for_inference=False,uncertainty_m=None,production_admission=False,
            old_V2_accepted_cells_unchanged=True,old_S413_STOP_unchanged=True))
        if time.time()>=total_deadline-600:raise TimeoutError('Insufficient inference/return reserve')
        view=clean_inference_view(root)
        command=[str(root/'env/bin/python'),'-B',str(root/'payload/code/tools/xian_cloud_release_inference.py'),
                 '--payload',str(view),'--release-dir',str(release),'--output',str(release/'INFERENCE'),
                 '--experiments','E0,E1,E2,E3']
        subprocess.run(command,check=True,timeout=max(1,min(1800,int(total_deadline-time.time()-600))))
        status['native_inference']='EXECUTED_GT_FREE_FROZEN_MODEL'
        # Only the second process receives the supervision-containing payload.
        command=[str(root/'geo_env/bin/python'),'-B',str(root/'payload/code/tools/xian_cloud_release_geo.py'),
                 '--payload',str(root/'payload'),'--predictions',str(release/'INFERENCE'),
                 '--output',str(root/'RELEASE_GEO')]
        subprocess.run(command,check=True,timeout=max(1,min(1800,int(total_deadline-time.time()-300))))
        status['geo_evaluation']='EXECUTED_REFERENCE_SCOPE_ONLY_NOT_ABSOLUTE_ACCURACY'
        command=[str(root/'env/bin/python'),'-B',str(root/'payload/code/tools/xian_cloud_public_final.py'),
                 '--payload',str(root/'payload'),'--release-dir',str(release),
                 '--output',str(root/'RELEASE_PUBLIC_EVAL')]
        subprocess.run(command,check=True,timeout=max(1,min(2400,int(total_deadline-time.time()-300))))
        status['public_evaluation']='EXECUTED_ONCE_FROZEN_MODELS_NO_TEST_SELECTION'
    except Exception as exc:
        status['error']=dict(type=type(exc).__name__,traceback=traceback.format_exc())
        status['unexecuted_stages_remain_NOT_RUN']=True
    finally:
        torch.cuda.empty_cache()
        status.update(finished_at=datetime.now().astimezone().isoformat(),cloud_unique_assets_deleted=False,
             ABSOLUTE_POSITION_ACCURACY='UNKNOWN',old_models_modified=False,
             provider_release='PENDING_F_DRIVE_VERIFICATION_AND_USER_PROVIDER_CONTROL')
        atomic(root/'FINAL_RECEIPT.json',dict(run_id=root.name,phase_results=status,
            ENGINEERING_PASS='PENDING_F_DRIVE_STRICT_RESTORE',
            SCIENTIFIC_VALIDATION_PASS=False,PRODUCTION_ADMISSION_PASS=False,
            READY_FOR_CITYWIDE=False,qualified_10m_pixel_count=0,
            ABSOLUTE_POSITION_ACCURACY='UNKNOWN',
            resume_state_present='SEE_ACTUAL_CHECKPOINT_MANIFESTS',resume_execution_tested=False,
            cloud_provider_release='PENDING_USER_RELEASE',billing_stopped='UNKNOWN'))
        report=('Current cloud batch completed or stopped with the actual states below.\n\n'
                'Research candidate only. No old model promotion, no V2 review creation, '
                'no absolute ground accuracy claim, no qualified GT10m pixels.\n\n'
                'F-drive SHA return and strict replay are separate pending gates. '
                'The provider billing/release action remains with the user console.\n\n'
                '```json\n'+json.dumps(status,indent=2,allow_nan=False)+'\n```\n')
        with (root/'FINAL_REPORT.md').open('x') as f:f.write(report)
        # Final phase barrier is last: export must not race report creation.
        atomic(root/'FOLLOWUP_EXIT.json',status)
        print(json.dumps(status),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        # Even a pre-import/hash failure is an actual failed handoff, not a
        # reason for the return worker to wait forever or invent success.
        if '--root' in sys.argv:
            failure_root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
            if failure_root.is_dir() and not (failure_root/'FOLLOWUP_EXIT.json').exists():
                atomic(failure_root/'FOLLOWUP_EXIT.json',dict(status='FAILED_BEFORE_PHASE_HANDOFF',
                    error_type=type(exc).__name__,traceback=traceback.format_exc(),
                    A='NOT_RUN',JOINT='NOT_RUN',native_inference='NOT_RUN',
                    scientific_validation=False,production_admission=False))
        raise
