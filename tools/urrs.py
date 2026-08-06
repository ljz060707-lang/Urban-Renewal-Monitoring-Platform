#!/usr/bin/env python
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
SRC=PROJECT_ROOT/'src'
TOOLS=PROJECT_ROOT/'tools'
for p in (SRC,TOOLS):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from urban_renewal.cli import finalize,inventory,models_verify,registry_validate,system_status
from urban_renewal.geometry import audit_alignment,audit_geometry,geometry_status,vectorize_geometry
from urban_renewal.inference import run_geometry_overlay
from urban_renewal.review.server import serve
from urban_renewal.review.app import prepare
from urban_renewal.review.review_store import ReviewStore
from urban_renewal.review.finalization import finalize_review
from urban_renewal.construction.cli import add_construction_parser,handle_construction
from urban_monitor.building_foundation_v2.onboarding import (
    DEFAULT_NATIVE_PYTHON,
    DEFAULT_NATIVE_REPO,
    create_readiness_run,
    finalize_readiness,
    input_contract,
    local_domain_protocol,
    prepare_annotation_package,
    run_readiness,
    satellite_infer,
    satellite_preflight,
)

def main(argv=None):
    parser=argparse.ArgumentParser(prog='urrs.py',description='Urban Renewal Remote Sensing System canonical entrypoint')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('status'); sub.add_parser('inventory')
    r=sub.add_parser('registry'); r.add_argument('action',choices=['validate'])
    m=sub.add_parser('models'); m.add_argument('action',choices=['verify'])
    g=sub.add_parser('geometry'); g.add_argument('action',choices=['status','diagnose','audit','vectorize','alignment','infer'])
    sub.add_parser('tongzhou').add_argument('action',choices=['run'])
    sub.add_parser('overlay').add_argument('action',choices=['run'])
    rv=sub.add_parser('review'); rv.add_argument('action',choices=['build','serve','status','export','finalize']); rv.add_argument('--review-dir'); rv.add_argument('--port',type=int,default=8765); rv.add_argument('--open-browser',action='store_true'); rv.add_argument('--review-file')
    add_construction_parser(sub)
    b=sub.add_parser('building', help='Frozen B0 building-foundation onboarding and explicit GeoTIFF inference')
    b.add_argument('action', choices=['readiness','finalize','satellite-preflight','satellite-infer','prepare-annotation-package','status'])
    b.add_argument('--run-dir')
    b.add_argument('--image')
    b.add_argument('--output')
    b.add_argument('--date', default='2024-01-01')
    b.add_argument('--rgb-bands', default='1,2,3')
    b.add_argument('--native-python', default=str(DEFAULT_NATIVE_PYTHON))
    b.add_argument('--native-repo', default=str(DEFAULT_NATIVE_REPO))
    b.add_argument('--config', default=str(PROJECT_ROOT/'outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/Swin_T_whu_windows_pilot5000.py'))
    b.add_argument('--device', default='cuda:0')
    b.add_argument('--batch-size', type=int, default=2)
    sc=sub.add_parser('system'); sc.add_argument('action',choices=['finalize']); sc.add_argument('--review-file')
    args=parser.parse_args(argv); root=PROJECT_ROOT; review_dir=Path(args.review_dir) if getattr(args,'review_dir',None) else root/'outputs/urrs/review/tongzhou_20260801'
    try:
        if args.command=='status': result=system_status(root)
        elif args.command=='inventory': result=inventory(root)
        elif args.command=='registry': result=registry_validate(root)
        elif args.command=='models': result=models_verify(root)
        elif args.command=='geometry':
            gr=root/'outputs/inference/beijing_tongzhou_building_geometry'
            if args.action in ('status','diagnose','infer'): result=geometry_status(gr)
            elif args.action=='audit': result=audit_geometry(gr,checkpoint_sha256='f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7')
            elif args.action=='vectorize': result=vectorize_geometry(gr)
            else: result=audit_alignment(gr)
        elif args.command=='tongzhou':
            gr=root/'outputs/inference/beijing_tongzhou_building_geometry'; audit_geometry(gr,checkpoint_sha256='f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7'); vectorize_geometry(gr); result=audit_alignment(gr)
        elif args.command=='overlay': result=run_geometry_overlay(root)
        elif args.command=='review':
            if args.action=='serve': serve(root,review_dir,args.port,args.open_browser); return 0
            data=prepare(root,review_dir); store=ReviewStore(review_dir)
            if args.action=='build': result=data
            elif args.action=='status': result={'status':'success','summary':store.summary(data['inventory']),'review_dir':str(review_dir)}
            elif args.action=='export': result={'status':'success','rows':len(store.export()),'csv':str(review_dir/'review_decisions.csv'),'jsonl':str(review_dir/'review_decisions.jsonl')}
            else:
                confirmation=json.loads(Path(args.review_file).read_text(encoding='utf-8')) if args.review_file else {'all_mandatory_reviewed':False}
                result=finalize_review(root,review_dir,confirmation)
        elif args.command=='construction': result=handle_construction(root,args)
        elif args.command=='building':
            def _run_dir() -> Path:
                if getattr(args, 'run_dir', None):
                    value = Path(args.run_dir)
                    return value if value.is_absolute() else root / value
                return create_readiness_run(root)
            if args.action == 'status':
                candidates = sorted((root/'outputs/readiness').glob('building_foundation_v2_satellite_onboarding_*/10_acceptance/FINAL_SATELLITE_ONBOARDING_READINESS.json'), key=lambda p: p.stat().st_mtime, reverse=True)
                if not candidates:
                    result = {'status': 'blocked', 'reason': 'no satellite onboarding readiness run found'}
                else:
                    path = candidates[0]
                    result = json.loads(path.read_text(encoding='utf-8')); result['receipt_path'] = str(path.resolve())
            elif args.action == 'readiness':
                requested = Path(args.run_dir) if getattr(args, 'run_dir', None) else None
                result = run_readiness(root, requested=requested, native_python=Path(args.native_python), native_repo=Path(args.native_repo))
            elif args.action == 'finalize':
                if not args.run_dir: raise ValueError('--run-dir is required for building finalize')
                run = Path(args.run_dir); run = run if run.is_absolute() else root / run
                if not (run/'06_tiled_inference_smoke/TILED_INFERENCE_EQUIVALENCE.json').exists():
                    raise FileNotFoundError('readiness smoke receipt is missing; refusing to finalize')
                # Refresh governance after the user commits source changes;
                # this does not rerun inference or read any satellite pixels.
                from urban_monitor.building_foundation_v2.onboarding import governance
                governance(root, run)
                result = finalize_readiness(root, run); result['run_dir'] = str(run.resolve()); result['smoke_rerun'] = False; result['user_satellite_pixel_reads'] = 0
            elif args.action == 'prepare-annotation-package':
                run = _run_dir(); prepare_annotation_package(root, run); result = {'status': 'pass', 'run_dir': str(run.resolve()), 'annotation_contract': str((run/'07_annotation_contract').resolve()), 'test_pixel_reads': 0}
            elif args.action == 'satellite-preflight':
                if not args.image: raise ValueError('--image is required for satellite-preflight')
                run = _run_dir(); result = satellite_preflight(root, run, Path(args.image), rgb_bands=tuple(int(v.strip()) for v in args.rgb_bands.split(',')))
                result['run_dir'] = str(run.resolve())
            else:
                if not args.image: raise ValueError('--image is required for satellite-infer')
                run = _run_dir(); config = Path(args.config); result = satellite_infer(root, run, Path(args.image), native_python=Path(args.native_python), native_repo=Path(args.native_repo), config=config, date=args.date, device=args.device, batch_size=args.batch_size); result['run_dir'] = str(run.resolve())
        else:
            result=finalize(root)
            if args.review_file:
                review=json.loads(Path(args.review_file).read_text(encoding='utf-8'))
                if review.get('status') != 'success': raise ValueError('human review receipt is not successful')
                from datetime import datetime
                stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
                state_path=root/'outputs/urrs/status'/('SYSTEM_STATE_'+stamp+'.json')
                release_path=root/'outputs/urrs/status'/('FINAL_RELEASE_MANIFEST_'+stamp+'.json')
                state=json.loads((root/'outputs/urrs/status/SYSTEM_STATE.json').read_text(encoding='utf-8'))
                state['human_review']=review
                state['gates']['HUMAN_REVIEW_READY']=True
                state['gates']['INFERENCE_READY']=bool(review.get('inference_ready',False) and state['gates'].get('NUMERICAL_AUDIT_READY',False))
                state['inference_ready']=state['gates']['INFERENCE_READY']
                state['current_stage']='candidate_release'
                state['review_status']=review.get('overall_verdict')
                state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
                release={'status':'success','review_file':str(Path(args.review_file).resolve()),'overall_verdict':review.get('overall_verdict'),'scientific_status':'candidate_not_ground_truth','inference_ready':state['inference_ready'],'generated_at':datetime.now().isoformat()}
                release_path.write_text(json.dumps(release,ensure_ascii=False,indent=2),encoding='utf-8')
                result.update({'human_review':review,'release_manifest':str(release_path),'state_snapshot':str(state_path)})
    except Exception as exc:
        result={'status':'failed','error':f'{type(exc).__name__}: {exc}'}; print(json.dumps(result,ensure_ascii=False,indent=2)); return 1
    print(json.dumps(result,ensure_ascii=False,indent=2))
    command_status=result.get('status',result.get('geometry_numerical_status',result.get('alignment_status','pass')))
    command_completed=command_status in ('pass','success','ready_for_pixelwise_temporal_tracking','awaiting_user_commit_after_compatibility_fix','satellite_onboarding_ready_b0_frozen') or (isinstance(command_status,str) and command_status.startswith('TONGZHOU_CONSTRUCTION_'))
    command_blocked=isinstance(command_status,str) and (command_status.startswith('FORMAL_TRAINING_BLOCKED_') or command_status in ('blocked','not_run'))
    return 0 if command_completed or command_blocked else 1
if __name__=='__main__': raise SystemExit(main())

