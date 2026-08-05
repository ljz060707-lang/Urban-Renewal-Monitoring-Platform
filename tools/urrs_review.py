#!/usr/bin/env python
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
SRC=PROJECT_ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from urban_renewal.review.inventory import build_inventory
from urban_renewal.review.app import prepare
from urban_renewal.review.server import serve
from urban_renewal.review.review_store import ReviewStore
from urban_renewal.review.finalization import finalize_review

def default_review_dir(): return PROJECT_ROOT/'outputs/urrs/review/tongzhou_20260801'
def build_review(project=PROJECT_ROOT,review_dir=None):
    rd=Path(review_dir or default_review_dir())
    return prepare(Path(project),rd)

def main(argv=None):
    p=argparse.ArgumentParser(prog='urrs_review.py')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('status','export','finalize','serve','build','resume'):
        x=sub.add_parser(name); x.add_argument('--review-dir',default=str(default_review_dir()))
    sub.choices['serve'].add_argument('--port',type=int,default=8765); sub.choices['serve'].add_argument('--open-browser',action='store_true')
    sub.choices['resume'].add_argument('--latest',action='store_true')
    sub.choices['finalize'].add_argument('--overall-verdict',choices=['PASS','CONDITIONAL_PASS','FAIL'])
    sub.choices['finalize'].add_argument('--reviewer',default='reviewer')
    sub.choices['finalize'].add_argument('--confirmation-json')
    args=p.parse_args(argv); rd=Path(args.review_dir)
    if args.command=='resume':
        runs=sorted((PROJECT_ROOT/'outputs/urrs/review').glob('tongzhou_*'))
        if runs: rd=runs[-1]
    if args.command=='serve':
        serve(PROJECT_ROOT,rd,args.port,args.open_browser); return 0
    data=prepare(PROJECT_ROOT,rd); store=ReviewStore(rd)
    if args.command in ('build','resume'): result={'status':'success',**data}
    elif args.command=='status': result={'status':'success','summary':store.summary(data['inventory']),'review_dir':str(rd)}
    elif args.command=='export': result={'status':'success','rows':len(store.export()),'csv':str(rd/'review_decisions.csv'),'jsonl':str(rd/'review_decisions.jsonl')}
    else:
        confirmation=json.loads(Path(args.confirmation_json).read_text(encoding='utf-8')) if args.confirmation_json else {'all_mandatory_reviewed':False}
        if args.overall_verdict: confirmation['overall_verdict']=args.overall_verdict
        confirmation['reviewer']=args.reviewer; result=finalize_review(PROJECT_ROOT,rd,confirmation)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result.get('status') in ('success','pass') else 1

if __name__=='__main__': raise SystemExit(main())

