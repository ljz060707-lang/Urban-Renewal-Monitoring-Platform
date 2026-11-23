"""Allowlisted immutable training bundle creation and content verification.

No access to SSH keys. The input is the explicit payload/ staging directory,
never the whole run directory (which also contains private connection files).
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for c in iter(lambda:stream.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--part',choices=('data','code','s_launch'),required=True);a=p.parse_args()
    run=a.run.resolve();root=run/'payload';target=run/f'cloud_{a.part}_payload.tar'
    if a.part=='s_launch':
        sources={
            'payload/code/tools/xian_cloud_s_train.py':root/'code/tools/xian_cloud_s_train.py',
            'payload/code/src/xian_r5/cloud_joint/segmentation.py':root/'code/src/xian_r5/cloud_joint/segmentation.py',
            'payload/config/segmentation_data.json':root/'config/segmentation_data.json',
            'payload/config/B0_model.py':root/'config/B0_model.py',
            'payload/weights/B0.pth':root/'weights/B0.pth',
            'incoming/PHASED_TRAINING_PROTOCOL.json':run/'PHASED_TRAINING_PROTOCOL.json',
            'incoming/run_xian_cloud_stage_s.py':Path(__file__).parent/'run_xian_cloud_stage_s.py'}
        receipt=run/'S_LAUNCH_MANIFEST.json'
        info={'schema':'XIAN_CLOUD_S_LAUNCH_V1','input_sha256':{n:sha(p) for n,p in sources.items()},
              'no_GT_map_in_S_training':True,'historical_models_unchanged':True,
              'new_model_or_production_promotion':False,'training_at_freeze':'NOT_RUN'}
        with receipt.open('x',encoding='utf-8') as stream:json.dump(info,stream,indent=2,allow_nan=False)
        print(json.dumps({'launch_manifest_sha256':sha(receipt),'input_files':len(sources)}),flush=True)
        return
    receipt=run/f'CLOUD_{a.part.upper()}_PAYLOAD_MANIFEST.json'
    if target.exists() or receipt.exists():raise ValueError('Append-only payload already exists')
    allowed=('data','config') if a.part=='data' else ('code','weights','prompts')
    members=[]
    for folder in allowed:
        for path in sorted((root/folder).rglob('*')):
            if not path.is_file():continue
            relative=path.relative_to(root).as_posix()
            if path.is_symlink() or path.suffix.lower() in ('.pem','.ppk','.key') or '__pycache__' in path.parts:
                raise ValueError('Forbidden payload member')
            members.append((path,relative,sha(path),path.stat().st_size))
    with tarfile.open(target,'x') as tar:
        for path,relative,_,_ in members:tar.add(path,arcname=relative,recursive=False)
    info={'part':a.part,'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size,
          'files':{name:{'sha256':h,'bytes':n} for _,name,h,n in members},
          'contains_secrets':False,'contains_old_protected_tests':False}
    with receipt.open('x',encoding='utf-8') as stream:json.dump(info,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({k:v for k,v in info.items() if k!='files'}),flush=True)


if __name__=='__main__':main()
