"""Generic upstream code/model only; filter by role before any file bytes download."""
from pathlib import Path
import json,urllib.request,base64,time,hashlib,traceback,re
R=Path('/root/xian20261111_r14_native_temporal');D=R/'external/dinov2';D.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=D/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'xian-admitted-research-code-only'})
 for attempt in range(3):
  try:
   with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)
  except Exception:
   if attempt==2:raise
   time.sleep(2)
def main():
 frozen=D/'ROLE_FILTER_BEFORE_FILE_DOWNLOAD.json'
 if frozen.exists():plan=json.loads(frozen.read_text());commit=plan['commit'];selected=plan['files']
 else:
  commit=get('https://api.github.com/repos/facebookresearch/dinov2/commits/main')['sha'];tree=get(f'https://api.github.com/repos/facebookresearch/dinov2/git/trees/{commit}?recursive=1')['tree'];allow={'dinov2/__init__.py','dinov2/models/__init__.py','dinov2/models/vision_transformer.py','dinov2/hub/backbones.py','LICENSE','MODEL_CARD.md','README.md'};selected=[r for r in tree if r['type']=='blob' and (r['path'] in allow or r['path'].startswith('dinov2/layers/') and r['path'].endswith('.py'))];assert len(selected)<30;save(frozen.name,dict(commit=commit,files=selected,roles='MODEL_IMPLEMENTATION_LICENSE_CARD_AND_GENERIC_PRETRAINED_WEIGHTS_ONLY',dataset_pixels_or_labels=False,protected_local_assets=False,generic_pretraining_overlap='LVD142M_GENERIC_CORPUS_NOT_AUDITED_FOR_ACCIDENTAL_OVERLAP;NO_PROJECT_PROTECTED_ASSET_READ_OR_TARGET_FINE_LABEL_SELECTION',weight_variant='ORIGINAL_GENERIC_VIT_BASE14_NOT_XRAY_OR_DOMAIN_FINETUNED'))
 records=[]
 for row in selected:
  path=D/'source'/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
  if not path.exists():
   payload=get(row['url']);content=base64.b64decode(payload['content']);assert len(content)==row['size'];path.write_bytes(content)
  content=path.read_bytes();assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==row['sha']
  records.append(dict(path=row['path'],blob_sha=row['sha'],sha256=sha(path),bytes=path.stat().st_size))
 readme=(D/'source/README.md').read_text();url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth';assert url in readme,'Verify official checkpoint URL before download'
 weight=D/'dinov2_vitb14_pretrain.pth'
 if not weight.exists():
  save('STATUS.json',dict(time=time.time(),stage='DOWNLOAD_OFFICIAL_GENERIC_WEIGHT',url=url));tmp=weight.with_suffix('.returning')
  with urllib.request.urlopen(url,timeout=120) as response,tmp.open('wb') as stream:
   total=0
   while True:
    block=response.read(1024*1024)
    if not block:break
    stream.write(block);total+=len(block)
    if total%(32*1024*1024)==0:save('STATUS.json',dict(time=time.time(),stage='DOWNLOAD_OFFICIAL_GENERIC_WEIGHT',bytes=total))
  assert 300_000_000<total<400_000_000;tmp.replace(weight)
 assert 300_000_000<weight.stat().st_size<400_000_000
 save('SOURCE_RECEIPT.json',dict(status='DOWNLOADED_CODE_AND_GENERIC_PRETRAINED_WEIGHT_NOT_YET_MODEL_FORWARD_ADMITTED',official_repository='https://github.com/facebookresearch/dinov2',paper='https://arxiv.org/abs/2304.07193',commit=commit,files=records,weights=dict(path=str(weight),url=url,bytes=weight.stat().st_size,sha256=sha(weight)),license='OFFICIAL_APACHE2_CODE_AND_GENERIC_DINOv2_WEIGHTS',training_corpus='LVD142M_GENERIC_SELF_SUPERVISED',project_protected_pixel_or_GT_reads=0,dataset_pixel_downloads=0,model_forwards=0,optimizer_updates=0));print('DINOV2_GENERIC_ASSETS_DOWNLOADED',weight.stat().st_size,sha(weight),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save('DOWNLOAD_FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
