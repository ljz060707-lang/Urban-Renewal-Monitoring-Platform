from pathlib import Path
import json,hashlib,urllib.request,datetime,base64,time
R=Path('/root/xian20261111_r14_native_temporal');D=R/'code/official_supcontrast';D.mkdir(exist_ok=True)
req=urllib.request.Request('https://api.github.com/repos/HobbitLong/SupContrast/commits/master',headers={'User-Agent':'xian-research-code-provenance'})
with urllib.request.urlopen(req,timeout=40) as r:commit=json.load(r)['sha']
files=[]
for name in ['losses.py','LICENSE']:
 url=f'https://api.github.com/repos/HobbitLong/SupContrast/contents/{name}?ref={commit}'
 request=urllib.request.Request(url,headers={'User-Agent':'xian-research-code-provenance'})
 for attempt in range(3):
  try:
   with urllib.request.urlopen(request,timeout=40) as r:payload=json.load(r)
   content=base64.b64decode(payload['content']);break
  except Exception:
   if attempt==2:raise
   time.sleep(2)
 (D/name).write_bytes(content);files.append(dict(path=name,url=url,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
receipt=dict(method='SUPERVISED_CONTRASTIVE_LOSS_ADAPTED_TO_WEAK_MAP_REFERENCE_IDENTITIES',paper='https://arxiv.org/abs/2004.11362',official_repository='https://github.com/HobbitLong/SupContrast',commit=commit,files=files,downloaded_data_or_checkpoint=False,source_or_protected_pixels=0,time=datetime.datetime.now(datetime.timezone.utc).isoformat());(D/'SOURCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
