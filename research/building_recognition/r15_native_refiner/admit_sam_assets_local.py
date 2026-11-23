"""Only official generic SAM code/checkpoint; no SA1B or other dataset downloads."""
from pathlib import Path
import subprocess,json,hashlib,time,concurrent.futures
R=Path(__file__).parent;D=R/'external/sam';D.mkdir(parents=True,exist_ok=True)
URL='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
def get(url):
 r=subprocess.run(['curl.exe','-sS','-L','--fail','--retry','2','--connect-timeout','15','--max-time','90',url],capture_output=True,check=True,timeout=120);return json.loads(r.stdout)
def download(url,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.returning');subprocess.run(['curl.exe','-sS','-L','--fail','--retry','2','--connect-timeout','15','--max-time','300','-o',str(tmp),url],check=True,timeout=360);tmp.replace(path);return path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=D/'ROLE_FILTER_BEFORE_FILE_DOWNLOAD.json'
if plan.exists():data=json.loads(plan.read_text(encoding='utf-8'));commit=data['commit'];selected=data['files']
else:
 commit=get('https://api.github.com/repos/facebookresearch/segment-anything/commits/main')['sha'];tree=get(f'https://api.github.com/repos/facebookresearch/segment-anything/git/trees/{commit}?recursive=1')['tree'];selected=[r for r in tree if r['type']=='blob' and (r['path'] in ['README.md','LICENSE'] or r['path'].startswith('segment_anything/') and r['path'].endswith('.py'))];data=dict(commit=commit,files=selected,weight_URL=URL,variant='ORIGINAL_GENERIC_SAM_VITB_SA1B_NOT_BUILDING_FINETUNE',dataset_pixel_or_label_downloads=False,historical_protected_asset_reads=0,generic_corpus_accidental_overlap='NOT_AUDITED_NO_PROJECT_FINE_LABEL_SELECTION',stage='ASSET_ADMISSION_NOT_MODEL_USE');plan.write_text(json.dumps(data,indent=2),encoding='utf-8')
def source_file(row):
 p=D/'source'/row['path'];url=f"https://raw.githubusercontent.com/facebookresearch/segment-anything/{commit}/{row['path']}"
 if not p.exists():download(url,p)
 b=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['sha'];return dict(path=row['path'],blob_sha=row['sha'],bytes=len(b),sha256=sha(p),URL=url)
readme=next(r for r in selected if r['path']=='README.md');source_file(readme);assert URL in (D/'source/README.md').read_text(encoding='utf-8')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(source_file,selected))
weight=D/'sam_vit_b_01ec64.pth'
if not weight.exists():download(URL,weight)
assert 300_000_000<weight.stat().st_size<500_000_000
receipt=dict(time=time.time(),status='DOWNLOADED_OFFICIAL_SAM_ASSETS_NOT_YET_FORWARD_ADMITTED',official_repository='https://github.com/facebookresearch/segment-anything',commit=commit,source_files=records,weights=dict(path=str(weight),URL=URL,bytes=weight.stat().st_size,sha256=sha(weight)),current_model_forwards=0,current_optimizer_updates=0,dataset_downloads=0,protected_reads=0);(D/'LOCAL_SOURCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print('OFFICIAL_SAM_ASSETS_DOWNLOADED',weight.stat().st_size,sha(weight))
