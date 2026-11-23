import os
os.environ['ALIGNMENT_CLEAN']='1'
from ml import *
import shutil,hashlib
snap=OUT/'final_training_source';snap.mkdir(exist_ok=True)
for name in ('ml.py','refine_alignment.py','calibrate_facade.py','infer_models.py','produce.py','run_pipeline.py'):
 source=Path(__file__).parent/name
 if source.exists():shutil.copy2(source,snap/name)
save('FINAL_TRAINING_CODE_BINDING.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.glob('*.py')})
x=np.load(IN/'align_x.npy',mmap_mode='r');y=np.load(IN/'align_y.npy',mmap_mode='r');s=np.load(IN/'align_shift.npy');good=np.zeros(len(x),'bool');iou=np.zeros(len(x),'float32')
for k in range(0,len(x),128):
 a=x[k:k+128,3]>.5;b=y[k:k+128]>.5;inter=(a&b).sum((1,2));union=(a|b).sum((1,2));v=inter/np.maximum(union,1);ratio=a.sum((1,2))/np.maximum(b.sum((1,2)),1);iou[k:k+len(v)]=v;good[k:k+len(v)]=(v>=.60)&(ratio>=.6)&(ratio<=1.65)&(np.linalg.norm(s[k:k+len(v)]*10,axis=1)<=25)
split=np.load(IN/'align_split.npy');good&=split>=0
assert all(((split==j)&good).sum()>100 for j in (0,1,2));np.save(IN/'alignment_quality.npy',good)
save('ALIGNMENT_SUPERVISION_FILTER.json',dict(rule='centered mask IoU>=0.60; raw/reference area ratio0.60..1.65; center distance<=25m; complete native context',kept=int(good.sum()),total=len(good),split_counts={str(j):int(((split==j)&good).sum()) for j in range(3)},purpose='Reject weak nearest-reference associations, not infer absent buildings'))
archive=OUT/'preliminary_alignment';archive.mkdir(exist_ok=True)
for p in list(OUT.glob('alignment_*')):
 if p.is_file():p.replace(archive/p.name)
status('REFINE_ALIGNMENT_CLEAN',kept=int(good.sum()));fit('alignment')
import subprocess,sys
subprocess.run([sys.executable,'-u',str(Path(__file__).parent/'calibrate_facade.py')],check=True)
save('MODEL_FREEZE.json',dict(time=datetime.now().astimezone().isoformat(),sequence='preliminary alignment, facade training, filtered alignment refinement, threshold calibration, freeze, then city inference',checkpoints=['alignment_best.pt','facade_best.pt'],protected_reads=0));status('MODELS_FROZEN_FINAL')
