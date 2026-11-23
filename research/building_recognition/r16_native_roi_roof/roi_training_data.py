from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,hashlib
from functools import lru_cache
import numpy as np,torch,cv2
from pycocotools import mask as mu
R=Path('/root/xian20261111_r16_roi_roof');D=R/'data'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
@lru_cache(maxsize=64)
def instances(p,digest):assert sha(p)==digest;return np.load(p)['instances']
@lru_cache(maxsize=128)
def valid(p,digest):assert sha(p)==digest;return np.load(p)['valid']
class ROIRoofDataset(torch.utils.data.Dataset):
 def __init__(self):
  manifest=json.loads((D/'SLOT_MANIFEST_FROZEN.json').read_text());supervision=json.loads((D/'SUPERVISION_SEPARATE.json').read_text());assert sha(supervision['COCO_source'])==supervision['COCO_source_sha256'];self.slots=manifest['rows'];self.labels=supervision['labels'];self.fit_slots=supervision['selected_FIT_slots'];self.cache=np.load(manifest['cache_path'],mmap_mode='r');coco=json.loads(Path(supervision['COCO_source']).read_text());self.annotations={a['id']:a for a in coco['annotations'] if not a.get('iscrowd')};assert len(self.fit_slots)==len(self.labels);self.slot_index={slot:i for i,slot in enumerate(self.fit_slots)}
 def __len__(self):return len(self.fit_slots)
 def __getitem__(self,i):
  cv2.setNumThreads(0);slot=self.fit_slots[i];row=self.slots[slot];label=self.labels[str(slot)];x,y,s=row['crop_native_xy_size'];g=mu.decode(mu.merge(mu.frPyObjects(self.annotations[label['annotation_id']]['segmentation'],1024,1024))).astype(bool);full_gt=int(g.sum());g=np.ascontiguousarray(g[y:y+s,x:x+s]);v=np.ascontiguousarray(valid(label['valid_path'],label['valid_sha256'])[y:y+s,x:x+s]);raw=np.ascontiguousarray(instances(row['cache_path'],row['cache_sha256'])[y:y+s,x:x+s]==row['query_id']);prior=-8+16*cv2.resize(raw.astype('float32'),(256,256),interpolation=cv2.INTER_AREA);point=(np.array(row['point_native_pixel_index'])-[x,y])*(1024/s);box=(np.array(row['box_native_inclusive'])-[x,y,x,y])*(1024/s)
  return dict(embedding=torch.from_numpy(self.cache[slot].copy()),point=torch.tensor(point,dtype=torch.float32),box=torch.tensor(box,dtype=torch.float32),prior=torch.from_numpy(prior[None]),target=torch.from_numpy(g[None]),valid=torch.from_numpy(v[None]),raw=torch.from_numpy(raw[None]),enabled=torch.tensor(bool((g&v).any())),size=s,index=i,slot=slot,GT_crop_fraction=float(g.sum()/max(1,full_gt)))
def collate(items):
 maximum=max(r['size'] for r in items);out={k:torch.stack([r[k] for r in items]) for k in ['embedding','point','box','prior','enabled']}
 for name in ['target','valid','raw']:out[name]=torch.stack([torch.nn.functional.pad(r[name],(0,maximum-r['size'],0,maximum-r['size']),value=False) for r in items])
 out.update(size=torch.tensor([r['size'] for r in items]),index=torch.tensor([r['index'] for r in items]),slot=torch.tensor([r['slot'] for r in items]),GT_crop_fraction=torch.tensor([r['GT_crop_fraction'] for r in items]));return out
