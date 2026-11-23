from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,hashlib
from functools import lru_cache
import numpy as np,torch,cv2
from pycocotools import mask as mu
R=Path('/root/xian20261111_r15');D=R/'data/native_roof_decoder'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@lru_cache(maxsize=128)
def embedding(path,digest):
 assert sha(path)==digest;return np.load(path)['image_embedding'][0]
@lru_cache(maxsize=64)
def instances(path,digest):
 assert sha(path)==digest;return np.load(path)['instances']
@lru_cache(maxsize=128)
def valid_map(path,digest):
 assert sha(path)==digest;return np.load(path)['valid']
class NativeRoofDataset(torch.utils.data.Dataset):
 def __init__(self,data,role):
  self.rows=[r for r in data['samples'] if r['role']==role];assert self.rows and all(r['loss_enabled'] and not r['GT_in_model_prompt'] for r in self.rows);assert sha(data['COCO_source'])==data['COCO_source_sha256'];coco=json.loads(Path(data['COCO_source']).read_text());self.annotations={a['id']:a for a in coco['annotations'] if not a.get('iscrowd')};assert all(r['annotation_id'] in self.annotations for r in self.rows)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  cv2.setNumThreads(0);r=self.rows[i];a=self.annotations[r['annotation_id']];g=mu.decode(mu.merge(mu.frPyObjects(a['segmentation'],1024,1024))).astype(bool);v=valid_map(r['valid_path'],r['valid_sha256']);raw=instances(r['cache_path'],r['cache_sha256'])==r['query_id'];prior=-8+16*cv2.resize(raw.astype('float32'),(256,256),interpolation=cv2.INTER_AREA);roi=np.zeros((1024,1024),bool);x0,y0,x1,y1=r['box_native_inclusive'];roi[max(0,int(y0)-64):min(1024,int(y1)+65),max(0,int(x0)-64):min(1024,int(x1)+65)]=True;roi|=g
  assert (g&v).any();return dict(embedding=torch.from_numpy(embedding(r['embedding_path'],r['embedding_sha256'])),point=torch.tensor(r['point_native_pixel_index'],dtype=torch.float32),box=torch.tensor(r['box_native_inclusive'],dtype=torch.float32),prior=torch.from_numpy(prior[None]),target=torch.from_numpy(g[None]),valid=torch.from_numpy(v[None]),roi=torch.from_numpy(roi[None]),sample_index=torch.tensor(i))
