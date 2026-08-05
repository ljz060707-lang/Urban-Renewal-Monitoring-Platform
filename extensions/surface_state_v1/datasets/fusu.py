from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class FUSUSampleDataset(Dataset):
    """500-pair package reader: smoke/development only, never formal validation."""
    def __init__(self, manifest,image_size=256):
        self.rows=list(csv.DictReader(Path(manifest).open(encoding="utf-8-sig"))); self.image_size=int(image_size)
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index]
        def read(p):
            with Image.open(p) as im: return torch.from_numpy(np.asarray(im.convert("RGB").resize((self.image_size,self.image_size),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/255)
        def mask(p):
            with Image.open(p) as im: return torch.from_numpy(np.asarray(im.resize((self.image_size,self.image_size),Image.Resampling.NEAREST),dtype=np.int64))
        return {"image_t1":read(r["image_t1"]),"image_t2":read(r["image_t2"]),"mask_t1":mask(r["mask_t1"]),"mask_t2":mask(r["mask_t2"]),"sample_id":r["sample_id"]}
