from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .cwld_codec import CWLDCodec

class CWLDManifestDataset(Dataset):
    def __init__(self, manifest, split, codec:CWLDCodec, image_size=256):
        self.rows=[r for r in csv.DictReader(Path(manifest).open(encoding="utf-8-sig")) if r.get("split")==split]
        if not self.rows: raise ValueError(f"CWLD split has no samples: {split}")
        if any("test" in " ".join(r.values()).casefold() for r in self.rows): raise RuntimeError("Test split is prohibited")
        self.codec,self.image_size=codec,int(image_size)
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row["image_t1"]) as im: image=im.convert("RGB").resize((self.image_size,self.image_size),Image.Resampling.BILINEAR)
        target=self.codec.encode_file(row["mask_t1"])
        target=Image.fromarray(target).resize((self.image_size,self.image_size),Image.Resampling.NEAREST)
        return {"image":torch.from_numpy(np.asarray(image,dtype=np.float32).transpose(2,0,1)/255),"target":torch.from_numpy(np.asarray(target,dtype=np.int64)),"sample_id":row["sample_id"],"split":row["split"]}
