"""Budgeted feature cache: one shared image/mask crop path for all labels."""
from pathlib import Path
import math,importlib.util
import numpy as np
from shapely.geometry import shape
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.enums import Resampling
from affine import Affine
import rasterio
from .contracts import sha

def observation_crop(record,size=32):
    g=shape(record['native_geometry']);side=int(math.ceil(max(16.,max(g.bounds[2]-g.bounds[0],g.bounds[3]-g.bounds[1])*1.8)))
    x=math.floor(g.centroid.x-side/2);y=math.floor(g.centroid.y-side/2)
    with rasterio.open(record['image_path']) as ds:
        rgb=ds.read([1,2,3],window=Window(x,y,side,side),boundless=True,fill_value=0,out_shape=(3,size,size),resampling=Resampling.bilinear)
    # Rasterize directly on the same scaled pixel-edge frame; no class branch.
    mask=rasterize([(g,255)],out_shape=(size,size),transform=Affine.translation(x,y)*Affine.scale(side/size,side/size),fill=0,dtype='uint8')
    return np.concatenate([rgb,mask[None]],0).astype(np.float32)/255.

def frozen_encoder(old_root,device):
    import torch
    path=old_root/'source/temporal_model.py';spec=importlib.util.spec_from_file_location('frozen_previous_encoder',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    checkpoint=old_root/'runs/T2_FULL_SPAN/seed_20261027/best.pt';c=torch.load(checkpoint,map_location='cpu');model=m.TemporalIdentityNet();model.load_state_dict(c['model'],strict=True)
    encoder=model.encoder.to(device).eval()
    for p in encoder.parameters():p.requires_grad_(False)
    return encoder,dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),source_sha256=sha(path),frozen=True,feature_dim=64,crop_size=32,
        capacity_upgrade_claim=False,stage='candidate-task-only control, not encoder upgrade')

def extract_embeddings(graph,encoder,device,batch_size=64):
    import torch
    # This function may only be called after the explicit training budget gate.
    observations=graph['observations'];out={};verified=set()
    for i in range(0,len(observations),batch_size):
        batch=observations[i:i+batch_size]
        for r in batch:
            p=r['image_path']
            if not p or not r.get('image_sha256'):raise ValueError('Image path/hash missing from graph input')
            if p not in verified:
                if sha(p)!=r['image_sha256']:raise ValueError('Source image changed')
                verified.add(p)
        x=torch.from_numpy(np.stack([observation_crop(r) for r in batch])).to(device)
        with torch.no_grad():z=encoder(x).cpu().numpy()
        for r,v in zip(batch,z):out[r['uid']]=v
    return out
