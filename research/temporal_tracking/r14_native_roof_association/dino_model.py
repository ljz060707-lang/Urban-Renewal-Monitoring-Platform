from pathlib import Path
import os,sys,json,hashlib
os.environ['XFORMERS_DISABLED']='1'
import torch
import torch.nn.functional as F
from native_temporal_contract import O,sha
D=O.parent/'external/dinov2'
def load_model():
 receipt=json.loads((D/'SOURCE_RECEIPT.json').read_text())
 for r in receipt['files']:assert sha(D/'source'/r['path'])==r['sha256']
 assert sha(receipt['weights']['path'])==receipt['weights']['sha256'];sys.path.insert(0,str(D/'source'))
 from dinov2.models.vision_transformer import vit_base
 model=vit_base(img_size=518,patch_size=14,init_values=1.,block_chunks=0,num_register_tokens=0,interpolate_antialias=False,interpolate_offset=.1);state=torch.load(receipt['weights']['path'],map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);model.eval()
 for p in model.parameters():p.requires_grad_(False)
 return model
@torch.no_grad()
def extract(model,rgba):
 rgb=F.interpolate(rgba[:,:3].float()/255,size=(224,224),mode='bilinear',align_corners=False);mean=rgb.new_tensor([.485,.456,.406])[None,:,None,None];std=rgb.new_tensor([.229,.224,.225])[None,:,None,None];result=model.forward_features((rgb-mean)/std);cls=F.normalize(result['x_norm_clstoken'].float(),dim=1);patch=result['x_norm_patchtokens'].float();mask=F.interpolate(rgba[:,3:4].float()/255,size=(16,16),mode='area').flatten(2).transpose(1,2);pooled=F.normalize((patch*mask).sum(1)/mask.sum(1).clamp_min(1e-8),dim=1);return cls,pooled
