from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from ..datasets.cwld import CWLDManifestDataset
from ..datasets.cwld_codec import CWLDCodec
from ..models.surface_state_net import SurfaceStateNet

def _sha(path):
 h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()
def smoke_train_cwld(config, output):
 output=Path(output); output.mkdir(parents=True,exist_ok=False); device=torch.device(config.get("device","cuda:0") if torch.cuda.is_available() else "cpu")
 codec=CWLDCodec(**config["codec"]); dataset=CWLDManifestDataset(config["manifest"],"train",codec,config["image_size"]); loader=DataLoader(dataset,batch_size=config["batch_size"],shuffle=True,num_workers=0,pin_memory=device.type=="cuda")
 model=SurfaceStateNet(cwld_classes=4).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"]); loss_fn=nn.CrossEntropyLoss(ignore_index=255); model.train(); steps=0; losses=[]
 while steps<config["smoke_steps"]:
  for batch in loader:
   x=batch["image"].to(device); y=batch["target"].to(device); z=model.forward_cwld(x)["logits"]; loss=loss_fn(z,y)
   if not torch.isfinite(loss): raise RuntimeError("non-finite CWLD smoke loss")
   optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss)); steps+=1
   if steps>=config["smoke_steps"]: break
 model.eval(); val=CWLDManifestDataset(config["manifest"],"validation",codec,config["image_size"]); batch=next(iter(DataLoader(val,batch_size=min(config["batch_size"],len(val))))); probs=model.forward_cwld(batch["image"].to(device))["probabilities"]
 if not torch.isfinite(probs).all() or probs.min()<0 or probs.max()>1: raise RuntimeError("invalid probability range")
 ckpt=output/"smoke_last.pth"; torch.save({"model_state_dict":model.state_dict(),"cwld_classes":4,"codec":config["codec"],"smoke_only":True},ckpt)
 result={"status":"success","steps":steps,"mean_loss":sum(losses)/len(losses),"device":str(device),"batch_shape":list(batch["image"].shape),"probability_range":[float(probs.min()),float(probs.max())],"checkpoint":str(ckpt.resolve()),"checkpoint_sha256":_sha(ckpt),"test_read_count":0,"formal_model":False,"gpu_memory_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0}
 (output/"SMOKE_RECEIPT.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result
