"""Non-invasive evidence overlay; deliberately never imports RSBuilding."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for part in iter(lambda:f.read(1024*1024),b""): h.update(part)
 return h.hexdigest()

class UrbanRenewalMasterOverlay:
 def __init__(self,full20_checkpoint,rsbuilding_checkpoint,surface_checkpoint=None):
  self.full20_checkpoint=Path(full20_checkpoint); self.rsbuilding_checkpoint=Path(rsbuilding_checkpoint); self.surface_checkpoint=Path(surface_checkpoint) if surface_checkpoint else None
 def audit(self):
  return {"integration_mode":"non_invasive_overlay","integration_status":"overlay_candidate","main_model_merge_status":"pending_safe_patch_capability","scientific_status":"candidate_not_ground_truth","full20":{"path":str(self.full20_checkpoint),"sha256":sha256(self.full20_checkpoint)},"rsbuilding":{"path":str(self.rsbuilding_checkpoint),"sha256":sha256(self.rsbuilding_checkpoint)},"surface_checkpoint":str(self.surface_checkpoint) if self.surface_checkpoint else None,"rsbuilding_imported":False,"whu_test_reads":0,"second_test_reads":0}
 def write_manifest(self,output):
  output=Path(output); output.mkdir(parents=True,exist_ok=False); result=self.audit(); (output/"MASTER_OVERLAY_MANIFEST.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result
