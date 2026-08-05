"""Independent extension CLI; does not modify the project's existing run.py."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"extensions"))
from surface_state_v1.audit import cwld_codec_audit,fusu_classification
from surface_state_v1.training.trainer import smoke_train_cwld
from surface_state_v1.datasets.fusu import FUSUSampleDataset
from surface_state_v1.models.surface_state_net import SurfaceStateNet
from surface_state_v1.models.tfr import TemporalFeatureRefinement
from surface_state_v1.models.temporal_change_head import TemporalChangeHead
from surface_state_v1.models.mti import MultiTaskIntegration
import torch

def main(argv=None):
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
 for name in ("cwld-audit","fusu-classify"): x=sub.add_parser(name);x.add_argument("--manifest",required=True);x.add_argument("--output",required=True)
 x=sub.add_parser("cwld-smoke");x.add_argument("--config",required=True);x.add_argument("--output",required=True)
 x=sub.add_parser("temporal-smoke");x.add_argument("--output",required=True)
 a=p.parse_args(argv)
 if a.cmd=="cwld-audit": r=cwld_codec_audit(a.manifest,a.output)
 elif a.cmd=="fusu-classify": r=fusu_classification(a.manifest,a.output)
 elif a.cmd=="cwld-smoke": r=smoke_train_cwld(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")),a.output)
 else:
  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); x=torch.rand(1,4,3,64,64,device=device,requires_grad=True); enc=SurfaceStateNet().to(device); f=enc.features(x.reshape(4,3,64,64)).reshape(1,4,24,64,64); tfr=TemporalFeatureRefinement(24).to(device); z=tfr(f,temporal_valid_mask=torch.tensor([[True,True,False,True]],device=device))["refined_features"][0]; change=TemporalChangeHead(24).to(device)(z); loss=change["dense_change_probability"].mean();loss.backward(); mti=MultiTaskIntegration()(torch.rand(1,4,64,64,device=device),change["dense_change_probability"]); r={"status":"success","tfr_shape":list(z.shape),"change_shape":list(change["dense_change_probability"].shape),"mti_shape":list(mti["optimal_building_sequence"].shape),"finite_loss":bool(torch.isfinite(loss)),"test_read_count":0,"formal_training":False}; Path(a.output).mkdir(parents=True,exist_ok=True); (Path(a.output)/"TEMPORAL_SMOKE_RECEIPT.json").write_text(json.dumps(r,indent=2),encoding="utf-8")
 print(json.dumps(r,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
