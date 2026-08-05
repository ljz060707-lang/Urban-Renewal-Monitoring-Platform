import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"extensions"))
from surface_state_v1.integration.master_overlay import UrbanRenewalMasterOverlay
p=argparse.ArgumentParser();p.add_argument('--full20',required=True);p.add_argument('--rsbuilding',required=True);p.add_argument('--surface');p.add_argument('--output',required=True);a=p.parse_args()
print(UrbanRenewalMasterOverlay(a.full20,a.rsbuilding,a.surface).write_manifest(a.output))
