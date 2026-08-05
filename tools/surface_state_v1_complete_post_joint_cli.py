from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"extensions"))
from surface_state_v1.training.complete_post_joint import complete_post_joint

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--joint-run",required=True)
    parser.add_argument("--pipeline-dir",required=True)
    args=parser.parse_args()
    print(json.dumps(complete_post_joint(repo=REPO,joint_run=args.joint_run,pipeline_dir=args.pipeline_dir),ensure_ascii=False,indent=2))
    return 0
if __name__=="__main__": raise SystemExit(main())