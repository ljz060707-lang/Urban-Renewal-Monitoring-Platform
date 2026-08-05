from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))
from surface_state_v1.training.runtime_compat import train

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=["fusu-train", "temporal-train", "joint-train"])
parser.add_argument("--config", required=True)
parser.add_argument("--run-dir")
args = parser.parse_args()
print(json.dumps(train(REPO / args.config, REPO, run_dir=args.run_dir), ensure_ascii=False, indent=2, default=str))
