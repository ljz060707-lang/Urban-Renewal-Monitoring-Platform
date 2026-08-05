from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.integration.tongzhou_acceptance import audit_tongzhou


def main() -> int:
    parser = argparse.ArgumentParser(description="Tongzhou SurfaceState v1 acceptance")
    parser.add_argument("--project", default=str(REPO))
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit_tongzhou(args.project, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
