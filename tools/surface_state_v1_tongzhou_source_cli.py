from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.integration.tongzhou_source import inventory_tongzhou_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Tongzhou source scenes")
    parser.add_argument("--source-root", default=r"D:\Data Set\Satellite\beijing_tongzhou")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = inventory_tongzhou_source(args.source_root, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
