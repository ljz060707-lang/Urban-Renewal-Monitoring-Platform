"""Make the report chart dataset scalar while preserving audited summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    for name in ["artifact_main.json", "artifact_construction.json", "artifact_height.json"]:
        path = root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        datasets = payload["snapshot"]["datasets"]
        datasets["height_models"] = [
            {key: value for key, value in row.items() if isinstance(value, (str, int, float, bool)) or value is None}
            for row in datasets.get("height_models", [])
        ]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "REPORT_CHART_DATASET_FLATTENED", "output_root": str(root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
