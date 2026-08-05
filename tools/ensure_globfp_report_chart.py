"""Ensure generated report artifacts contain the required native chart block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CHART = {
    "id": "height_model_mae_comparison",
    "type": "bar",
    "title": "Height model validation MAE (m)",
    "dataset": "height_models",
    "sourceId": "height_validation",
    "encodings": {
        "x": {"field": "model", "type": "nominal"},
        "y": {"field": "MAE_m", "type": "quantitative"},
        "tooltip": [
            {"field": "model"},
            {"field": "MAE_m"},
            {"field": "RMSE_m"},
        ],
    },
}
BLOCK = {"id": "height_model_chart_block", "type": "chart", "chartId": CHART["id"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    for name in ["artifact_main.json", "artifact_construction.json", "artifact_height.json"]:
        path = root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = payload.setdefault("manifest", {})
        charts = manifest.setdefault("charts", [])
        charts[:] = [item for item in charts if item.get("id") != CHART["id"]]
        charts.append(CHART)
        blocks = manifest.setdefault("blocks", [])
        blocks[:] = [item for item in blocks if item.get("id") != BLOCK["id"]]
        insert_at = next((i for i, item in enumerate(blocks) if item.get("id") == "report_body"), len(blocks))
        blocks.insert(insert_at, BLOCK)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "REPORT_CHART_BLOCKS_READY", "output_root": str(root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
