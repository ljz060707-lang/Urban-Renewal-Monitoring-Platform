"""Audit existing Nanjing/Tongzhou static outputs after candidate freeze."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def inspect_root(root: Path) -> dict[str, Any]:
    files = [p for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
    names = [p.name.lower() for p in files]
    return {"root": str(root), "exists": root.is_dir(), "file_count": len(files), "probability_count": sum("probability" in n for n in names), "mask_count": sum(n.endswith("mask.tif") for n in names), "gpkg_count": sum(n.endswith(".gpkg") for n in names), "diagnostic_only": True, "used_for_selection": False, "test_used": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--nanjing-root", type=Path, default=Path(r"D:\Data Set\Satellite\nanjing_jiangbei 20"))
    parser.add_argument("--tongzhou-root", type=Path, default=Path(r"D:\Data Set\Satellite\beijing_tongzhou"))
    args = parser.parse_args()
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), "candidate_frozen_before_diagnostics": True, "nanjing": inspect_root(args.nanjing_root), "tongzhou": inspect_root(args.tongzhou_root), "selection_data_used": False, "threshold_selection_used": False, "test_pixel_reads": 0, "scientific_status": "diagnostic_proxy_not_accuracy"}
    write_json(args.run_dir / "08_diagnostics" / "REAL_SCENE_STATIC_DIAGNOSTICS.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
