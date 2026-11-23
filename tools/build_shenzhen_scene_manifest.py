"""Build a four-scene Shenzhen manifest without guessing tile acquisition dates."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def load_mode_date(root: Path, epoch_index: int, task_name: str) -> dict:
    receipt = root / "01_data_registry" / "date_coverage" / f"epoch_{epoch_index:02d}_{task_name}" / "DATE_COVERAGE_RECEIPT.json"
    if not receipt.is_file():
        return {"status": "missing", "path": str(receipt)}
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    summary = root / "01_data_registry" / "date_coverage" / f"epoch_{epoch_index:02d}_{task_name}" / "acquisition_date_summary.csv"
    rows = list(csv.DictReader(summary.open(encoding="utf-8-sig"))) if summary.is_file() else []
    known = [row for row in rows if row.get("date_known") == "True"]
    mode = max(known, key=lambda row: float(row.get("area_share") or 0), default=None)
    return {
        "status": "pass",
        "receipt": str(receipt),
        "mode_date": mode.get("acquisition_date") if mode else "DATE_UNKNOWN",
        "mode_date_area_share": float(mode.get("area_share") or 0) if mode else 0.0,
        "known_area_share": float(sum(float(row.get("area_share") or 0) for row in known)),
        "unknown_area_share": float(sum(float(row.get("area_share") or 0) for row in rows if row.get("date_known") != "True")),
        "date_source": "GEID_tile_log_area_summary",
        "date_unknown_policy": "preserve_unknown_no_nominal_fill",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.epoch_manifest.read_text(encoding="utf-8"))
    scenes = []
    seen = set()
    for epoch in manifest.get("epochs", []):
        epoch_index = int(epoch["epoch_index"])
        task_name = str(epoch["task_name"])
        nominal = str(epoch.get("nominal_input_date") or "")
        if nominal in seen:
            raise SystemExit(f"duplicate nominal scene date: {nominal}")
        seen.add(nominal)
        source = Path(epoch["tif_path"])
        coverage = load_mode_date(args.output_root, epoch_index, task_name)
        scenes.append(
            {
                "scene_date": nominal,
                "epoch_id": f"epoch_{epoch_index:02d}_{task_name}",
                "nominal_input_date": nominal,
                "actual_mode_date": coverage.get("mode_date", "DATE_UNKNOWN"),
                "actual_mode_date_area_share": coverage.get("mode_date_area_share"),
                "known_date_area_share": coverage.get("known_area_share"),
                "unknown_date_area_share": coverage.get("unknown_area_share"),
                "selected_source": str(source),
                "source_sha256": epoch.get("source_sha256"),
                "source_exists": source.is_file(),
                "bands": epoch.get("bands"),
                "crs": epoch.get("crs"),
                "transform": epoch.get("transform"),
                "width": epoch.get("width"),
                "height": epoch.get("height"),
                "selection_reason": "GEID completion receipt; nominal epoch label is storage/order key only",
                "date_policy": "per_tile actual date when unambiguous; DATE_UNKNOWN otherwise; nominal date never substituted",
                "scientific_status": "candidate_not_ground_truth",
            }
        )
    scenes.sort(key=lambda row: row["scene_date"])
    if len(scenes) != 4 or not all(row["source_exists"] for row in scenes):
        raise SystemExit("four unique existing scenes are required")
    payload = {
        "schema_version": "urrs.shenzhen.scene_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene_discovery_status": "pass",
        "source_root": str(Path(r"D:\Data Set").resolve()),
        "target_dates": [row["scene_date"] for row in scenes],
        "scenes": scenes,
        "actual_date_source": "01_data_registry/date_coverage",
        "date_unknown_policy": "unknown_stays_DATE_UNKNOWN",
        "raw_tif_policy": "immutable_read_only",
        "test_pixel_reads": 0,
        "scientific_status": "candidate_not_ground_truth",
    }
    target = args.output_root / "SHENZHEN_SCENES.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing scene manifest: {target}")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(target), "dates": payload["target_dates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
