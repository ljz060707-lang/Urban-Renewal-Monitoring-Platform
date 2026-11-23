"""Create an immutable Shenzhen epoch manifest from the GEID completion receipt."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Data Set"))
    args = parser.parse_args()
    receipt = json.loads(args.completion_receipt.read_text(encoding="utf-8"))
    if receipt.get("four_tif_merge_complete") is not True:
        raise SystemExit("completion receipt does not assert four_tif_merge_complete=true")
    epochs = []
    for index, row in enumerate(receipt.get("tifs", [])):
        tif = Path(row["path"])
        task_name = str(row["task_name"])
        epochs.append(
            {
                "epoch_index": index,
                "task_name": task_name,
                "nominal_input_date": row.get("nominal_input_date"),
                "tif_path": str(tif),
                "source_dir": str(args.data_root / task_name),
                "source_sha256": row.get("sha256"),
                "width": row.get("width"),
                "height": row.get("height"),
                "bands": row.get("bands"),
                "crs": row.get("crs"),
                "transform": row.get("transform"),
                "source_policy": "immutable_read_only",
                "scientific_status": "candidate_not_ground_truth",
            }
        )
    if len(epochs) != 4:
        raise SystemExit(f"expected four epochs, found {len(epochs)}")
    payload = {
        "schema_version": "urrs.shenzhen.epoch_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completion_receipt": str(args.completion_receipt.resolve()),
        "completion_receipt_sha256": __import__("hashlib").sha256(
            args.completion_receipt.read_bytes()
        ).hexdigest(),
        "source_root": str(args.data_root.resolve()),
        "raw_tif_policy": "immutable_read_only_no_move_no_overwrite_no_retile",
        "epochs": epochs,
        "test_pixel_reads": 0,
    }
    target = args.output_root / "01_data_registry" / "GEID_EPOCH_MANIFEST.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {target}")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(target), "epoch_count": len(epochs)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
