"""Attach immutable GloBFP audit provenance to a v2 run."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.audit_dir / "01_globfp_audit"
    target = args.run_dir / "01_globfp_audit"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("GLOBFP_READINESS.json", "GLOBFP_SCHEMA_AUDIT.json", "GLOBFP_HEIGHT_AUDIT.json", "VECTOR_RASTER_INVENTORY.csv", "FILE_INVENTORY.csv"):
        path = source / name
        if path.exists(): shutil.copy2(path, target / name)
    hash_inventory = Path(r"D:\Codex\projects\Documents-Codex\2026-07-18\urban-renewal-phase1-starter-v0-2\outputs\urban-renewal-phase1-starter\outputs\data_preparation\globfp_china_nanjing_20260804_120325\00_globfp_audit\SHA256_INVENTORY.csv")
    if hash_inventory.exists(): shutil.copy2(hash_inventory, target / "SHA256_INVENTORY_REUSED_FROM_PRIOR_AUDIT.csv")
    readiness = json.loads((source / "GLOBFP_READINESS.json").read_text(encoding="utf-8"))
    gate = {"status": "blocked_height_supervision", "height_fields": readiness.get("height_fields_observed", []), "height_unit_status": readiness.get("height_unit_status"), "image_pairs": 0, "height_training_allowed": False, "reason": "vector-only C_FOOTPRINT_HEIGHT_ONLY assets; no legal image/footprint pairing and no unit proof", "created_at": datetime.now(timezone.utc).isoformat(), "source_data_modified": False, "test_pixel_reads": 0}
    (args.run_dir / "03_height_audit").mkdir(parents=True, exist_ok=True)
    (args.run_dir / "03_height_audit" / "HEIGHT_SUPERVISION_GATE.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.run_dir / "01_globfp_audit" / "HASH_PROVENANCE.json").write_text(json.dumps({"status": "reused_prior_full_hash_inventory", "source": str(hash_inventory), "current_metadata_audit": str(source), "source_data_modified": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
