"""Package a frozen v2 candidate or an explicit baseline-retained result."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if not args.baseline_checkpoint.exists(): raise SystemExit(f"baseline missing: {args.baseline_checkpoint}")
    package = args.run_dir / "09_model_package" / "URRS-Building-Foundation-v2-best-static-candidate"
    package.mkdir(parents=True, exist_ok=True)
    target = package / "iter_2000_baseline_reference.pth"
    if not target.exists(): shutil.copy2(args.baseline_checkpoint, target)
    digest = sha256(target)
    status = {"status": "baseline_retained_no_material_improvement", "scientific_status": "baseline_retained_no_material_improvement", "checkpoint": str(target.resolve()), "checkpoint_sha256": digest, "source_checkpoint_sha256": digest, "test_used": False, "locked_test_pixel_reads": 0, "candidate_reason": "No eligible v2 candidate was promoted; project-native smoke is not source-checkpoint initialized", "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(args.run_dir / "12_acceptance" / "PROPOSED_CURRENT_BEST_REGISTRY_ENTRY.json", status)
    write_json(package / "MODEL_CONTRACT.json", {"outputs": ["building_probability", "building_mask", "building_boundary_probability", "building_distance_map", "building_quality_or_uncertainty", "building_objects"], "height_status": "not_available", "scientific_status": status["scientific_status"]})
    (package / "MODEL_CARD.md").write_text("# URRS Building Foundation v2\n\nThis package retains the frozen RSBuilding iter_2000 baseline because no eligible v2 candidate was promoted. It is a static building candidate pending final human review, not Ground Truth or national production validation.\n", encoding="utf-8")
    resolved_config = args.run_dir / "resolved_config.yaml"
    if resolved_config.exists(): shutil.copy2(resolved_config, package / "CONFIG_SNAPSHOT.yaml")
    dataset_manifest = args.run_dir / "05_data" / "building_manifest.csv"
    if dataset_manifest.exists(): shutil.copy2(dataset_manifest, package / "DATASET_MANIFEST.csv")
    inference_tool = Path(__file__).resolve().parent / "infer_building_foundation_v2.py"
    if inference_tool.exists(): shutil.copy2(inference_tool, package / "infer_building_foundation_v2.py")
    (package / "SHA256SUMS.txt").write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    write_json(package / "PACKAGE_RECEIPT.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
