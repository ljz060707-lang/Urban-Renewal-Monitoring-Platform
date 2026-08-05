"""Audit grouped leakage for a Building Foundation v2 run manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    manifest = args.manifest or args.run_dir / "05_data" / "building_manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"manifest missing: {manifest}")
    rows = read_rows(manifest)
    by_hash: dict[str, set[str]] = defaultdict(set)
    by_label_hash: dict[str, set[str]] = defaultdict(set)
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row.get("split", "")
        if not split: continue
        # Image identity is the leakage-critical key.  A repeated binary mask
        # (for example an all-building or all-background tile) is recorded as
        # a diagnostic but is not treated as the same original scene.
        if row.get("image_sha256"): by_hash[row["image_sha256"]].add(split)
        if row.get("label_sha256"): by_label_hash[row["label_sha256"]].add(split)
        for key in ("group_id", "scene_id", "base_id"):
            if row.get(key): by_group[row[key]].add(split)
    hash_overlap = {k: sorted(v) for k, v in by_hash.items() if len(v) > 1}
    group_overlap = {k: sorted(v) for k, v in by_group.items() if len(v) > 1}
    label_hash_overlap = {k: sorted(v) for k, v in by_label_hash.items() if len(v) > 1}
    payload = {"status": "pass" if not hash_overlap and not group_overlap else "blocked_leakage", "manifest": str(manifest.resolve()), "row_count": len(rows), "split_counts": {s: sum(r.get("split") == s for r in rows) for s in {r.get("split") for r in rows}}, "hash_overlap_count": len(hash_overlap), "label_hash_overlap_count": len(label_hash_overlap), "group_overlap_count": len(group_overlap), "hash_overlaps": hash_overlap, "label_hash_overlaps_diagnostic": label_hash_overlap, "group_overlaps": group_overlap, "critical_leakage_count": len(hash_overlap) + len(group_overlap), "test_pixel_reads": 0, "source_data_modified": False}
    out = args.run_dir / "04_leakage" / "BUILDING_V2_LEAKAGE_AUDIT.json"
    write_json(out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
