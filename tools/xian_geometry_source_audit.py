"""Audit the three-epoch Geometry sources actually used by the rasterizer.

Writes:
  F:\\xian\\西安_当前最佳候选交付_20260917\\00_handoff\\GEOMETRY_SOURCE_PATHS.csv
  F:\\xian\\西安_当前最佳候选交付_20260917\\00_handoff\\GEOMETRY_SOURCE_PATHS.json

Fields: epoch, absolute_path, file_size, feature_count, geometry_type, crs,
sha256, plus the expected object count and a hard equality gate.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parents[0] / "work" / "python-packages"))
spec = importlib.util.spec_from_file_location("xian_raster_core", TOOLS / "xian_rasterize_current_best.py")
core = importlib.util.module_from_spec(spec)
sys.modules["xian_raster_core"] = core
spec.loader.exec_module(core)

import pyogrio  # noqa: E402

HANDOFF = core.DELIVERY / "00_handoff"
EXPECTED = {"2018": 195_140, "2019": 182_193, "2020": 281_417}


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    rows = []
    gate = True
    for year, (scene, date, filename) in core.SCENES.items():
        resolved = core.geom_path(filename)
        primary = core.GEOM_DIR / filename
        info = pyogrio.read_info(resolved)
        raw_fields = info.get("fields")
        fields = [] if raw_fields is None else [str(x) for x in raw_fields]
        count = int(info["features"])
        expected = EXPECTED[year]
        ok = count == expected
        gate = gate and ok
        rows.append(
            {
                "epoch": year,
                "scene_key": scene,
                "image_date": date,
                "absolute_path": str(resolved),
                "resolved_by": "GEOM_ASCII(transferred)" if resolved.parent == core.GEOM_ASCII else "GEOM_DIR(primary)",
                "primary_copy": str(primary),
                "primary_copy_exists": bool(primary.exists()),
                "file_size": resolved.stat().st_size,
                "feature_count": count,
                "expected_feature_count": expected,
                "count_gate": "PASS" if ok else "FAIL",
                "geometry_type": str(info.get("geometry_type")),
                "crs": str(info.get("crs")),
                "field_count": len(fields),
                "has_object_id": "object_id" in fields,
                "has_area_m2": "area_m2" in fields,
                "all_fields": "|".join(fields),
                "last_write_time": resolved.stat().st_mtime,
                "sha256": sha256(resolved),
            }
        )
    out_csv = HANDOFF / "GEOMETRY_SOURCE_PATHS.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (HANDOFF / "GEOMETRY_SOURCE_PATHS.json").write_text(
        json.dumps(
            {
                "source_selection_rule": (
                    "core.geom_path(): prefer F:\\xian\\_xian_geom_transfer\\<file> when it exists and is "
                    "non-empty, else F:\\xian\\西安_三期建筑追踪与三态最终正式候选版_20260916\\02_三期建筑几何\\remote_full_geometry\\<file>"
                ),
                "OBJECT_COUNT_GATE": "PASS" if gate else "FAIL",
                "candidate_status": "CANDIDATE_NOT_GROUND_TRUTH",
                "epochs": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("OBJECT_COUNT_GATE =", "PASS" if gate else "FAIL")
    for r in rows:
        print(r["epoch"], r["feature_count"], "exp", r["expected_feature_count"], r["count_gate"], r["absolute_path"])


if __name__ == "__main__":
    main()
