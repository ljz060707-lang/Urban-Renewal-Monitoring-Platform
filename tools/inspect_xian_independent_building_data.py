#!/usr/bin/env python3
"""Read-only inventory of the local Xi'an building polygon layer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fiona
from shapely.geometry import shape


ROOT = Path(r"F:\xian")
SOURCE = ROOT / "西安建筑轮廓数据" / "西安.shp"
PRJ = SOURCE.with_suffix(".prj")
OUTPUT = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\18_reconstruction\independent_ground_inventory_01")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with fiona.open(SOURCE) as layer:
        profile = {"driver": layer.driver, "crs": str(layer.crs),
                   "crs_wkt": layer.crs_wkt, "schema": layer.schema,
                   "feature_count": len(layer), "bounds": list(layer.bounds)}
        samples = []
        for index, feature in enumerate(layer):
            if index == 10:
                break
            geom = shape(feature["geometry"]) if feature.get("geometry") else None
            samples.append({"fid": feature["id"], "properties": dict(feature["properties"]),
                            "geometry_type": geom.geom_type if geom else None,
                            "area_in_source_units": geom.area if geom else None,
                            "bounds_in_source_crs": list(geom.bounds) if geom else None})
    files = [SOURCE, SOURCE.with_suffix(".dbf"), SOURCE.with_suffix(".shx"), PRJ]
    result = {"schema_version": "xian_r5.local_building_layer_inventory.v1",
              "source": str(SOURCE), "source_sha256": {str(path): sha256(path) for path in files if path.is_file()},
              "projection_file_text": PRJ.read_text(encoding="utf-8", errors="replace"),
              "profile": profile, "first_10_features": samples,
              "classification": "SOURCE_SEMANTICS_NOT_YET_ESTABLISHED",
              "ground_truth_status": "NOT_ASSUMED"}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "INVENTORY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"feature_count": result["profile"]["feature_count"],
                      "crs": result["profile"]["crs"], "schema": result["profile"]["schema"],
                      "classification": result["classification"],
                      "output": str(OUTPUT)}, ensure_ascii=False))
