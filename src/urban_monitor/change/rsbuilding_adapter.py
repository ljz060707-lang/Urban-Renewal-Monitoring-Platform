from __future__ import annotations

import json
from pathlib import Path

from .geospatial import grid_from_dataset, same_grid


EXPECTED = {
    "building_t1_probability": "building_t1_probability.tif",
    "building_t2_probability": "building_t2_probability.tif",
    "building_t1_mask": "building_t1_mask.tif",
    "building_t2_mask": "building_t2_mask.tif",
}


def load_rsbuilding_exports(scene_dir: str | Path, reference_path: str | Path, report_path: str | Path):
    import rasterio

    scene = Path(scene_dir)
    result = {}
    report = {"scene": str(scene.resolve()), "reference": str(Path(reference_path).resolve()), "outputs": {}}
    with rasterio.open(reference_path) as reference:
        report["reference_grid"] = grid_from_dataset(reference).__dict__
        for name, filename in EXPECTED.items():
            path = scene / filename
            if not path.is_file():
                report["outputs"][name] = {"status": "pending", "path": str(path)}
                continue
            with rasterio.open(path) as source:
                aligned = same_grid(reference, source)
                report["outputs"][name] = {
                    "status": "ready" if aligned else "alignment_required",
                    "path": str(path.resolve()),
                    "grid": grid_from_dataset(source).__dict__,
                }
                if not aligned:
                    raise ValueError(
                        f"RSBuilding output is not aligned with input grid: {path}; "
                        "run explicit geospatial reprojection before fusion"
                    )
                result[name] = source.read(1)
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

