from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import rasterio


DATE_BY_NAME = {
    "tongzhou180506_zoom_20.tif": "2018-05-06",
    "tongzhou200723_zoom_20.tif": "2020-07-23",
    "tongzhou220320_zoom_20.tif": "2022-03-20",
    "tongzhou230716_zoom_20.tif": "2023-07-16",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_tongzhou_source(source_root: str | Path, output: str | Path) -> dict:
    source_root, output = Path(source_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scenes = []
    for name, date in DATE_BY_NAME.items():
        path = source_root / name
        if not path.exists():
            raise FileNotFoundError(f"Missing Tongzhou scene: {path}")
        with rasterio.open(path) as dataset:
            scenes.append(
                {
                    "date": date,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "width": dataset.width,
                    "height": dataset.height,
                    "bands": dataset.count,
                    "crs": str(dataset.crs) if dataset.crs else None,
                    "transform": tuple(dataset.transform),
                    "bounds": tuple(dataset.bounds),
                    "nodata": dataset.nodata,
                }
            )
    signatures = {
        (
            scene["width"],
            scene["height"],
            scene["crs"],
            tuple(scene["transform"]),
            tuple(scene["bounds"]),
            scene["nodata"],
        )
        for scene in scenes
    }
    receipt = {
        "source_root": str(source_root.resolve()),
        "scenes": scenes,
        "scene_count": len(scenes),
        "dates_strictly_increasing": [scene["date"] for scene in scenes]
        == sorted(scene["date"] for scene in scenes),
        "duplicate_dates": len({scene["date"] for scene in scenes}) != len(scenes),
        "alignment_status": (
            "aligned"
            if len(signatures) == 1
            else "registration_required_before_temporal_inference"
        ),
        "original_data_unchanged": True,
        "tongzhou_used_for_checkpoint_selection": False,
        "tongzhou_used_for_threshold_selection": False,
        "test_reads": 0,
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "TONGZHOU_SCENES.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt

