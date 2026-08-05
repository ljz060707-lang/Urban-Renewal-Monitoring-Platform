from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from urban_renewal.review.construction_inventory_v2 import build_construction_inventory
from urban_renewal.review.transition_preview_v2 import DATES, regenerate_transition_previews


def _write_scene(path: Path, date_index: int) -> str:
    data = np.empty((3, 16, 16), dtype=np.uint16)
    base = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)
    for band in range(3):
        data[band] = base + 100 + date_index * 400 + band * 25
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=3,
        dtype="uint16",
        crs="EPSG:32650",
        transform=from_origin(0, 16, 1, 1),
        nodata=0,
    ) as destination:
        destination.write(data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _make_project(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path
    manifest_dir = project_root / "outputs" / "inference" / "beijing_tongzhou_building_geometry"
    manifest_dir.mkdir(parents=True)
    scenes = []
    for date_index, date in enumerate(DATES):
        source = tmp_path / f"scene_{date}.tif"
        digest = _write_scene(source, date_index)
        scenes.append(
            {
                "scene_date": date,
                "selected_source": str(source),
                "bands": 3,
                "source_sha256": digest,
            }
        )
    (manifest_dir / "TONGZHOU_SCENES.json").write_text(
        json.dumps({"scene_discovery_status": "pass", "scenes": scenes}),
        encoding="utf-8",
    )
    review_dir = project_root / "review"
    review_dir.mkdir()
    geometry_records = []
    for index in range(16):
        for date in DATES:
            geometry_records.append(
                {
                    "sample_id": f"geometry_{date}_{index:02d}",
                    "review_group": "geometry",
                    "source_paths": {"scene": str(tmp_path / f"scene_{date}.tif")},
                }
            )
    (review_dir / "REVIEW_INVENTORY.json").write_text(
        json.dumps({"records": geometry_records}, ensure_ascii=False),
        encoding="utf-8",
    )
    return project_root, review_dir


def test_manifest_backed_transition_preview_is_date_specific_and_gate_safe(tmp_path):
    project_root, review_dir = _make_project(tmp_path)
    output_dir = tmp_path / "transition_preview_rebuild"
    result = regenerate_transition_previews(project_root, review_dir, output_dir)

    assert result["transition_count"] == 96
    assert result["valid_transition_preview_count"] == 96
    assert result["invalid_transition_preview_count"] == 0
    assert result["review_allowed_count"] == 0
    assert len(list((output_dir / "raw").glob("*.png"))) == 64
    assert len(list((output_dir / "previews").glob("*.png"))) == 96
    assert (output_dir / "TRANSITION_PREVIEW_VALIDATION.csv").is_file()
    assert (output_dir / "INVALID_PREVIEW_SAMPLES.csv").is_file()

    inventory = json.loads((output_dir / "TRANSITION_REVIEW_INVENTORY.json").read_text(encoding="utf-8"))
    assert {row["transition"] for row in inventory["records"]} == {
        "20180506_to_20230716",
        "20200723_to_20230716",
        "20220320_to_20230716",
        "20180506_to_20200723",
        "20200723_to_20220320",
        "20220320_to_20230716_construction",
    }
    assert {row["source_metadata"]["before"]["rgb_band_selection_source"] for row in inventory["records"]} == {
        "manifest_band_count_three_default"
    }
    assert all(row["preview_quality"]["nonzero_fraction"] > 0 for row in inventory["records"])
    source_hashes = {
        row["source_metadata"][side]["source_sha256"]
        for row in inventory["records"]
        for side in ("before", "after")
    }
    assert len(source_hashes) == 4
    assert all(row["review_allowed"] is False for row in inventory["records"])
    assert all("full20" not in row["evidence_missing"] for row in inventory["records"] if not row["full20_supported"])

    before_inventory = (review_dir / "REVIEW_INVENTORY.json").read_bytes()
    assert (review_dir / "REVIEW_INVENTORY.json").read_bytes() == before_inventory


def test_construction_inventory_groups_regenerated_transitions(tmp_path):
    project_root, review_dir = _make_project(tmp_path)
    output_dir = tmp_path / "transition_preview_rebuild"
    regenerate_transition_previews(project_root, review_dir, output_dir)

    migration_dir = tmp_path / "review_migration_20260802_122123"
    migration_dir.mkdir()
    preview_dir = migration_dir / "transition_previews"
    preview_dir.mkdir()
    for path in output_dir.iterdir():
        if path.is_file():
            (preview_dir / path.name).write_bytes(path.read_bytes())
        elif path.name in {"raw", "previews"}:
            target = preview_dir / path.name
            target.mkdir()
            for child in path.iterdir():
                (target / child.name).write_bytes(child.read_bytes())
    inventory_path = preview_dir / "TRANSITION_REVIEW_INVENTORY.json"
    inventory_path.write_text(
        (output_dir / "TRANSITION_REVIEW_INVENTORY.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    inventory = build_construction_inventory(review_dir, {"records": []})
    assert inventory["track_count"] == 16
    assert inventory["records"][0]["dates"] == list(DATES)
    assert len(inventory["records"][0]["pair_preview_paths"]) == 6
    assert inventory["records"][0]["review_allowed"] is False
    assert inventory["records"][0]["scientific_status"] == "experimental_candidate"
