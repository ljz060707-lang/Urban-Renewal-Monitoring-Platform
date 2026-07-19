import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pycocotools import mask as mask_utils
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from urban_monitor.coco import build_spatial_coco
from urban_monitor.splitting import build_spatial_blocks


def _prepared(root: Path):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    image = np.full((3, 128, 128), 100, dtype=np.uint8)
    with rasterio.open(
        root / "images" / "2024-01-01.tif",
        "w", driver="GTiff", height=128, width=128, count=3, dtype="uint8",
        crs="EPSG:32650", transform=from_origin(0, 128, 1, 1),
    ) as target:
        target.write(image)
    courtyard = Polygon([(0, 96), (32, 96), (32, 128), (0, 128)], holes=[[(8, 104), (24, 104), (24, 120), (8, 120)]])
    labels = gpd.GeoDataFrame(
        {
            "object_id": ["courtyard", "bottom_left", "bottom_right", "top_right"],
            "source_date": ["2024-01-01"] * 4,
        },
        geometry=[courtyard, box(0, 0, 32, 32), box(64, 0, 96, 32), box(64, 96, 96, 128)],
        crs="EPSG:32650",
    )
    labels.to_file(root / "labels" / "2024-01-01.gpkg", driver="GPKG")


def test_spatial_coco_is_stable_and_preserves_hole(tmp_path):
    prepared = tmp_path / "prepared"
    _prepared(prepared)
    blocks = build_spatial_blocks((0, 0, 128, 128), "EPSG:32650", block_size_m=64, validation_fraction=.25, seed=42)
    first = build_spatial_coco(
        prepared, blocks, ["2024-01-01"], tile_size=32, overlap=0,
        min_object_visibility=.20, max_empty_to_positive_ratio=1,
    )
    train_first = json.loads((prepared / "coco" / "annotations_train.json").read_text(encoding="utf-8"))
    val_first = json.loads((prepared / "coco" / "annotations_val.json").read_text(encoding="utf-8"))
    ids_first = ([item["id"] for item in train_first["images"] + val_first["images"]],
                 [item["id"] for item in train_first["annotations"] + val_first["annotations"]])
    build_spatial_coco(
        prepared, blocks, ["2024-01-01"], tile_size=32, overlap=0,
        min_object_visibility=.20, max_empty_to_positive_ratio=1,
    )
    train_second = json.loads((prepared / "coco" / "annotations_train.json").read_text(encoding="utf-8"))
    val_second = json.loads((prepared / "coco" / "annotations_val.json").read_text(encoding="utf-8"))
    ids_second = ([item["id"] for item in train_second["images"] + val_second["images"]],
                  [item["id"] for item in train_second["annotations"] + val_second["annotations"]])
    assert ids_first == ids_second
    assert first["splits"]["train"]["images"] and first["splits"]["val"]["images"]
    all_annotations = train_first["annotations"] + val_first["annotations"]
    courtyard = next(item for item in all_annotations if item["object_id"] == "courtyard" and item["visibility"] > .9)
    encoded = dict(courtyard["segmentation"])
    encoded["counts"] = encoded["counts"].encode("ascii")
    mask = mask_utils.decode(encoded)
    assert mask.sum() < 32 * 32 and mask[16, 16] == 0
    assert len(list((prepared / "coco" / "previews").glob("*.jpg"))) == 20


def test_train_and_val_tiles_share_no_block(tmp_path):
    prepared = tmp_path / "prepared"
    _prepared(prepared)
    blocks = build_spatial_blocks((0, 0, 128, 128), "EPSG:32650", block_size_m=64, validation_fraction=.25, seed=42)
    build_spatial_coco(prepared, blocks, ["2024-01-01"], tile_size=32, overlap=16, min_object_visibility=.2)
    train = json.loads((prepared / "coco" / "annotations_train.json").read_text(encoding="utf-8"))
    val = json.loads((prepared / "coco" / "annotations_val.json").read_text(encoding="utf-8"))
    assert {item["block_id"] for item in train["images"]}.isdisjoint({item["block_id"] for item in val["images"]})


def test_empty_label_date_can_supply_hard_negative_tiles(tmp_path):
    prepared = tmp_path / "prepared"
    _prepared(prepared)
    source = prepared / "images" / "2024-01-01.tif"
    target = prepared / "images" / "2025-01-01.tif"
    target.write_bytes(source.read_bytes())
    gpd.GeoDataFrame({"object_id": []}, geometry=[], crs="EPSG:32650").to_file(
        prepared / "labels" / "2025-01-01.gpkg", driver="GPKG"
    )
    blocks = build_spatial_blocks((0, 0, 128, 128), "EPSG:32650", block_size_m=64, validation_fraction=.25, seed=42)
    build_spatial_coco(
        prepared, blocks, ["2024-01-01", "2025-01-01"], tile_size=32, overlap=0,
        min_object_visibility=.2, include_empty_tiles=True, max_empty_to_positive_ratio=1,
    )
    payloads = [
        json.loads((prepared / "coco" / f"annotations_{split}.json").read_text(encoding="utf-8"))
        for split in ("train", "val")
    ]
    assert any(item["date"] == "2025-01-01" for payload in payloads for item in payload["images"])


def test_dataset_reads_window_generated_real_tile(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    from urban_monitor.dataset import CocoBuildingDataset

    prepared = tmp_path / "prepared"
    _prepared(prepared)
    blocks = build_spatial_blocks((0, 0, 128, 128), "EPSG:32650", block_size_m=64, validation_fraction=.25, seed=42)
    build_spatial_coco(prepared, blocks, ["2024-01-01"], tile_size=32, overlap=0, min_object_visibility=.2)
    dataset = CocoBuildingDataset(prepared / "coco", "annotations_train.json")
    image, target = dataset[0]
    assert tuple(image.shape) == (3, 32, 32)
    assert target["masks"].shape[-2:] == (32, 32)
