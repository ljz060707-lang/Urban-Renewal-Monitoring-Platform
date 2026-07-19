import geopandas as gpd
from shapely.geometry import box

from urban_monitor.splitting import assign_geometry_splits, build_spatial_blocks, split_for_tile


def test_spatial_split_is_deterministic():
    first = build_spatial_blocks((0, 0, 1000, 1000), "EPSG:32650", block_size_m=250, seed=42)
    second = build_spatial_blocks((0, 0, 1000, 1000), "EPSG:32650", block_size_m=250, seed=42)
    assert first[["block_id", "split"]].to_dict("records") == second[["block_id", "split"]].to_dict("records")
    assert set(first["split"]) == {"train", "val"}


def test_same_spatial_block_is_same_split_across_dates():
    blocks = build_spatial_blocks((0, 0, 1000, 1000), "EPSG:32650", block_size_m=250, seed=42)
    labels = gpd.GeoDataFrame(
        {"source_date": ["2023-01-01", "2024-01-01"]},
        geometry=[box(100, 100, 120, 120), box(102, 102, 122, 122)],
        crs="EPSG:32650",
    )
    assigned = assign_geometry_splits(labels, blocks)
    assert assigned.block_id.nunique() == 1 and assigned.split.nunique() == 1


def test_boundary_crossing_tile_is_excluded():
    blocks = build_spatial_blocks((0, 0, 1000, 1000), "EPSG:32650", block_size_m=500, seed=42)
    assert split_for_tile(box(100, 100, 300, 300), blocks) is not None
    assert split_for_tile(box(400, 100, 600, 300), blocks) is None


def test_retained_train_and_val_tiles_do_not_overlap():
    blocks = build_spatial_blocks((0, 0, 1000, 1000), "EPSG:32650", block_size_m=500, seed=42)
    tiles = [box(x, y, x + 200, y + 200) for x in range(0, 801, 100) for y in range(0, 801, 100)]
    assigned = [(tile, split_for_tile(tile, blocks)) for tile in tiles]
    train = [tile for tile, value in assigned if value and value[0] == "train"]
    val = [tile for tile, value in assigned if value and value[0] == "val"]
    assert train and val
    assert all(a.intersection(b).area == 0 for a in train for b in val)
