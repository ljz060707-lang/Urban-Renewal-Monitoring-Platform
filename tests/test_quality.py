import numpy as np
import rasterio
from rasterio.transform import from_origin

from urban_monitor.quality import compose_quality_masks, generate_quality_mask


def test_quality_mask_marks_bright_patch(tmp_path):
    image = tmp_path / "image.tif"
    output = tmp_path / "quality.tif"
    x = np.linspace(20, 200, 64, dtype=np.uint8)
    rgb = np.stack([np.tile(x, (64, 1))] * 3)
    rgb[:, 10:20, 10:20] = 255
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=3,
        dtype="uint8",
        crs="EPSG:32650",
        transform=from_origin(0, 64, 1, 1),
    ) as dst:
        dst.write(rgb)
    generate_quality_mask(image, output, dilation_pixels=0, min_global_dynamic_range=5)
    with rasterio.open(output) as src:
        usable = src.read(1)
        reason = src.read(2)
    assert usable[15, 15] == 0
    assert reason[15, 15] == 2
    assert usable[40, 40] == 1


def test_quality_composition_manual_overrides_provider(tmp_path):
    image = tmp_path / "image.tif"
    provider = tmp_path / "provider.tif"
    manual = tmp_path / "manual.tif"
    output = tmp_path / "composite.tif"
    profile = dict(driver="GTiff", height=8, width=8, count=1, dtype="uint8", crs="EPSG:32650", transform=from_origin(0, 8, 1, 1))
    with rasterio.open(image, "w", **{**profile, "count": 3}) as dst:
        dst.write(np.ones((3, 8, 8), dtype=np.uint8) * 80)
    provider_data = np.ones((8, 8), dtype=np.uint8); provider_data[2, 2] = 0
    manual_data = np.ones((8, 8), dtype=np.uint8); manual_data[2, 2] = 0; manual_data[3, 3] = 0
    for path, data in [(provider, provider_data), (manual, manual_data)]:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
    compose_quality_masks(image, output, provider_mask_path=provider, manual_mask_path=manual)
    with rasterio.open(output) as src:
        reason = src.read(2)
    assert reason[2, 2] == 7
    assert reason[3, 3] == 7


def test_registration_issue_invalidates_whole_scene(tmp_path):
    image = tmp_path / "image.tif"; output = tmp_path / "quality.tif"
    with rasterio.open(image, "w", driver="GTiff", height=4, width=4, count=3, dtype="uint8", crs="EPSG:32650", transform=from_origin(0, 4, 1, 1)) as dst:
        dst.write(np.ones((3, 4, 4), dtype=np.uint8))
    compose_quality_masks(image, output, registration_issue=True)
    with rasterio.open(output) as src:
        assert not src.read(1).any()
        assert np.all(src.read(2) == 8)
