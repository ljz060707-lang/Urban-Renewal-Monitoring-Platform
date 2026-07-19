import numpy as np
import rasterio
from rasterio.transform import from_origin

from urban_monitor.alignment_diagnostics import diagnose_alignment
from urban_monitor.crs import reproject_raster_to_reference
from urban_monitor.manifest import DateRecord


def _raster(path, transform, width=16, height=16):
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1, dtype="uint8", crs="EPSG:32650", transform=transform) as dst:
        dst.write(np.ones((height, width), dtype=np.uint8), 1)


def test_alignment_diagnostic_dynamic_buffer(tmp_path):
    a = tmp_path / "a.tif"; b = tmp_path / "b.tif"; transform = from_origin(500000, 3200000, 1, 1)
    _raster(a, transform); _raster(b, transform)
    records = [DateRecord("2024-01-01", a, None, None, "train", off_nadir_deg=2), DateRecord("2025-01-01", b, None, None, "train", off_nadir_deg=12)]
    report = diagnose_alignment(records, base_buffer_m=3, off_nadir_scale_m_per_degree=.5)
    assert report.iloc[0]["recommended_tracking_buffer_m"] == 8
    assert report.iloc[0]["overlap_fraction"] == 1


def test_reproject_raster_matches_reference_grid(tmp_path):
    source = tmp_path / "source.tif"; reference = tmp_path / "ref.tif"; output = tmp_path / "out.tif"
    _raster(source, from_origin(500000, 3200000, 2, 2), 8, 8)
    _raster(reference, from_origin(500000, 3200000, 1, 1), 16, 16)
    reproject_raster_to_reference(source, reference, output)
    with rasterio.open(reference) as ref, rasterio.open(output) as out:
        assert (out.crs, out.transform, out.width, out.height) == (ref.crs, ref.transform, ref.width, ref.height)
