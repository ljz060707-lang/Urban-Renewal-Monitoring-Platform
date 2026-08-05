import numpy as np

from urban_monitor.building_foundation_v2.diagnostics import oracle_by_stride


def test_stride_oracle_is_finite_and_records_resolution():
    mask = np.zeros((64, 64), dtype=np.uint8); mask[12:52, 12:52] = 1
    rows = oracle_by_stride(mask, strides=(1, 2, 4, 8))
    assert [row["stride"] for row in rows] == [1, 2, 4, 8]
    assert all(np.isfinite(row["oracle_bf1@2"]) for row in rows)
    assert rows[0]["oracle_bf1@2"] == 1.0

