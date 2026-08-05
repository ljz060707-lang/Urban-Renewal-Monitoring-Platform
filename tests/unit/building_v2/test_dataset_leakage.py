from pathlib import Path

import pytest

from urban_monitor.building_foundation_v2.data import WHUBuildingDataset


def test_dataset_refuses_test_rows():
    with pytest.raises(ValueError):
        WHUBuildingDataset([{"split": "test", "image_path": "x", "label_path": "y", "sample_id": "x"}])
