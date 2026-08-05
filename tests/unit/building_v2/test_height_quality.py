import json
from pathlib import Path


def test_height_is_not_enabled_by_default():
    from urban_monitor.building_foundation_v2.model import BuildingFoundationNet
    assert BuildingFoundationNet(base_channels=4).height_enabled is False
