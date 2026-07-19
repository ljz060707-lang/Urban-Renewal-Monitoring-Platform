from pathlib import Path

import pytest

from urban_monitor.project_discovery import (
    discover_images,
    match_images_and_labels,
    parse_date_from_filename,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("2023-07-15.tif", "2023-07-15"),
        ("2023_07_15.tif", "2023-07-15"),
        ("20230715.tif", "2023-07-15"),
    ],
)
def test_parse_supported_dates(name, expected):
    assert parse_date_from_filename(name) == expected


def test_duplicate_image_date_lists_conflicts(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    first = images / "2023-07-15.tif"
    second = images / "20230715.tiff"
    first.touch()
    second.touch()
    with pytest.raises(ValueError) as exc:
        discover_images(tmp_path)
    assert str(first.resolve()) in str(exc.value) and str(second.resolve()) in str(exc.value)


def test_image_without_label_is_an_error(tmp_path):
    image = tmp_path / "2023-07-15.tif"
    image.touch()
    with pytest.raises(ValueError, match="images without labels: 2023-07-15"):
        match_images_and_labels({"2023-07-15": image}, {})


def test_invalid_calendar_date_is_an_error():
    with pytest.raises(ValueError, match="Invalid calendar date"):
        parse_date_from_filename("2023-02-30.tif")
