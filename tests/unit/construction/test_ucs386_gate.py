from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from urban_renewal.construction.ucs386_gate import build_ucs386_subset_gate


def _write_fixture_dataset(root: Path, split_dir: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    image_dir = root / "images_tif"
    label_dir = root / "label"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / "fixture.tif"
    label_path = label_dir / "fixture.png"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=4,
        dtype="int16",
        crs="EPSG:32650",
        transform=from_origin(0, 2, 1, 1),
    ) as dataset:
        dataset.write(np.asarray([[[10, 20], [30, 40]]] * 4, dtype=np.int16))
    label = Image.fromarray(np.asarray([[0, 1], [2, 3]], dtype=np.uint8), mode="P")
    label.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * (768 - 12))
    label.save(label_path)

    split_dir.mkdir(parents=True)
    header = [
        "sample_id",
        "scene_id",
        "split",
        "image_path",
        "label_path",
        "crs",
        "grid_x",
        "grid_y",
        "image_sha256",
        "label_sha256",
    ]
    specs = {
        "train": ("UCS_TRAIN_MANIFEST.csv", 296),
        "validation": ("UCS_VALIDATION_MANIFEST.csv", 44),
        "test": ("UCS_TEST_LOCKED_MANIFEST.csv", 46),
    }
    for split, (filename, count) in specs.items():
        with (split_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            for index in range(count):
                writer.writerow(
                    {
                        "sample_id": f"{split}-{index}",
                        "scene_id": f"EPSG:32650:{split}:{index}",
                        "split": split,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "crs": "EPSG:32650",
                        "grid_x": str(index),
                        "grid_y": "0",
                        "image_sha256": "same-image-hash",
                        "label_sha256": "same-label-hash",
                    }
                )


def test_ucs386_gate_keeps_manifest_copies_and_locks_test_pixels(tmp_path: Path) -> None:
    ucs_root = tmp_path / "ucs"
    split_dir = tmp_path / "source_split"
    output_dir = tmp_path / "gate"
    _write_fixture_dataset(ucs_root, split_dir)

    payload = build_ucs386_subset_gate(
        tmp_path,
        ucs_root,
        split_dir=split_dir,
        output_dir=output_dir,
        rgb_mapping=(3, 2, 1),
        license_status="confirmed_training_allowed",
        license_source="explicit_user_confirmation",
    )

    assert payload["status"] == "FORMAL_SUBSET_TRAINING_BLOCKED"
    assert payload["required_checks"]["locked_test_pixel_reads_zero"] is True
    assert payload["test_pixel_reads"] == 0
    assert payload["required_checks"]["image_hashes_disjoint"] is False
    assert (output_dir / "UCS386_SUBSET_GATE.json").exists()
    assert (output_dir / "UCS386_CLASS_COVERAGE.json").exists()
    assert (output_dir / "UCS386_SPLIT_AUDIT.json").exists()
    assert (output_dir / "UCS386_TRAINING_APPROVAL_RECEIPT.json").exists()
    assert (output_dir / "UCS386_SCOPE_LIMITATIONS.md").exists()
    for filename in ("UCS_TRAIN_MANIFEST.csv", "UCS_VALIDATION_MANIFEST.csv", "UCS_TEST_LOCKED_MANIFEST.csv"):
        assert (output_dir / filename).read_bytes() == (split_dir / filename).read_bytes()
