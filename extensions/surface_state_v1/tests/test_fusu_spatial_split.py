import csv
from pathlib import Path

from PIL import Image

from surface_state_v1.datasets.fusu_split import split_fusu_sample


def test_grouped_fallback_has_no_group_overlap(tmp_path):
    rows = []
    for index in range(20):
        image = tmp_path / f"{index}.png"
        Image.new("RGB", (2, 2)).save(image)
        rows.append({"dataset": "fusu", "sample_id": f"6_{index}", "scene_id": f"filename_bucket_{index // 5:04d}", "split": "unassigned", "image_t1": str(image), "image_t2": str(image), "mask_t1": str(image), "mask_t2": str(image)})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = split_fusu_sample(manifest, tmp_path / "out")
    assert result["split_mode"] == "grouped_sample_split"
    assert result["train_validation_hash_overlap"] == 0
    assert result["sample_spatial_validation"]
