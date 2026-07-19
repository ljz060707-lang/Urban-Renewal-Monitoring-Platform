import pandas as pd

from urban_monitor.manifest import load_manifest


def test_chinese_and_space_manifest_path(tmp_path):
    root = tmp_path / "北京 示例片区"; root.mkdir(); image = root / "影像 2024.tif"; image.write_bytes(b"placeholder")
    manifest = root / "清单.csv"
    pd.DataFrame([{"date": "2024-01-01", "image_path": image.name, "notes": "中文备注"}]).to_csv(manifest, index=False, encoding="utf-8-sig")
    record = load_manifest(manifest, root)[0]
    assert record.image_path == image and record.notes == "中文备注"


def test_utf8_bom_manifest_roundtrip(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"date": "2024-01-01", "image_path": "影像.tif"}]).to_csv(manifest, index=False, encoding="utf-8-sig")
    assert load_manifest(manifest, tmp_path)[0].date == "2024-01-01"
