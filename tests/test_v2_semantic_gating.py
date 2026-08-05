import numpy as np
from PIL import Image

from urban_monitor.change.adapters.second import SECONDAdapter
from urban_monitor.change.schema import ChangeSample


def test_second_mapping_pending_keeps_semantic_without_directional(tmp_path):
    root = tmp_path / "second"
    root.mkdir()
    adapter = SECONDAdapter(root, {"mapping_verified": False})
    image = np.zeros((8, 8), dtype=np.uint8)
    for name in ("t1", "t2", "g1", "g2"):
        Image.fromarray(image).save(tmp_path / f"{name}.png")
    sample = ChangeSample(
        "second:train:a", "second", "train", tmp_path / "t1.png", tmp_path / "t2.png",
        gt_t1_path=tmp_path / "g1.png", gt_t2_path=tmp_path / "g2.png",
        has_binary=True, has_semantic=True, has_directional=False, width=8, height=8,
    )
    result = adapter.normalize(sample, tmp_path / "out")
    assert result.has_semantic and result.has_binary
    assert not result.has_directional
    assert result.gt_new_path is None and result.gt_dem_path is None

