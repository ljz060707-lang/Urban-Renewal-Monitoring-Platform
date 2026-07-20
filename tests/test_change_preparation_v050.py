from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import yaml

from urban_monitor.change.preparation import prepare_public_change


def _save(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data.astype(np.uint8)).save(path)


def test_prepare_writes_index_statistics_and_previews(tmp_path: Path):
    root = tmp_path / "dsifn" / "train"
    image = np.full((16, 16, 3), 40, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    _save(root / "T1" / "sample.png", image)
    _save(root / "T2" / "sample.png", image + 10)
    _save(root / "GT" / "sample.png", mask)
    config = {
        "output_root": str(tmp_path / "outputs"),
        "datasets": {
            "s2looking": {"enabled": False}, "second": {"enabled": False},
            "dsifn": {"enabled": True, "root": str(root.parent)},
        },
        "audit": {"preview_count": 30},
    }
    config_path = tmp_path / "change.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run = prepare_public_change(config_path, run_id="test")
    frame = pd.read_csv(run / "prepared" / "public_change_index.csv")

    assert len(frame) == 1
    assert bool(frame.loc[0, "has_binary"])
    assert not bool(frame.loc[0, "has_directional"])
    assert (run / "dataset_audit" / "dsifn_summary.csv").is_file()
    assert len(list((run / "dataset_audit" / "dsifn_previews").glob("*.jpg"))) == 1
    assert (run / "RUN_MANIFEST.json").is_file()
