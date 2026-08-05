from __future__ import annotations

import csv
import json
from pathlib import Path

from urban_renewal.construction.recovery_audit import audit_false_recovery_candidates


def test_recovery_audit_rejects_numeric_cross_dataset_candidates_without_reading_pixels(tmp_path: Path) -> None:
    ucs = tmp_path / "ucs"
    (ucs / "images_tif").mkdir(parents=True)
    (ucs / "label").mkdir(parents=True)
    for stem in ("1", "2", "3"):
        (ucs / "label" / f"{stem}.png").write_bytes(b"not-read")
    (ucs / "images_tif" / "1.tif").write_bytes(b"paired")
    whu = tmp_path / "whu"; whu.mkdir()
    dsifn = tmp_path / "dsifn"; dsifn.mkdir()
    (whu / "2.tif").write_bytes(b"candidate")
    (whu / "3.tif").write_bytes(b"candidate")
    (dsifn / "2.jpg").write_bytes(b"candidate")
    output = tmp_path / "audit"

    result = audit_false_recovery_candidates(
        tmp_path,
        ucs,
        output_dir=output,
        roots=(("WHU_Building", "test", whu), ("DSIFN", "test_t1", dsifn)),
    )

    assert result["valid_ucs_recovery_candidates"] == 0
    assert result["rejected_cross_dataset_candidates"] == 3
    assert result["missing_ucs_images"] == 2
    final = json.loads((output / "UCS_MISSING_IMAGE_FINAL_STATUS.json").read_text(encoding="utf-8"))
    assert final["missing_image_status"] == "true_local_absence"
    assert final["total_pixel_reads"] == 0
    with (output / "FALSE_RECOVERY_CANDIDATES.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert {row["rejection_code"] for row in rows} == {"cross_dataset_numeric_stem_collision"}
