from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .base import ChangeAdapter, files_by_stem, image_size, named_directory, read_binary, require_same_stems, split_directories, write_mask
from ..schema import ChangeSample


class S2LookingAdapter(ChangeAdapter):
    dataset_name = "s2looking"

    def discover(self) -> list[ChangeSample]:
        records: list[ChangeSample] = []
        for split, folder in split_directories(self.root):
            dirs = {
                "t1": named_directory(folder, ["Image1", "T1", "A"]),
                "t2": named_directory(folder, ["Image2", "T2", "B"]),
                "cd": named_directory(folder, ["label", "GT", "GT_CD"]),
                "new": named_directory(folder, ["label1", "GT_NEW"]),
                "dem": named_directory(folder, ["label2", "GT_DEM"]),
            }
            files = {name: files_by_stem(path) for name, path in dirs.items()}
            for stem in require_same_stems(files):
                width, height = image_size(files["t1"][stem])
                records.append(ChangeSample(
                    sample_id=f"s2looking:{split}:{stem}", dataset_name=self.dataset_name, split=split,
                    t1_path=files["t1"][stem], t2_path=files["t2"][stem], gt_cd_path=files["cd"][stem],
                    gt_new_path=files["new"][stem], gt_dem_path=files["dem"][stem], has_binary=True,
                    has_directional=True, has_semantic=False, width=width, height=height,
                ))
        self.validate(records)
        return records

    def normalize(self, sample: ChangeSample, output_dir: Path) -> ChangeSample:
        new = read_binary(sample.gt_new_path)
        dem = read_binary(sample.gt_dem_path)
        cd = read_binary(sample.gt_cd_path)
        derived = np.logical_or(new, dem).astype(np.uint8)
        mismatch = float(np.mean(cd != derived))
        target = output_dir / sample.split / sample.sample_id.rsplit(":", 1)[-1]
        paths = {
            # Unified contract follows the documented directional definition.
            "gt_cd_path": write_mask(target.with_name(target.name + "_cd.png"), derived),
            "gt_new_path": write_mask(target.with_name(target.name + "_new.png"), new),
            "gt_dem_path": write_mask(target.with_name(target.name + "_dem.png"), dem),
        }
        return replace(sample, **paths, notes=f"label_or_directional_mismatch={mismatch:.8f}")
