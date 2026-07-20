from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .base import ChangeAdapter, files_by_stem, image_size, named_directory, read_binary, require_same_stems, split_directories, write_mask
from ..schema import ChangeSample


class DSIFNAdapter(ChangeAdapter):
    dataset_name = "dsifn"

    def discover(self) -> list[ChangeSample]:
        records: list[ChangeSample] = []
        for split, folder in split_directories(self.root):
            dirs = {
                "t1": named_directory(folder, ["T1", "Image1", "A"]),
                "t2": named_directory(folder, ["T2", "Image2", "B"]),
                "cd": named_directory(folder, ["GT", "label", "GT_CD", "mask"]),
            }
            files = {name: files_by_stem(path) for name, path in dirs.items()}
            for stem in require_same_stems(files):
                width, height = image_size(files["t1"][stem])
                records.append(ChangeSample(
                    sample_id=f"dsifn:{split}:{stem}", dataset_name=self.dataset_name, split=split,
                    t1_path=files["t1"][stem], t2_path=files["t2"][stem], gt_cd_path=files["cd"][stem],
                    has_binary=True, has_directional=False, has_semantic=False, width=width, height=height,
                    notes="binary-only; direction is intentionally unavailable",
                ))
        self.validate(records)
        return records

    def normalize(self, sample: ChangeSample, output_dir: Path) -> ChangeSample:
        cd = read_binary(sample.gt_cd_path)
        target = output_dir / sample.split / (sample.sample_id.rsplit(":", 1)[-1] + "_cd.png")
        return replace(sample, gt_cd_path=write_mask(target, cd), gt_new_path=None, gt_dem_path=None)
