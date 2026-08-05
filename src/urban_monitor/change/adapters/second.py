from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .base import ChangeAdapter, files_by_stem, image_size, limited_stems, named_directory, read_binary, require_same_stems, split_directories, write_mask
from ..schema import ChangeSample


def load_color_mapping(path: str | Path | None) -> dict[tuple[int, int, int], int]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = payload.get("colors", payload)
    result: dict[tuple[int, int, int], int] = {}
    for key, value in mapping.items():
        color = tuple(int(part) for part in str(key).replace(" ", "").split(","))
        if len(color) != 3 or any(part < 0 or part > 255 for part in color):
            raise ValueError(f"Invalid SECOND RGB color: {key}")
        result[color] = int(value)
    return result


def semantic_ids(path: str | Path, color_mapping: dict[tuple[int, int, int], int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.int64)
    if array.ndim == 3 and array.shape[2] == 1:
        return array[..., 0].astype(np.int64)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Unsupported SECOND semantic label shape: {path}: {array.shape}")
    mapping = color_mapping or {}
    if not mapping:
        raise ValueError(
            f"SECOND RGB label requires an explicit verified class_mapping_file; refusing to guess colors: {path}"
        )
    rgb = array[..., :3].astype(np.uint8)
    output = np.full(rgb.shape[:2], -1, dtype=np.int64)
    for color, class_id in mapping.items():
        output[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)] = class_id
    unknown = np.unique(rgb[output < 0].reshape(-1, 3), axis=0) if np.any(output < 0) else np.empty((0, 3), dtype=np.uint8)
    if len(unknown):
        raise ValueError(f"SECOND color mapping missing colors in {path}: {unknown[:10].tolist()}")
    return output


class SECONDAdapter(ChangeAdapter):
    dataset_name = "second"

    def __init__(self, root: str | Path, config: dict | None = None):
        super().__init__(root, config)
        self.building_class_id = self.config.get("building_class_id")
        self.mapping_verified = (
            bool(self.config["mapping_verified"])
            if "mapping_verified" in self.config
            else self.building_class_id is not None
        )
        if self.mapping_verified and self.building_class_id is None:
            raise ValueError("SECOND mapping_verified=true requires building_class_id")
        self.color_mapping = load_color_mapping(self.config.get("class_mapping_file"))

    def discover(self) -> list[ChangeSample]:
        records: list[ChangeSample] = []
        for split, folder in split_directories(self.root, self.config.get("allowed_splits")):
            dirs = {
                "t1": named_directory(folder, ["T1", "Image1", "A"]),
                "t2": named_directory(folder, ["T2", "Image2", "B"]),
                "gt1": named_directory(folder, ["GT_T1", "label1", "A_L"]),
                "gt2": named_directory(folder, ["GT_T2", "label2", "B_L"]),
            }
            cd_dir = named_directory(folder, ["GT_CD", "label", "GT"], required=False)
            files = {name: files_by_stem(path) for name, path in dirs.items()}
            if cd_dir:
                files["cd"] = files_by_stem(cd_dir)
            for stem in limited_stems(require_same_stems(files), split, self.config):
                width, height = image_size(files["t1"][stem])
                records.append(ChangeSample(
                    sample_id=f"second:{split}:{stem}", dataset_name=self.dataset_name, split=split,
                    t1_path=files["t1"][stem], t2_path=files["t2"][stem],
                    gt_cd_path=files.get("cd", {}).get(stem), gt_t1_path=files["gt1"][stem], gt_t2_path=files["gt2"][stem],
                    has_binary=True, has_directional=self.mapping_verified, has_semantic=True, width=width, height=height,
                ))
        if not records:
            raise ValueError(f"second: no samples found under {self.root}")
        # SECOND 的二值变化标签允许缺省，因为它可由两期语义标签可靠派生。
        # 这里不伪造 gt_cd_path；只校验实际存在的两期影像和语义标签。
        for record in records:
            paths = (record.t1_path, record.t2_path, record.gt_t1_path, record.gt_t2_path)
            if any(path is None or not Path(path).is_file() for path in paths):
                raise ValueError(f"{record.sample_id}: missing T1/T2 or semantic supervision")
            sizes = [image_size(Path(path)) for path in paths if path is not None]
            if record.gt_cd_path:
                sizes.append(image_size(record.gt_cd_path))
            if len(set(sizes)) != 1:
                raise ValueError(f"{record.sample_id}: image/label size mismatch: {sizes}")
        return records

    def normalize(self, sample: ChangeSample, output_dir: Path) -> ChangeSample:
        gt1 = semantic_ids(sample.gt_t1_path, self.color_mapping)
        gt2 = semantic_ids(sample.gt_t2_path, self.color_mapping)
        if gt1.shape != gt2.shape:
            raise ValueError(f"{sample.sample_id}: semantic label size mismatch")
        cd = (gt1 != gt2).astype(np.uint8)
        if sample.gt_cd_path:
            provided = read_binary(sample.gt_cd_path)
            mismatch = float(np.mean(cd != provided))
        else:
            mismatch = 0.0
        target = output_dir / sample.split / sample.sample_id.rsplit(":", 1)[-1]
        paths = {
            "gt_cd_path": write_mask(target.with_name(target.name + "_cd.png"), cd),
            "gt_t1_path": write_mask(target.with_name(target.name + "_t1.png"), gt1),
            "gt_t2_path": write_mask(target.with_name(target.name + "_t2.png"), gt2),
        }
        if self.mapping_verified:
            building = int(self.building_class_id)

            # SECOND semantic void / ignore pixels must never become
            # directional building supervision.
            ignore_index = int(self.config.get("semantic_ignore_index", self.config.get("ignore_index", 255)))

            valid = (
                (gt1 != ignore_index)
                & (gt2 != ignore_index)
            )

            new = (
                valid
                & (gt1 != building)
                & (gt2 == building)
            ).astype(np.uint8)

            dem = (
                valid
                & (gt1 == building)
                & (gt2 != building)
            ).astype(np.uint8)

            paths["gt_new_path"] = write_mask(
                target.with_name(target.name + "_new.png"),
                new,
            )

            paths["gt_dem_path"] = write_mask(
                target.with_name(target.name + "_dem.png"),
                dem,
            )
            paths["gt_new_path"] = write_mask(target.with_name(target.name + "_new.png"), new)
            paths["gt_dem_path"] = write_mask(target.with_name(target.name + "_dem.png"), dem)
        return replace(
            sample,
            **paths,
            has_directional=self.mapping_verified,
            notes=f"provided_cd_mismatch={mismatch:.8f};mapping_verified={self.mapping_verified}",
        )
