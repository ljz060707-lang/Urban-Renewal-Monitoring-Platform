from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


CHANGE_INDEX_COLUMNS = [
    "sample_id", "dataset_name", "split", "source_split", "t1_path", "t2_path", "gt_cd_path",
    "gt_new_path", "gt_dem_path", "gt_t1_path", "gt_t2_path", "has_binary",
    "has_directional", "has_semantic", "width", "height", "notes",
]


@dataclass(frozen=True)
class ChangeSample:
    sample_id: str
    dataset_name: str
    split: str
    t1_path: Path
    t2_path: Path
    gt_cd_path: Path | None = None
    gt_new_path: Path | None = None
    gt_dem_path: Path | None = None
    gt_t1_path: Path | None = None
    gt_t2_path: Path | None = None
    has_binary: bool = False
    has_directional: bool = False
    has_semantic: bool = False
    width: int = 0
    height: int = 0
    notes: str = ""
    source_split: str = ""

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        for key in ("t1_path", "t2_path", "gt_cd_path", "gt_new_path", "gt_dem_path", "gt_t1_path", "gt_t2_path"):
            row[key] = str(row[key]) if row[key] else ""
        return row


def validate_sample(sample: ChangeSample) -> None:
    if not sample.sample_id or not sample.dataset_name or not sample.split:
        raise ValueError("sample_id, dataset_name and split are required")
    for path in (sample.t1_path, sample.t2_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    supervision = {
        "has_binary": (sample.gt_cd_path,),
        "has_directional": (sample.gt_new_path, sample.gt_dem_path),
        "has_semantic": (sample.gt_t1_path, sample.gt_t2_path),
    }
    for flag, paths in supervision.items():
        if getattr(sample, flag) and any(path is None or not Path(path).is_file() for path in paths):
            raise ValueError(f"{sample.sample_id}: {flag}=true but supervision path is missing")
