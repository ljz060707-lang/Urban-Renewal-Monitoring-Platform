from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DateRecord:
    date: str
    image_path: Path
    label_path: Path | None
    quality_mask_path: Path | None
    split: str
    source: str = ""
    off_nadir_deg: float | None = None
    sun_elevation_deg: float | None = None
    season: str = ""
    quality_notes: str = ""
    provider_mask_path: Path | None = None
    manual_mask_path: Path | None = None
    sensor: str = ""
    gsd_m: float | None = None
    crs: str = ""
    coordinate_encoding: str = "projected"
    acquisition_time: str = ""
    registration_status: str = "unchecked"
    notes: str = ""


def _optional_path(root: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    return path if path.is_absolute() else root / path


def _optional_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_manifest(path: str | Path, data_root: str | Path) -> list[DateRecord]:
    manifest_path = Path(path)
    root = Path(data_root)
    frame = pd.read_csv(manifest_path, dtype=str, encoding="utf-8-sig").fillna("")
    required = {"date", "image_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame["date"].astype(str).str[:10].duplicated().any():
        raise ValueError("Manifest dates must be unique")
    records: list[DateRecord] = []
    for row in frame.to_dict("records"):
        image = Path(row["image_path"])
        image = image if image.is_absolute() else root / image
        records.append(
            DateRecord(
                date=str(row["date"])[:10],
                image_path=image,
                label_path=_optional_path(root, row.get("label_path")),
                quality_mask_path=_optional_path(root, row.get("quality_mask_path")),
                split=str(row.get("split") or "train"),
                source=str(row.get("source") or ""),
                off_nadir_deg=_optional_float(row.get("off_nadir_deg")),
                sun_elevation_deg=_optional_float(row.get("sun_elevation_deg")),
                season=str(row.get("season") or ""),
                quality_notes=str(row.get("quality_notes") or ""),
                provider_mask_path=_optional_path(root, row.get("provider_mask_path")),
                manual_mask_path=_optional_path(root, row.get("manual_mask_path")),
                sensor=str(row.get("sensor") or row.get("source") or ""),
                gsd_m=_optional_float(row.get("gsd_m")),
                crs=str(row.get("crs") or ""),
                coordinate_encoding=str(row.get("coordinate_encoding") or "projected").lower(),
                acquisition_time=str(row.get("acquisition_time") or ""),
                registration_status=str(row.get("registration_status") or "unchecked").lower(),
                notes=str(row.get("notes") or row.get("quality_notes") or ""),
            )
        )
    records.sort(key=lambda r: r.date)
    return records
