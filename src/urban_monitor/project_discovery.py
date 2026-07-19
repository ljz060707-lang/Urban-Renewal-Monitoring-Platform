from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import rasterio

from .config import load_project_config


DATE_PATTERNS = (
    re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"),
    re.compile(r"^(\d{4})_(\d{2})_(\d{2})$"),
    re.compile(r"^(\d{4})(\d{2})(\d{2})$"),
)

MANIFEST_COLUMNS = [
    "date", "image_path", "label_path", "quality_mask_path", "provider_mask_path",
    "manual_mask_path", "sensor", "source", "gsd_m", "off_nadir_deg",
    "sun_elevation_deg", "season", "crs", "coordinate_encoding",
    "registration_status", "split", "notes",
]


@dataclass(frozen=True)
class DiscoveredProject:
    aoi_id: str
    project_dir: Path
    config: dict
    images: dict[str, Path]
    labels: dict[str, Path]


def parse_date_from_filename(path: str | Path) -> str:
    stem = Path(path).stem
    for pattern in DATE_PATTERNS:
        match = pattern.fullmatch(stem)
        if match:
            value = "-".join(match.groups())
            try:
                return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
            except ValueError as exc:
                raise ValueError(f"Invalid calendar date in filename: {path}") from exc
    raise ValueError(
        f"Cannot parse date from {Path(path).name}; expected YYYY-MM-DD, YYYY_MM_DD, or YYYYMMDD"
    )


def _discover(directory: Path, patterns: Iterable[str], kind: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required {kind} directory does not exist: {directory}")
    found: dict[str, list[Path]] = {}
    for pattern in patterns:
        for path in sorted(directory.glob(str(pattern))):
            if path.is_file():
                date = parse_date_from_filename(path)
                found.setdefault(date, []).append(path.resolve())
    conflicts = {date: paths for date, paths in found.items() if len(paths) > 1}
    if conflicts:
        details = "; ".join(f"{date}: {', '.join(str(p) for p in paths)}" for date, paths in conflicts.items())
        raise ValueError(f"Duplicate {kind} files for the same date: {details}")
    if not found:
        raise FileNotFoundError(f"No supported {kind} files found in {directory}")
    return {date: paths[0] for date, paths in found.items()}


def discover_images(project_dir: str | Path, patterns: Iterable[str] | None = None) -> dict[str, Path]:
    return _discover(Path(project_dir) / "images", patterns or ("*.tif", "*.tiff"), "image")


def discover_labels(project_dir: str | Path, patterns: Iterable[str] | None = None) -> dict[str, Path]:
    return _discover(Path(project_dir) / "labels", patterns or ("*.gpkg", "*.geojson", "*.shp"), "label")


def match_images_and_labels(images: dict[str, Path], labels: dict[str, Path]) -> list[tuple[str, Path, Path]]:
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels or missing_images:
        parts = []
        if missing_labels:
            parts.append(f"images without labels: {', '.join(missing_labels)}")
        if missing_images:
            parts.append(f"labels without images: {', '.join(missing_images)}")
        raise ValueError("Date matching failed; " + "; ".join(parts))
    return [(date, images[date], labels[date]) for date in sorted(images)]


def discover_project(aoi_id: str, data_root: str | Path = "data") -> DiscoveredProject:
    project_dir = (Path(data_root) / aoi_id).resolve()
    config = load_project_config(project_dir)
    data_cfg = config["data"]
    images = discover_images(project_dir, data_cfg["image_glob"])
    labels = discover_labels(project_dir, data_cfg["label_glob"])
    match_images_and_labels(images, labels)
    return DiscoveredProject(aoi_id, project_dir, config, images, labels)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def build_manifest(project: DiscoveredProject, output_path: str | Path | None = None) -> Path:
    rows: list[dict] = []
    quality_dir = project.project_dir / "quality"
    quality = {}
    if quality_dir.is_dir() and any(quality_dir.glob("*.tif*")):
        quality = _discover(quality_dir, ("*.tif", "*.tiff"), "quality mask")
    for date, image_path, label_path in match_images_and_labels(project.images, project.labels):
        with rasterio.open(image_path) as source:
            if source.crs is None:
                raise ValueError(f"Image CRS is missing: {image_path}")
            gsd = (abs(float(source.transform.a)) + abs(float(source.transform.e))) / 2.0
            crs = source.crs.to_string()
            coordinate_encoding = "projected" if source.crs.is_projected else "wgs84"
        rows.append(
            {
                "date": date,
                "image_path": _relative(image_path, project.project_dir),
                "label_path": _relative(label_path, project.project_dir),
                "quality_mask_path": _relative(quality[date], project.project_dir) if date in quality else "",
                "provider_mask_path": "",
                "manual_mask_path": "",
                "sensor": "",
                "source": "",
                "gsd_m": gsd,
                "off_nadir_deg": "",
                "sun_elevation_deg": "",
                "season": "",
                "crs": crs,
                "coordinate_encoding": coordinate_encoding,
                "registration_status": "unchecked",
                "split": "spatial_blocks",
                "notes": "",
            }
        )
    output = Path(output_path) if output_path else project.project_dir / "manifest.generated.csv"
    if output.name.lower() == "manifest.csv":
        raise ValueError("Automatic discovery never overwrites user manifest.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def resolve_manifest(
    project: DiscoveredProject,
    explicit_manifest: str | Path | None = None,
    *,
    generate: bool = True,
) -> tuple[Path, str]:
    if explicit_manifest:
        path = Path(explicit_manifest).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Explicit manifest does not exist: {path}")
        return path, "command_line"
    configured = project.config.get("manifest") or project.config.get("data", {}).get("manifest")
    if configured:
        path = Path(configured)
        path = path if path.is_absolute() else project.project_dir / path
        if not path.exists():
            raise FileNotFoundError(f"project.yaml manifest does not exist: {path}")
        return path.resolve(), "project_yaml"
    manual = project.project_dir / "manifest.csv"
    if manual.exists():
        return manual.resolve(), "manual"
    generated = project.project_dir / "manifest.generated.csv"
    if generated.exists() and not generate:
        return generated.resolve(), "generated"
    return build_manifest(project, generated).resolve(), "auto_discovery"
