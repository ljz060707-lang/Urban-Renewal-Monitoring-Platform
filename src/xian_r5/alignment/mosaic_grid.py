"""Exact source-tile to stitched-raster grid mapping for Xi'an z20 mosaics.

The GEID task date is only a nominal label. Callers must supply each tile's
actual acquisition date from the per-tile date audit. This module validates
the immutable mosaic georeference and maps source JPEG components to exact
pixel windows; it does not estimate or apply a ground correction.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from affine import Affine
from PIL import Image
import rasterio


@dataclass(frozen=True)
class TileRasterWindow:
    tile_x: int
    tile_y: int
    col_off: int
    row_off: int
    width: int
    height: int

    @property
    def pixel_window(self) -> list[int]:
        """Return ``[column, row, width, height]`` in the stitched TIFF."""
        return [self.col_off, self.row_off, self.width, self.height]


@dataclass(frozen=True)
class TileModelInputGrid:
    """Exact model-input pixel grid mapped through one source tile."""

    tile_x: int
    tile_y: int
    model_input_width_px: int
    model_input_height_px: int
    pixel_crs: str
    pixel_to_crs_affine: tuple[float, float, float, float, float, float]
    source_pixel_window: tuple[int, int, int, int]


@dataclass(frozen=True)
class MosaicTileGrid:
    """Validated mapping between GEID z20 source tiles and a stitched TIFF."""

    geid_path: Path
    mosaic_path: Path
    tile_root: Path
    scene_id: str
    nominal_task_date: str
    zoom: int
    min_tile_x: int
    min_tile_y: int
    max_tile_x: int
    max_tile_y: int
    tile_width_px: int
    tile_height_px: int
    mosaic_crs: str
    mosaic_width_px: int
    mosaic_height_px: int
    mosaic_transform: tuple[float, float, float, float, float, float]
    mosaic_bounds: tuple[float, float, float, float]
    geid_size_bytes: int
    geid_mtime_ns: int
    mosaic_size_bytes: int
    mosaic_mtime_ns: int

    @classmethod
    def open(
        cls,
        geid_path: Path,
        mosaic_path: Path,
        tile_root: Path,
        *,
        tile_size_px: tuple[int, int] = (256, 256),
    ) -> "MosaicTileGrid":
        """Read task metadata and validate its full-raster grid without pixel scans.

        ``tile_root`` is the scene's ``20`` directory containing
        ``<x>/gesh_<x>_<y>_20.jpg`` source components. The returned date is
        explicitly named ``nominal_task_date``; it must not substitute for
        tile-level actual dates.
        """
        geid_path, mosaic_path, tile_root = (
            Path(geid_path), Path(mosaic_path), Path(tile_root)
        )
        if not geid_path.is_file():
            raise FileNotFoundError(geid_path)
        if not mosaic_path.is_file():
            raise FileNotFoundError(mosaic_path)
        if len(tile_size_px) != 2 or min(tile_size_px) < 1:
            raise ValueError("Source tile dimensions must be two positive integers")
        geid_stat_before = geid_path.stat()
        mosaic_stat_before = mosaic_path.stat()

        parser = configparser.ConfigParser(interpolation=None)
        parser.optionxform = str
        try:
            parser.read_string(geid_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, configparser.Error) as exc:
            raise ValueError(f"Cannot read GEID metadata: {geid_path}") from exc

        def option(section_name: str, key: str) -> str:
            sections = {name.casefold(): name for name in parser.sections()}
            actual_section = sections.get(section_name.casefold())
            if actual_section is None:
                raise ValueError(f"GEID is missing [{section_name}]")
            values = {name.casefold(): value for name, value in parser.items(actual_section)}
            if key.casefold() not in values:
                raise ValueError(f"GEID is missing {section_name}.{key}")
            return values[key.casefold()].strip()

        def integer(section: str, key: str) -> int:
            try:
                return int(option(section, key))
            except ValueError as exc:
                raise ValueError(f"GEID {section}.{key} must be an integer") from exc

        def number(section: str, key: str) -> float:
            try:
                value = float(option(section, key))
            except ValueError as exc:
                raise ValueError(f"GEID {section}.{key} must be numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"GEID {section}.{key} must be finite")
            return value

        try:
            nominal_date = date.fromisoformat(option("HistoryDate", "HistoryDate")).isoformat()
        except ValueError as exc:
            raise ValueError("GEID HistoryDate must be an ISO date task label") from exc
        zoom1, zoom2 = integer("Zoom", "Zoom1"), integer("Zoom", "Zoom2")
        if zoom1 != zoom2 or zoom1 != 20:
            raise ValueError("This source-tile interface currently requires a z20 GEID")
        min_x, max_x = integer("XY", "MinX_20"), integer("XY", "MaxX_20")
        min_y, max_y = integer("XY", "MinY_20"), integer("XY", "MaxY_20")
        if min_x > max_x or min_y > max_y:
            raise ValueError("GEID tile range is inverted")

        left = number("AREA", "Left_Longitude_download_20")
        right = number("AREA", "Right_Longitude_download_20")
        top = number("AREA", "Top_Latitude_download_20")
        bottom = number("AREA", "Bottom_Latitude_download_20")
        if left >= right or bottom >= top:
            raise ValueError("GEID z20 download bounds are invalid")

        expected_width = (max_x - min_x + 1) * tile_size_px[0]
        expected_height = (max_y - min_y + 1) * tile_size_px[1]
        expected_transform = (
            (right - left) / expected_width,
            0.0,
            left,
            0.0,
            -(top - bottom) / expected_height,
            top,
        )
        expected_bounds = (left, bottom, right, top)
        with rasterio.open(mosaic_path) as raster:
            if raster.crs is None or raster.crs.to_epsg() != 4326:
                raise ValueError("Stitched z20 source TIFF must be georeferenced in EPSG:4326")
            if raster.width != expected_width or raster.height != expected_height:
                raise ValueError(
                    "Stitched TIFF dimensions do not equal the inclusive GEID tile grid"
                )
            actual_transform = tuple(float(value) for value in tuple(raster.transform)[:6])
            if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-10)
                   for a, b in zip(actual_transform, expected_transform)):
                raise ValueError("Stitched TIFF affine transform disagrees with GEID bounds")
            actual_bounds = tuple(float(value) for value in raster.bounds)
            if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8)
                   for a, b in zip(actual_bounds, expected_bounds)):
                raise ValueError("Stitched TIFF bounds disagree with GEID download bounds")
            crs = raster.crs.to_string()
            width, height = raster.width, raster.height
            transform = actual_transform
            bounds = actual_bounds

        geid_stat_after = geid_path.stat()
        mosaic_stat_after = mosaic_path.stat()
        if ((geid_stat_before.st_size, geid_stat_before.st_mtime_ns)
                != (geid_stat_after.st_size, geid_stat_after.st_mtime_ns)
                or (mosaic_stat_before.st_size, mosaic_stat_before.st_mtime_ns)
                != (mosaic_stat_after.st_size, mosaic_stat_after.st_mtime_ns)):
            raise RuntimeError("GEID or stitched mosaic changed while its grid was validated")

        return cls(
            geid_path=geid_path.resolve(),
            mosaic_path=mosaic_path.resolve(),
            tile_root=tile_root.resolve(),
            scene_id=geid_path.stem,
            nominal_task_date=nominal_date,
            zoom=zoom1,
            min_tile_x=min_x,
            min_tile_y=min_y,
            max_tile_x=max_x,
            max_tile_y=max_y,
            tile_width_px=int(tile_size_px[0]),
            tile_height_px=int(tile_size_px[1]),
            mosaic_crs=crs,
            mosaic_width_px=width,
            mosaic_height_px=height,
            mosaic_transform=transform,
            mosaic_bounds=bounds,
            geid_size_bytes=geid_stat_after.st_size,
            geid_mtime_ns=geid_stat_after.st_mtime_ns,
            mosaic_size_bytes=mosaic_stat_after.st_size,
            mosaic_mtime_ns=mosaic_stat_after.st_mtime_ns,
        )

    def _assert_grid_sources_unchanged(self) -> None:
        geid_stat = self.geid_path.stat()
        mosaic_stat = self.mosaic_path.stat()
        if ((geid_stat.st_size, geid_stat.st_mtime_ns)
                != (self.geid_size_bytes, self.geid_mtime_ns)
                or (mosaic_stat.st_size, mosaic_stat.st_mtime_ns)
                != (self.mosaic_size_bytes, self.mosaic_mtime_ns)):
            raise RuntimeError("GEID or stitched mosaic changed after grid validation")

    def window_for_tile(self, tile_x: int, tile_y: int) -> TileRasterWindow:
        """Map a source component to its exact row/column window."""
        if not (self.min_tile_x <= tile_x <= self.max_tile_x
                and self.min_tile_y <= tile_y <= self.max_tile_y):
            raise ValueError(f"Tile {tile_x}:{tile_y} is outside {self.scene_id}'s GEID grid")
        return TileRasterWindow(
            tile_x=int(tile_x),
            tile_y=int(tile_y),
            col_off=(int(tile_x) - self.min_tile_x) * self.tile_width_px,
            # GEID Y increases northward; TIFF rows increase southward.
            row_off=(self.max_tile_y - int(tile_y)) * self.tile_height_px,
            width=self.tile_width_px,
            height=self.tile_height_px,
        )

    def source_path_for_tile(self, tile_x: int, tile_y: int) -> Path:
        """Return the deterministic source JPEG path, without scanning the scene."""
        self.window_for_tile(tile_x, tile_y)
        jpg = self.tile_root / str(tile_x) / f"gesh_{tile_x}_{tile_y}_{self.zoom}.jpg"
        jpeg = self.tile_root / str(tile_x) / f"gesh_{tile_x}_{tile_y}_{self.zoom}.jpeg"
        if jpg.is_file():
            return jpg
        if jpeg.is_file():
            return jpeg
        return jpg

    def model_input_grid_for_tile(
        self,
        tile_x: int,
        tile_y: int,
        *,
        model_input_width_px: int,
        model_input_height_px: int,
    ) -> TileModelInputGrid:
        """Derive model-input pixels to mosaic-CRS coordinates for a full tile.

        The model input must be an aspect-preserving resize of the entire source
        tile (no crop, padding, or letterboxing). This keeps the transform
        deterministic and avoids trusting an affine supplied by a prediction.
        """
        dimensions = (model_input_width_px, model_input_height_px)
        self._assert_grid_sources_unchanged()
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in dimensions):
            raise ValueError("Model-input dimensions must be positive integers")
        source_aspect = self.tile_width_px / self.tile_height_px
        model_aspect = model_input_width_px / model_input_height_px
        if not math.isclose(source_aspect, model_aspect, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Model input must preserve the full source-tile aspect ratio; "
                "crop/pad transforms are not supported"
            )

        window = self.window_for_tile(tile_x, tile_y)
        model_to_source = Affine.scale(
            self.tile_width_px / model_input_width_px,
            self.tile_height_px / model_input_height_px,
        )
        model_to_mosaic = (
            Affine(*self.mosaic_transform)
            @ Affine.translation(window.col_off, window.row_off)
            @ model_to_source
        )
        return TileModelInputGrid(
            tile_x=int(tile_x),
            tile_y=int(tile_y),
            model_input_width_px=model_input_width_px,
            model_input_height_px=model_input_height_px,
            pixel_crs=self.mosaic_crs,
            pixel_to_crs_affine=tuple(float(value) for value in tuple(model_to_mosaic)[:6]),
            source_pixel_window=tuple(window.pixel_window),
        )

    def component_receipt(
        self,
        tile_x: int,
        tile_y: int,
        *,
        actual_date: str,
        actual_date_source_ref: str,
        expected_sha256: str | None = None,
        expected_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Validate one source component and return traceable raster-window evidence.

        Actual date is mandatory and supplied by the per-tile date audit; the
        nominal GEID task date is never promoted to an acquisition date here.
        """
        self._assert_grid_sources_unchanged()
        try:
            actual_date = date.fromisoformat(actual_date).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError("A tile-level actual ISO acquisition date is required") from exc
        if not actual_date_source_ref:
            raise ValueError("Actual-date provenance is required for each source tile")
        path = self.source_path_for_tile(tile_x, tile_y)
        if not path.is_file():
            raise FileNotFoundError(path)
        if expected_path is not None:
            expected = Path(expected_path).resolve()
            if path.resolve() != expected:
                raise ValueError(f"Tile provenance path mismatch for {tile_x}:{tile_y}")

        before = path.stat()
        try:
            with Image.open(path) as image:
                image_format = image.format
                image_width, image_height = image.size
                image.verify()
        except Exception as exc:
            raise ValueError(f"Cannot decode source image tile: {path}") from exc
        if (image_format not in {"JPEG", "JPG"}
                or image_width != self.tile_width_px
                or image_height != self.tile_height_px):
            raise ValueError(f"Source tile dimensions/format disagree with the GEID grid: {path}")
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source tile changed while being audited: {path}")
        if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
            raise ValueError(f"Source tile hash mismatch for {tile_x}:{tile_y}")
        self._assert_grid_sources_unchanged()

        window = self.window_for_tile(tile_x, tile_y)
        a, b, c, d, e, f = self.mosaic_transform
        west, north = a * window.col_off + b * window.row_off + c, d * window.col_off + e * window.row_off + f
        east, south = (a * (window.col_off + window.width) + b * (window.row_off + window.height) + c,
                       d * (window.col_off + window.width) + e * (window.row_off + window.height) + f)
        return {
            "tile_id": f"{tile_x}:{tile_y}",
            "actual_date": actual_date,
            "actual_date_source_ref": actual_date_source_ref,
            "nominal_task_date": self.nominal_task_date,
            "geid_path": str(self.geid_path),
            "mosaic_path": str(self.mosaic_path),
            "mosaic_size_bytes": self.mosaic_size_bytes,
            "mosaic_pixel_hash_status": "NOT_COMPUTED_FULL_RASTER_SCAN",
            "source_tile_path": str(path.resolve()),
            "source_tile_sha256": digest,
            "source_tile_size_bytes": after.st_size,
            "source_tile_format": image_format,
            "source_tile_pixel_size": [image_width, image_height],
            "mosaic_crs": self.mosaic_crs,
            "mosaic_transform": list(self.mosaic_transform),
            "pixel_window_col_row_width_height": window.pixel_window,
            "window_bounds_epsg4326": [west, south, east, north],
        }

    def summary(self) -> dict[str, Any]:
        """Serialize the validated grid while distinguishing nominal and real dates."""
        return {
            "scene_id": self.scene_id,
            "nominal_task_date": self.nominal_task_date,
            "actual_acquisition_date_status": "MUST_BE_RESOLVED_PER_TILE",
            "zoom": self.zoom,
            "tile_window_inclusive": {
                "min_x": self.min_tile_x,
                "min_y": self.min_tile_y,
                "max_x": self.max_tile_x,
                "max_y": self.max_tile_y,
            },
            "tile_count": ((self.max_tile_x - self.min_tile_x + 1)
                           * (self.max_tile_y - self.min_tile_y + 1)),
            "tile_pixel_size": [self.tile_width_px, self.tile_height_px],
            "mosaic_path": str(self.mosaic_path),
            "mosaic_pixel_hash_status": "NOT_COMPUTED_FULL_RASTER_SCAN",
            "mosaic_size_bytes": self.mosaic_size_bytes,
            "mosaic_crs": self.mosaic_crs,
            "mosaic_width_height": [self.mosaic_width_px, self.mosaic_height_px],
            "mosaic_transform": list(self.mosaic_transform),
            "mosaic_bounds": list(self.mosaic_bounds),
        }


def _resolved_path_key(path: str | Path) -> str:
    """Return a stable, platform-aware identity for a recorded source path."""
    return os.path.normcase(str(Path(path).resolve()))


def _epoch_for_recorded_tile_path(
    entry: dict[str, Any],
    *,
    source_root: Path,
    scenes_by_epoch: dict[str, str],
) -> str:
    """Resolve a legacy tile's source epoch only by exact scene-path agreement."""
    match = re.fullmatch(r"(\d+):(\d+)", str(entry.get("tile_id") or ""))
    tile_path = entry.get("path")
    if match is None or not tile_path:
        raise ValueError("Legacy source tile is missing an exact tile ID or path")
    tile_x, tile_y = (int(value) for value in match.groups())
    recorded_key = _resolved_path_key(str(tile_path))
    matches = []
    for epoch, scene in scenes_by_epoch.items():
        for suffix in ("jpg", "jpeg"):
            expected = (source_root / scene / "20" / str(tile_x)
                        / f"gesh_{tile_x}_{tile_y}_20.{suffix}")
            if _resolved_path_key(expected) == recorded_key:
                matches.append(str(epoch))
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely resolve source epoch from recorded tile path: "
            f"{entry.get('tile_id')} ({len(matches)} matches)"
        )
    return matches[0]


def load_mosaic_grids_by_source_epoch(
    diagnostic_path: Path,
    *,
    source_root: Path,
    scenes_by_epoch: dict[str, str],
    tile_size_px: tuple[int, int] = (256, 256),
) -> dict[str, MosaicTileGrid]:
    """Load the stitched-TIFF grids selected by a fixed-case tile-date receipt.

    A single actual date may draw tiles from more than one nominal source task;
    grids are therefore keyed by each tile's ``selected_source_epoch``, not by
    actual date. Tile-level actual dates remain supplied separately to
    ``component_receipt``.
    """
    diagnostic_path = Path(diagnostic_path)
    source_root = Path(source_root)
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if (diagnostic.get("schema_version") != "xian_r5.mosaic_registration_diagnostic.v1"
            or diagnostic.get("scope") != "FIXED_CASE_LOCAL_ONLY"):
        raise ValueError("Source-grid loading requires a fixed-case mosaic diagnostic")
    association = diagnostic.get("actual_date_association")
    date_epoch_map = (association.get("candidate_source_epochs_by_actual_date", {})
                      if isinstance(association, dict) else {})
    legacy_epochs = diagnostic.get("epochs", {})
    tiles_by_date = diagnostic.get("submeter_tiles")
    if not isinstance(tiles_by_date, dict) or not tiles_by_date:
        raise ValueError("Fixed-case mosaic diagnostic has no actual-date tile lists")

    grids: dict[str, MosaicTileGrid] = {}
    for actual_date, entries in tiles_by_date.items():
        try:
            date.fromisoformat(str(actual_date))
        except ValueError as exc:
            raise ValueError(f"Invalid actual acquisition date in mosaic receipt: {actual_date}") from exc
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Missing exact-date source tiles: {actual_date}")
        if isinstance(date_epoch_map, dict) and actual_date in date_epoch_map:
            allowed_values = date_epoch_map[actual_date]
        else:
            legacy_summary = (legacy_epochs.get(actual_date, {})
                              if isinstance(legacy_epochs, dict) else {})
            legacy_epoch = (legacy_summary.get("selected_source_epoch")
                            if isinstance(legacy_summary, dict) else None)
            allowed_values = [legacy_epoch] if legacy_epoch else []
        if not isinstance(allowed_values, list) or not allowed_values:
            raise ValueError(
                f"Missing selected source epochs for actual date in mosaic receipt: {actual_date}"
            )
        allowed_epochs = {str(value) for value in allowed_values if str(value)}
        if not allowed_epochs:
            raise ValueError(f"Invalid selected source epochs for actual date: {actual_date}")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid source tile entry for actual date: {actual_date}")
            path_epoch = _epoch_for_recorded_tile_path(
                entry, source_root=source_root, scenes_by_epoch=scenes_by_epoch)
            recorded_epoch = str(entry.get("selected_source_epoch") or path_epoch)
            if recorded_epoch != path_epoch or recorded_epoch not in allowed_epochs:
                raise ValueError(
                    f"Selected source epoch/path is not in the actual-date receipt: "
                    f"{actual_date}/{recorded_epoch}"
                )
            if recorded_epoch in grids:
                continue
            scene = scenes_by_epoch.get(recorded_epoch)
            if not scene:
                raise ValueError(
                    f"No GEID scene mapping for selected source epoch: {recorded_epoch}"
                )
            grids[recorded_epoch] = MosaicTileGrid.open(
                source_root / f"{scene}.geid",
                source_root / f"{scene}_combined" / f"{scene}_zoom_20.tif",
                source_root / scene / "20",
                tile_size_px=tile_size_px,
            )

    return grids


def sha256_file(path: Path) -> str:
    """Hash one bounded provenance file or source tile in streaming chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def axis_aligned_window_for_bounds(
    bounds: tuple[float, float, float, float] | list[float],
    transform: tuple[float, ...] | list[float],
    width: int,
    height: int,
) -> list[int] | None:
    """Return a clipped ``[col, row, width, height]`` window for north-up bounds.

    The result is the pixel envelope of an input geometry's bounds, not an
    exact rasterized footprint. Pure affine arithmetic avoids driver-dependent
    coordinate-transform paths for this metadata-only grid operation.
    """
    if len(bounds) != 4 or len(transform) < 6 or width < 1 or height < 1:
        raise ValueError("Raster-window inputs have invalid dimensions")
    left, bottom, right, top = (float(value) for value in bounds)
    a, b, c, d, e, f = (float(value) for value in transform[:6])
    values = (left, bottom, right, top, a, b, c, d, e, f)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Raster-window inputs must be finite")
    if left >= right or bottom >= top or a <= 0 or e >= 0:
        raise ValueError("Expected non-empty bounds on a north-up raster")
    if not math.isclose(b, 0.0, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
            d, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Only north-up, axis-aligned rasters are supported")

    columns = ((left - c) / a, (right - c) / a)
    rows = ((bottom - f) / e, (top - f) / e)
    col0 = max(0, math.floor(min(columns)))
    row0 = max(0, math.floor(min(rows)))
    col1 = min(width, math.ceil(max(columns)))
    row1 = min(height, math.ceil(max(rows)))
    if col0 >= col1 or row0 >= row1:
        return None
    return [col0, row0, col1 - col0, row1 - row0]
