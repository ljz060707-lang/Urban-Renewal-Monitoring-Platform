"""Build a bounded, read-only-source packet for human roof/ground review.

The exported scene rasters preserve the source tile grid in EPSG:32649 for
overlay and digitizing. They are not orthorectified and do not establish
absolute ground accuracy. The reviewed-footprint layer is intentionally empty.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import fiona
import numpy as np
import rasterio
from affine import Affine
from PIL import Image, ImageDraw, ImageFont
from rasterio.features import geometry_window
from shapely.geometry import Polygon, box, mapping, shape


REVIEW_LAYER = "reviewed_ground_footprints"
REVIEW_CANDIDATE_LAYER = "ground_footprint_review_candidates"
PACKET_SCHEMA = "xian_r5.ground_footprint_review_packet.v1"
REVIEW_FIELDS = {
    "uid": "str:96",
    "actual_date": "str:16",
    "source_ref": "str:254",
    "method": "str:40",
    "independent_ground_evidence": "int",
    "reviewed": "int",
    "positional_sigma_m": "float",
    "roof_to_ground_dx_m": "float",
    "roof_to_ground_dy_m": "float",
    "reviewer_id": "str:96",
    "reviewed_at": "str:40",
    "evidence_path": "str:512",
    "evidence_sha256": "str:64",
    "physical_identity_key": "str:96",
    "identity_reviewed": "int",
}
REVIEW_CANDIDATE_FIELDS = {
    "uid": "str:96",
    "actual_date": "str:16",
    "candidate_id": "str:80",
    "source_ref": "str:254",
    "review_status": "str:48",
    "reviewer_id": "str:96",
    "reviewed_at": "str:40",
    "evidence_path": "str:512",
    "evidence_sha256": "str:64",
    "uncertainty_basis": "str:40",
    "positional_sigma_m": "float",
    "notes": "str:512",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _required_file(parent: Path, name: str) -> Path:
    path = parent / name
    if not path.is_file():
        raise FileNotFoundError(f"Incomplete fixed-case run; missing {path}")
    return path.resolve()


def _validate_sources(source_run: Path, diagnostic_path: Path,
                      qa_receipt_path: Path | None = None):
    qa_path = (qa_receipt_path.resolve() if qa_receipt_path is not None
               else _required_file(source_run, "ENGINEERING_QA.json"))
    if not qa_path.is_file():
        raise FileNotFoundError(f"Engineering QA receipt is missing: {qa_path}")
    product_path = _required_file(source_run, "PRODUCT_MANIFEST.json")
    receipt_path = _required_file(source_run, "EVIDENCE_RECEIPT.json")
    geopackage_path = _required_file(source_run, "building_lineage.gpkg")
    qa, product, receipt = map(_read_json, (qa_path, product_path, receipt_path))
    diagnostic = _read_json(diagnostic_path)

    if (qa.get("schema_version") != "xian_r5.reconstruction_engineering_qa.v1"
            or qa.get("structural_qa") != "PASS"
            or qa.get("errors") not in ([], {})):
        raise ValueError("Review packet requires a structurally passing fixed-case run")
    for key in ("r4_status_sha256_unchanged", "tracked_input_hashes_unchanged",
                "pixel_input_hashes_unchanged", "base_raster_hashes_unchanged"):
        if qa.get(key) is not True:
            raise ValueError(f"Source QA does not verify unchanged inputs: {key}")
    if (receipt.get("schema_version") != "xian_r5.grounded_fixed_case_run.v1"
            or qa.get("case_id") != receipt.get("case_id")):
        raise ValueError("Fixed-case QA and receipt are incompatible")
    if receipt.get("image_grid_alignment_applied_count") != 0:
        raise ValueError("Review packet source must not contain applied image-grid shifts")
    if product.get("crs") != "EPSG:32649" or product.get("image_alignment_field_applied") is not False:
        raise ValueError("Review packet source must use EPSG:32649 and unapplied alignment candidates")
    reference = product.get("reference_grid")
    if not isinstance(reference, dict) or not reference.get("path"):
        raise ValueError("Fixed-case product manifest has no verified 10 m reference path")
    if diagnostic.get("case_id") != receipt.get("case_id"):
        raise ValueError("Mosaic diagnostic and fixed-case run refer to different cases")
    if (diagnostic.get("parameters") or {}).get("registration_applied") is not False:
        raise ValueError("Mosaic diagnostic must explicitly state that no registration was applied")

    paths = {
        "engineering_qa": qa_path,
        "product_manifest": product_path,
        "evidence_receipt": receipt_path,
        "source_geopackage": geopackage_path,
        "mosaic_diagnostic": diagnostic_path.resolve(),
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    return qa, product, receipt, diagnostic, paths, hashes


def _read_vectors(geopackage: Path):
    roofs = []
    candidates = []
    with fiona.open(geopackage, layer="roof_candidates") as source:
        if not source.crs or source.crs.to_epsg() != 32649:
            raise ValueError("Roof candidate layer must use EPSG:32649")
        for feature in source:
            if not feature.get("geometry"):
                raise ValueError(f"Roof candidate has no geometry: {feature.id}")
            geometry = shape(feature["geometry"])
            props = dict(feature["properties"])
            uid, actual_date = props.get("uid"), props.get("actual_date")
            if (not isinstance(uid, str) or not uid.strip()
                    or not isinstance(actual_date, str) or not actual_date.strip()
                    or geometry.is_empty or not geometry.is_valid
                    or geometry.geom_type not in {"Polygon", "MultiPolygon"}):
                raise ValueError(f"Invalid dated roof candidate: {feature.id}")
            roofs.append({"geometry": geometry, "properties": props})
    if not roofs:
        raise ValueError("Fixed-case run has no roof candidates")

    with fiona.open(geopackage, layer="footprint_candidates") as source:
        if not source.crs or source.crs.to_epsg() != 32649:
            raise ValueError("Footprint candidate layer must use EPSG:32649")
        for feature in source:
            if not feature.get("geometry"):
                raise ValueError(f"Footprint candidate has no geometry: {feature.id}")
            geometry = shape(feature["geometry"])
            if geometry.is_empty or not geometry.is_valid:
                raise ValueError(f"Invalid footprint candidate geometry: {feature.id}")
            candidates.append({"geometry": geometry,
                               "properties": dict(feature["properties"])})
    if len({row["properties"]["uid"] for row in roofs}) != len(roofs):
        raise ValueError("Roof candidate UIDs must be unique in the review packet")
    return roofs, candidates


def _tile_support(transform: Affine, tile_col: int, tile_row: int):
    c0, r0 = tile_col * 256, tile_row * 256
    return Polygon([
        transform @ (c0, r0), transform @ (c0 + 256, r0),
        transform @ (c0 + 256, r0 + 256), transform @ (c0, r0 + 256),
    ])


def _raster_bounds(transform: Affine, width: int, height: int):
    corners = [transform @ point for point in (
        (0, 0), (width, 0), (width, height), (0, height))]
    return (min(point[0] for point in corners), min(point[1] for point in corners),
            max(point[0] for point in corners), max(point[1] for point in corners))


def _assemble_date_raster(actual_date: str, diagnostic: dict[str, Any],
                          output_path: Path):
    grid = diagnostic.get("grid") or {}
    tile_window = grid.get("tile_window") or {}
    try:
        x0, y0 = int(tile_window["x0"]), int(tile_window["y0"])
        nx, ny = int(tile_window["nx"]), int(tile_window["ny"])
        transform = Affine(*[float(value) for value in grid["pixel_transform"][:6]])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("Mosaic diagnostic has an invalid tile grid") from exc
    if nx < 1 or ny < 1 or grid.get("crs") != "EPSG:32649":
        raise ValueError("Mosaic diagnostic must define a positive EPSG:32649 tile grid")

    epoch = (diagnostic.get("epochs") or {}).get(actual_date)
    records = (diagnostic.get("submeter_tiles") or {}).get(actual_date)
    if not isinstance(epoch, dict) or not isinstance(records, list):
        raise ValueError(f"Mosaic diagnostic has no exact-date tile inventory: {actual_date}")
    wanted = nx * ny
    if (epoch.get("wanted_tiles") != wanted
            or epoch.get("exact_date_tiles_read") != wanted
            or epoch.get("exact_date_tile_coverage_fraction") != 1.0
            or len(records) != wanted):
        raise ValueError(f"Exact-date tile coverage is incomplete for {actual_date}")

    rgb = np.zeros((ny * 256, nx * 256, 3), dtype=np.uint8)
    valid = np.zeros((ny * 256, nx * 256), dtype=bool)
    tile_receipts = []
    tile_supports = []
    seen = set()
    for record in records:
        try:
            tile_x, tile_y = (int(value) for value in str(record["tile_id"]).split(":"))
            source_path = Path(record["path"]).resolve()
            expected_hash = str(record["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed exact-date tile record for {actual_date}") from exc
        if (tile_x, tile_y) in seen:
            raise ValueError(f"Duplicate tile in exact-date inventory: {actual_date}/{tile_x}:{tile_y}")
        seen.add((tile_x, tile_y))
        if not (x0 <= tile_x < x0 + nx and y0 <= tile_y < y0 + ny):
            raise ValueError(f"Tile lies outside the recorded mosaic grid: {actual_date}/{tile_x}:{tile_y}")
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise ValueError(f"Source tile hash mismatch: {source_path}")
        with Image.open(source_path) as source:
            tile = np.asarray(source.convert("RGB"))
        if tile.shape != (256, 256, 3):
            raise ValueError(f"Source tile must be 256x256 RGB: {source_path}")
        if _sha256(source_path) != expected_hash:
            raise RuntimeError(f"Source tile changed while reading: {source_path}")
        col = tile_x - x0
        row = (y0 + ny - 1) - tile_y
        y1, x1 = row * 256, col * 256
        rgb[y1:y1 + 256, x1:x1 + 256] = tile
        valid[y1:y1 + 256, x1:x1 + 256] = True
        tile_receipts.append({"tile_id": f"{tile_x}:{tile_y}",
                              "path": str(source_path), "sha256": expected_hash})
        tile_supports.append((f"{tile_x}:{tile_y}",
                              _tile_support(transform, col, row)))
    if len(seen) != wanted or not bool(valid.all()):
        raise ValueError(f"Mosaic contains missing pixel support for {actual_date}")

    profile = {
        "driver": "GTiff", "width": rgb.shape[1], "height": rgb.shape[0],
        "count": 3, "dtype": "uint8", "crs": grid["crs"], "transform": transform,
        "compress": "deflate", "predictor": 2, "tiled": True,
        "blockxsize": 256, "blockysize": 256,
        "photometric": "RGB", "BIGTIFF": "IF_SAFER",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as sink:
        sink.write(np.moveaxis(rgb, 2, 0))
        sink.update_tags(
            PRODUCT_KIND="RAW_ACTUAL_DATE_SCENE_REVIEW_CONTEXT",
            ACTUAL_DATE=actual_date,
            ACTUAL_DATE_SOURCE="TILE_ACTUAL_DATE_REGISTRY",
            CRS_SEMANTICS="SOURCE_Z20_TILE_GRID_TRANSFORMED_TO_EPSG32649",
            ORTHORECTIFIED="false",
            GROUND_FOOTPRINT_CONFIRMED="false",
            POSITIONAL_ACCURACY_CLAIMED="false",
            ROOF_TO_GROUND_CORRECTION="false",
            TILE_COVERAGE="COMPLETE_EXACT_DATE_TILES",
        )
    return {"path": output_path, "transform": transform,
            "width": rgb.shape[1], "height": rgb.shape[0],
            "bounds": _raster_bounds(transform, rgb.shape[1], rgb.shape[0]),
            "tiles": tile_receipts, "tile_supports": tile_supports,
            "sha256": _sha256(output_path)}


def _write_base_context(base_path: Path, bounds: tuple[float, float, float, float],
                        output_path: Path):
    before_hash = _sha256(base_path)
    with rasterio.open(base_path) as source:
        if (source.crs is None or source.crs.to_epsg() != 32649
                or source.count < 3 or any(abs(value - 10.0) > 1e-8 for value in source.res)):
            raise ValueError("Reference context must be the provided EPSG:32649 10 m RGB base")
        try:
            window = geometry_window(source, [mapping(box(*bounds))],
                                     pad_x=1, pad_y=1, boundless=False)
        except Exception as exc:
            raise ValueError("B04 mosaic footprint falls outside the 10 m reference raster") from exc
        window = window.round_offsets().round_lengths()
        if (window.width < 1 or window.height < 1 or window.col_off < 0 or window.row_off < 0
                or window.col_off + window.width > source.width
                or window.row_off + window.height > source.height):
            raise ValueError("Computed 10 m context window exceeds the reference raster")
        data = source.read((1, 2, 3), window=window)
        profile = source.profile.copy()
        profile.update(width=int(window.width), height=int(window.height),
                       transform=source.window_transform(window), count=3,
                       compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        tags = source.tags()
    if _sha256(base_path) != before_hash:
        raise RuntimeError("10 m base raster changed while creating review context")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as sink:
        sink.write(data)
        sink.update_tags(
            PRODUCT_KIND="10M_BASE_REVIEW_CONTEXT_ONLY",
            SOURCE_PATH=str(base_path),
            SOURCE_SHA256=before_hash,
            SOURCE_ACQUISITION_DATE_STATUS=tags.get(
                "ACQUISITION_DATE_STATUS", "UNKNOWN_ANNUAL_FILE_LABEL_ONLY"),
            REFERENCE_ROLE="FIXED_GRID_CONTEXT_NOT_SUBMETER_GROUND_CONTROL",
            POSITIONAL_ACCURACY_CLAIMED="false",
        )
    return {"path": output_path, "source_path": str(base_path),
            "source_sha256": before_hash, "sha256": _sha256(output_path)}


def _draw_outline(draw: ImageDraw.ImageDraw, geometry, inverse: Affine,
                  crop_left: int, crop_top: int, image_top: int,
                  color: tuple[int, int, int], width: int):
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    for polygon in polygons:
        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            points = []
            for x, y in ring.coords:
                col, row = inverse @ (x, y)
                points.append((int(round(col - crop_left)),
                               int(round(row - crop_top + image_top))))
            if len(points) >= 2:
                draw.line(points, fill=color, width=width, joint="curve")


def _write_review_card(uid: str, focus_date: str, roof: dict,
                       roofs_by_date: dict[str, list[dict]],
                       candidates: list[dict], scenes: dict[str, dict],
                       dates: list[str], output_path: Path, *, patch_size: int = 512):
    focus_geometry = roof["geometry"]
    center = focus_geometry.centroid
    font = ImageFont.load_default()
    header, footer = 56, 42
    card = Image.new("RGB", (patch_size * len(dates), patch_size + header + footer),
                     (24, 27, 32))
    canvas = ImageDraw.Draw(card)
    title = f"{uid} | focus acquisition {focus_date} | REVIEW ONLY - NO GROUND FOOTPRINT ASSERTED"
    canvas.text((8, 8), title[:120], fill=(245, 245, 245), font=font)
    colors = [(60, 225, 235), (255, 213, 50), (110, 255, 110),
              (230, 110, 250), (255, 150, 70)]

    candidate_geometries = [
        item["geometry"] for item in candidates
        if item["properties"].get("uid") == uid
    ]
    for index, actual_date in enumerate(dates):
        scene = scenes[actual_date]
        inverse = ~scene["transform"]
        center_col, center_row = inverse @ (center.x, center.y)
        crop_left = int(math.floor(center_col - patch_size / 2))
        crop_top = int(math.floor(center_row - patch_size / 2))
        raw = Image.fromarray(scene["rgb"])
        panel = Image.new("RGB", (patch_size, patch_size), (8, 12, 18))
        src_left, src_top = max(crop_left, 0), max(crop_top, 0)
        src_right = min(crop_left + patch_size, raw.width)
        src_bottom = min(crop_top + patch_size, raw.height)
        if src_right > src_left and src_bottom > src_top:
            panel.paste(raw.crop((src_left, src_top, src_right, src_bottom)),
                        (src_left - crop_left, src_top - crop_top))
        x_offset = index * patch_size
        card.paste(panel, (x_offset, header))
        canvas = ImageDraw.Draw(card)
        canvas.text((x_offset + 8, 34), actual_date, fill=(255, 255, 255), font=font)

        for roof_record in roofs_by_date.get(actual_date, []):
            geom = roof_record["geometry"]
            point = geom.centroid
            pcol, prow = inverse @ (point.x, point.y)
            if (crop_left - 16 <= pcol <= crop_left + patch_size + 16
                    and crop_top - 16 <= prow <= crop_top + patch_size + 16):
                props = roof_record["properties"]
                color = ((255, 213, 50) if props.get("uid") == uid
                         else colors[index % len(colors)])
                _draw_outline(canvas, geom, inverse, crop_left, crop_top,
                              header, color, 3 if props.get("uid") == uid else 2)
        for candidate_geometry in candidate_geometries:
            pcol, prow = inverse @ (candidate_geometry.centroid.x,
                                   candidate_geometry.centroid.y)
            if (crop_left - 16 <= pcol <= crop_left + patch_size + 16
                    and crop_top - 16 <= prow <= crop_top + patch_size + 16):
                _draw_outline(canvas, candidate_geometry, inverse, crop_left,
                              crop_top, header, (255, 80, 170), 2)

    footer_text = ("Cyan/date color: roof candidates  |  Yellow: focus roof  |  Magenta: external roof candidate  |  "
                   "Raw tile-grid view; not orthorectified; positional accuracy unknown")
    canvas.text((8, header + patch_size + 10), footer_text, fill=(230, 230, 230), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    card.save(output_path, format="PNG", optimize=True)
    return _sha256(output_path)


def _write_review_geopackage(path: Path, roofs: list[dict], candidates: list[dict],
                             tile_ids_by_uid: dict[str, list[str]]):
    roof_fields = {
        "uid": "str:96", "actual_date": "str:16", "nominal_epoch": "str:16",
        "source_ref": "str:254", "geometry_basis": "str:40",
        "footprint_status": "str:40", "physical_id": "int",
        "image_tile_ids_json": "str:1024", "review_state": "str:24",
    }
    candidate_fields = {
        "uid": "str:96", "actual_date": "str:16", "candidate_id": "str:80",
        "source_ref": "str:254", "source_date": "str:16",
        "source_epoch_label": "str:32", "source_geometry_semantics": "str:48",
        "source_date_status": "str:48", "source_date_candidates_json": "str:256",
        "source_scene_linkage_status": "str:80", "association": "str:40",
        "status": "str:40",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with fiona.open(path, "w", driver="GPKG", layer="roof_candidates",
                    schema={"geometry": "Unknown", "properties": roof_fields},
                    crs="EPSG:32649") as sink:
        for record in roofs:
            props = record["properties"]
            sink.write({"geometry": mapping(record["geometry"]), "properties": {
                "uid": props["uid"], "actual_date": props["actual_date"],
                "nominal_epoch": props.get("epoch"),
                "source_ref": props.get("source_ref"),
                "geometry_basis": "VISIBLE_ROOF_CANDIDATE",
                "footprint_status": props.get("status") or "ROOF_ONLY_UNRESOLVED",
                "physical_id": props.get("physical_id"),
                "image_tile_ids_json": json.dumps(tile_ids_by_uid[props["uid"]]),
                "review_state": "NOT_REVIEWED",
            }})
    with fiona.open(path, "w", driver="GPKG", layer="footprint_candidates",
                    schema={"geometry": "Unknown", "properties": candidate_fields},
                    crs="EPSG:32649") as sink:
        for record in candidates:
            props = record["properties"]
            sink.write({"geometry": mapping(record["geometry"]), "properties": {
                "uid": props.get("uid"), "actual_date": props.get("actual_date"),
                "candidate_id": props.get("candidate_id"),
                "source_ref": props.get("source_ref"), "source_date": props.get("source_date"),
                "source_epoch_label": props.get("source_epoch_label"),
                "source_geometry_semantics": props.get("source_geometry_semantics"),
                "source_date_status": props.get("source_date_status"),
                "source_date_candidates_json": props.get("source_date_candidates_json"),
                "source_scene_linkage_status": props.get("source_scene_linkage_status"),
                "association": props.get("association"),
                "status": "UNVERIFIED_CANDIDATE",
            }})
    with fiona.open(path, "w", driver="GPKG", layer=REVIEW_LAYER,
                    schema={"geometry": "Polygon", "properties": REVIEW_FIELDS},
                    crs="EPSG:32649"):
        pass
    with fiona.open(path, "w", driver="GPKG", layer=REVIEW_CANDIDATE_LAYER,
                    schema={"geometry": "Polygon", "properties": REVIEW_CANDIDATE_FIELDS},
                    crs="EPSG:32649"):
        pass


def _instructions(case_id: str, review_gpkg: str, scene_names: list[str]) -> str:
    scenes = "\n".join(f"- `{name}`" for name in scene_names)
    return f"""# Ground-footprint review packet — {case_id}

This packet is a review aid, not a corrected building product. The date-specific
scene GeoTIFFs are assembled from the exact-date z20 source tiles and retain the
source tile-grid transform in EPSG:32649. They are **not orthorectified** and
their absolute positional accuracy is unknown. The 10 m base clip is context
only and is not sub-meter ground control.

## Files

{scenes}
- `base_10m_context.tif` — clipped fixed-grid context; not building ground truth.
- `{review_gpkg}#roof_candidates` — source roof polygons, explicitly not ground footprints.
- `{review_gpkg}#footprint_candidates` — unverified external candidates, including the user-reported 2019 image roof boundaries.
- `{review_gpkg}#{REVIEW_CANDIDATE_LAYER}` — editable manual visible-boundary candidates. These remain unverified and are written only to candidate outputs.
- `{review_gpkg}#{REVIEW_LAYER}` — empty editable layer with the reconstruction runner's required review schema.
- `cards/` — same-location three-date visual context with candidate outlines; use the raw GeoTIFFs for tracing.
- `review_index.csv` and `REVIEW_PACKET.json` — dates, tile hashes, geometry lineage, and output hashes.

## Candidate-only visual review

If a ground-edge interpretation is useful but the source image has no independent
absolute control, digitize it in `{REVIEW_CANDIDATE_LAYER}`. Use the exact `uid`
and `actual_date` from `roof_candidates`, a unique `candidate_id`,
`review_status=VISIBLE_GROUND_BOUNDARY_CANDIDATE`, your reviewer ID and a
timezone-bearing `reviewed_at`. Set `evidence_path` to the exact-date scene
GeoTIFF and copy its SHA-256 from `REVIEW_PACKET.json`. Set
`uncertainty_basis=UNKNOWN_ABSOLUTE_GEOREF` (or
`RELATIVE_SOURCE_GRID_ONLY`) and leave `positional_sigma_m` null. This records
what is visible without inventing absolute accuracy. The reconstruction runner
keeps it in the unverified candidate vector/TIF products; it does not confirm a
ground footprint, physical ID, presence, or change.

## Formal footprint review and return

In QGIS, load the scene GeoTIFFs and the GeoPackage. Add a feature to
`{REVIEW_LAYER}` only when an independently interpretable visible ground boundary
is present and its total positional uncertainty (including image georeferencing,
not only digitizing precision) can be supported at 2 m or better. Draw the
ground boundary, do not translate the roof outline. If the boundary or absolute
position is not reliable, leave that observation unresolved and do not add a
feature.

For each accepted feature, populate every field in the layer schema. Use
`method=VISIBLE_GROUND_BOUNDARY`, `independent_ground_evidence=1`,
`reviewed=1`, a timezone-bearing `reviewed_at`, the actual date and exact UID
from `roof_candidates`, and `positional_sigma_m` in [0, 2]. Set
`evidence_path` to the corresponding `scenes/<date>_RGB_UTM.tif` file and
`evidence_sha256` to its value in `REVIEW_PACKET.json`. Keep
`physical_identity_key` null and `identity_reviewed=0` unless cross-date physical
identity is independently resolved. Then pass this GeoPackage to
`--reviewed-ground-footprint-layer`; the runner rechecks all hashes and gates.

An empty review layer is valid and yields zero confirmed ground footprints.
No image shift or roof-to-ground correction is applied by this exporter.
"""


def export_ground_footprint_review_packet(
    source_run_dir: Path,
    mosaic_diagnostic_path: Path,
    output_dir: Path,
    *,
    patch_size_px: int = 512,
    qa_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Export a fixed-case imagery/vector review package without changing inputs."""
    source_run = Path(source_run_dir).resolve()
    diagnostic_path = Path(mosaic_diagnostic_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite review packet: {output_dir}")
    if patch_size_px < 128 or patch_size_px % 16:
        raise ValueError("Review-card patch size must be a multiple of 16 and at least 128 px")

    qa, product, receipt, diagnostic, source_paths, source_hashes = _validate_sources(
        source_run, diagnostic_path, qa_receipt_path)
    roofs, candidates = _read_vectors(source_paths["source_geopackage"])
    dates = product.get("actual_date_order")
    if (not isinstance(dates, list) or len(dates) < 2
            or dates != sorted(set(dates))):
        raise ValueError("Product manifest must provide sorted unique actual dates")
    if set(dates) != set((diagnostic.get("epochs") or {}).keys()):
        raise ValueError("Mosaic diagnostic dates do not match the fixed-case product dates")
    if any(record["properties"].get("actual_date") not in dates for record in roofs):
        raise ValueError("A roof candidate has no matching actual-date mosaic")

    base_path = Path(product["reference_grid"]["path"]).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"10 m reference raster is missing: {base_path}")
    base_hash = _sha256(base_path)
    sources_hashes_before = dict(source_hashes)
    sources_hashes_before["reference_raster"] = base_hash
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.tmp-",
                                     dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        scenes: dict[str, dict[str, Any]] = {}
        scene_records = []
        for actual_date in dates:
            path = stage / "scenes" / f"{actual_date}_RGB_UTM.tif"
            scene = _assemble_date_raster(actual_date, diagnostic, path)
            scene["rgb"] = None
            with rasterio.open(path) as source:
                scene["rgb"] = np.moveaxis(source.read(), 0, 2)
            scenes[actual_date] = scene
            scene_records.append({
                "actual_date": actual_date,
                "path": path.relative_to(stage).as_posix(),
                "sha256": scene["sha256"], "crs": "EPSG:32649",
                "transform": list(scene["transform"]),
                "width": scene["width"], "height": scene["height"],
                "tile_count": len(scene["tiles"]), "tiles": scene["tiles"],
                "orthorectified": False, "positional_accuracy_claimed": False,
            })

        all_bounds = [scene["bounds"] for scene in scenes.values()]
        bounds = (min(item[0] for item in all_bounds),
                  min(item[1] for item in all_bounds),
                  max(item[2] for item in all_bounds),
                  max(item[3] for item in all_bounds))
        base_context = _write_base_context(
            base_path, bounds, stage / "base_10m_context.tif")

        roofs_by_date: dict[str, list[dict]] = {actual_date: [] for actual_date in dates}
        for record in roofs:
            roofs_by_date[record["properties"]["actual_date"]].append(record)
        tile_ids_by_uid = {}
        for record in roofs:
            props, geometry = record["properties"], record["geometry"]
            actual_date = props["actual_date"]
            tile_ids_by_uid[props["uid"]] = [
                tile_id for tile_id, support in scenes[actual_date]["tile_supports"]
                if geometry.intersects(support)
            ]
        _write_review_geopackage(stage / "ground_footprint_review.gpkg",
                                 roofs, candidates, tile_ids_by_uid)

        card_records = []
        for record in roofs:
            props = record["properties"]
            uid = str(props["uid"])
            safe_uid = re.sub(r"[^A-Za-z0-9._-]+", "_", uid).strip("._-") or "observation"
            card_path = stage / "cards" / f"{safe_uid}.png"
            card_hash = _write_review_card(
                uid, props["actual_date"], record, roofs_by_date, candidates,
                scenes, dates, card_path, patch_size=patch_size_px)
            card_records.append({"uid": uid, "actual_date": props["actual_date"],
                                 "nominal_epoch_label": props.get("epoch"),
                                 "image_tile_ids": tile_ids_by_uid[uid],
                                 "card_path": card_path.relative_to(stage).as_posix(),
                                 "card_sha256": card_hash})

        with (stage / "review_index.csv").open("w", encoding="utf-8-sig", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=(
                "uid", "actual_date", "nominal_epoch_label", "image_tile_ids_json",
                "review_card", "review_card_sha256", "raw_scene_raster",
                "raw_scene_sha256", "review_state"))
            writer.writeheader()
            for row in card_records:
                actual_date = row["actual_date"]
                writer.writerow({
                    "uid": row["uid"], "actual_date": actual_date,
                    "nominal_epoch_label": row["nominal_epoch_label"],
                    "image_tile_ids_json": json.dumps(row["image_tile_ids"]),
                    "review_card": row["card_path"],
                    "review_card_sha256": row["card_sha256"],
                    "raw_scene_raster": next(item["path"] for item in scene_records
                                              if item["actual_date"] == actual_date),
                    "raw_scene_sha256": next(item["sha256"] for item in scene_records
                                              if item["actual_date"] == actual_date),
                    "review_state": "NOT_REVIEWED",
                })

        instructions_path = stage / "REVIEW_INSTRUCTIONS.md"
        instructions_path.write_text(_instructions(
            receipt["case_id"], "ground_footprint_review.gpkg",
            [item["path"] for item in scene_records]), encoding="utf-8")
        review_gpkg = stage / "ground_footprint_review.gpkg"
        output_files = [
            *[stage / item["path"] for item in scene_records],
            stage / "base_10m_context.tif", review_gpkg,
            stage / "review_index.csv", instructions_path,
            *[stage / item["card_path"] for item in card_records],
        ]
        output_hashes = {path.relative_to(stage).as_posix(): _sha256(path)
                         for path in output_files}

        current_source_hashes = {name: _sha256(path)
                                 for name, path in source_paths.items()}
        current_source_hashes["reference_raster"] = _sha256(base_path)
        if current_source_hashes != sources_hashes_before:
            raise RuntimeError("One or more fixed-case review inputs changed during export")
        for scene in scene_records:
            for tile in scene["tiles"]:
                tile_path = Path(tile["path"])
                if _sha256(tile_path) != tile["sha256"]:
                    raise RuntimeError(f"Source tile changed during review export: {tile_path}")

        manifest = {
            "schema_version": PACKET_SCHEMA,
            "case_id": receipt["case_id"],
            "scope": "FIXED_CASE_REVIEW_PACKET_ONLY",
            "actual_date_order": dates,
            "source_run_dir": str(source_run),
            "mosaic_diagnostic_path": str(diagnostic_path),
            "source_inputs": {
                name: {"path": str(path), "sha256": sources_hashes_before[name]}
                for name, path in source_paths.items()
            },
            "reference_10m": {
                "path": str(base_path), "sha256": base_hash,
                "context_clip": "base_10m_context.tif",
                "role": "FIXED_GRID_CONTEXT_ONLY_NOT_SUBMETER_GROUND_CONTROL",
            },
            "scene_rasters": scene_records,
            "roof_candidate_count": len(roofs),
            "external_footprint_candidate_association_count": len(candidates),
            "source_geometry_semantics": sorted({
                str(record["properties"].get("source_geometry_semantics") or "UNKNOWN")
                for record in candidates}),
            "source_geometry_is_ground_truth": False,
            "reviewed_ground_footprints": {
                "gpkg": "ground_footprint_review.gpkg",
                "layer": REVIEW_LAYER,
                "template_feature_count": 0,
                "runner_acceptance_contract": "xian_r5.reviewed_ground_footprints.v1",
            },
            "reviewed_ground_footprint_candidates": {
                "gpkg": "ground_footprint_review.gpkg",
                "layer": REVIEW_CANDIDATE_LAYER,
                "template_feature_count": 0,
                "runner_import_contract": "xian_r5.reviewed_ground_footprint_candidates.v1",
                "formal_ground_footprint_confirmed": False,
                "positional_accuracy": "UNKNOWN_OR_RELATIVE_ONLY; NUMERIC_SIGMA_NOT_CLAIMED",
            },
            "ground_footprint_correction_applied": False,
            "identity_or_change_state_created": False,
            "review_card_count": len(card_records),
            "review_cards": card_records,
            "review_state": "UNREVIEWED_CANDIDATES_ONLY",
            "positional_accuracy_claimed": False,
            "source_hash_recheck": "PASS",
            "output_files_sha256": output_hashes,
        }
        (stage / "REVIEW_PACKET.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    return manifest
