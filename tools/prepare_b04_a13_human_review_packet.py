#!/usr/bin/env python3
"""Prepare the frozen, a13-only R5 human FIT/HOLDOUT evidence packet.

This tool never writes to source imagery or the candidate GeoPackage. The
candidate displacement is used only to render a clearly labelled review overlay.
It does not create ties, review provenance, a disposition, or downstream products.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Use only the already-installed, run-scoped geospatial runtime dependencies.
sys.path.insert(
    0, r"C:\Users\ljz52\AppData\Local\Temp\xian-r5-contract-test-deps-cp312-20261001"
)
sys.path.insert(0, r"C:\Users\ljz52\.cache\codex-runtimes\xian-geospatial-temp")
import numpy as np
import pyogrio
import rasterio
from PIL import Image, ImageDraw
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, from_bounds


BASE = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\19_ground_registration_evidence")
PARENT = BASE / "B04_20261001_01"
PREVIEW_DIR = BASE / "B04_20260929_01" / "04_manual_review" / "RELATIVE_BASE_SCREEN_20260929_C"
PREVIEW_MANIFEST = PREVIEW_DIR / "PREVIEW_MANIFEST.json"
CANDIDATE = Path(
    r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\18_reconstruction"
    r"\B04_target_spatial_composition_grid2_4x4_20260929_01\spatial_alignment_candidates.gpkg"
)
REFERENCE = Path(r"F:\xian\xian底图\xian搴曞浘\Xian_2019_TrueColor.tif")
DATE_TABLE = Path(r"F:\xian\xian\date_audit\tables\tile_actual_date.csv")
TILE_GRID_SOURCE = Path(__file__).resolve().parent / "assess_xian_external_footprints.py"
RUN = BASE / "B04_20261001_02"
PACKET = RUN / "04_manual_review" / "A13_FIRST_PRIORITY_V1"
REPORT_PATH = RUN / "13_reports" / "B04_FORMAL_HUMAN_FIT_HOLDOUT_PACKET_REPORT.json"
RECEIPT_PATH = RUN / "14_receipts" / "B04_FORMAL_HUMAN_FIT_HOLDOUT_PACKET_RECEIPT.json"
OUTPUT_MANIFEST_PATH = RUN / "14_receipts" / "B04_FORMAL_HUMAN_FIT_HOLDOUT_OUTPUT_SHA256.json"

DATES = ("2018-04-29", "2018-11-01", "2019-12-02")
SCENES = {"2018-04-29": "xa180429", "2018-11-01": "xa190101", "2019-12-02": "xa201231"}
VECTORS = {
    "2018-04-29": (11.564963541962188, 0.7880925058356899),
    "2018-11-01": (10.441276772758874, -5.83664230003489),
    "2019-12-02": (7.866473980292583, -2.84459660384605),
}
EXPECTED_CANDIDATE_SHA = "8c50d60e3a72db1cbf0b8d2af113908b4595c39f255564193271de9882d5a0ea"
EXPECTED_REFERENCE_SHA = "63fe103fa6bc783202e8ca2597ffd7e4b1f249e3789e4114e55b0d302185a8b6"
EXPECTED_DATE_TABLE_SHA = "e3db9e2e3347b26a43101a97e2019c2683834784b564fe2cb492f322ce96fad4"
EXPECTED_SUPPORT_SHA = "810018296b1ce5fe793494ecb22e5ef3f03cf004692c05ff8f8472759fb5aba4"

# Existing R5 local tile-grid mapping used for relative display only.
ORIGIN_X, ORIGIN_Y = 420534, 311895
ORIGIN_LON, ORIGIN_LAT = 108.75778198242188, 34.16130065917969
STEP_DEG = 0.0006866455078125
DISPLAY_RES_M = 0.25
DISPLAY_MARGIN_M = 20.0
TIE_FIELDS = [
    "tie_id", "field_id", "cell_id", "moving_date", "FIT_or_HOLDOUT",
    "independence_group", "feature_type", "ge_x", "ge_y", "reference_x",
    "reference_y", "candidate_dx_m", "candidate_dy_m", "residual_dx_m",
    "residual_dy_m", "residual_m", "ge_localization_bound_m",
    "reference_localization_bound_m", "adjusted_bound_m", "localization_basis",
    "evidence_annotation", "evidence_path", "evidence_sha256",
    "within_exact_support", "precision_eligible",
    "precision_ineligibility_reason", "feature_class",
    "stable_nonbuilding_ground_evidence", "stability_basis",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def relative(path: Path) -> str:
    return path.relative_to(PACKET).as_posix()


def main() -> None:
    if RUN.exists():
        raise FileExistsError(f"Append-only run exists; refusing overwrite: {RUN}")
    required = [
    PREVIEW_MANIFEST,
    CANDIDATE,
    REFERENCE,
    DATE_TABLE,
    TILE_GRID_SOURCE,
    Path(__file__).resolve(),
        PARENT / "13_reports" / "R5_RELATIVE_ALIGNMENT_REVIEW_CONTRACT_REPORT.json",
        PARENT / "13_reports" / "R5_RELATIVE_ALIGNMENT_REVIEW_CONTRACT_TEST_BOUNDARY_ADDENDUM.json",
        PARENT / "14_receipts" / "R5_RELATIVE_ALIGNMENT_REVIEW_CONTRACT_RECEIPT.json",
        PARENT / "14_receipts" / "R5_RELATIVE_ALIGNMENT_REVIEW_CONTRACT_TEST_BOUNDARY_RECEIPT.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing audited inputs: " + "; ".join(missing))

    inputs = {str(path): sha256(path) for path in required}
    if inputs[str(CANDIDATE)] != EXPECTED_CANDIDATE_SHA:
        raise RuntimeError("Frozen candidate GeoPackage hash changed")
    if inputs[str(REFERENCE)] != EXPECTED_REFERENCE_SHA:
        raise RuntimeError("Existing reference raster hash changed")
    if inputs[str(DATE_TABLE)] != EXPECTED_DATE_TABLE_SHA:
        raise RuntimeError("Actual-date evidence table hash changed")

    manifest = json.loads(PREVIEW_MANIFEST.read_text(encoding="utf-8-sig"))
    records = [
        item for item in manifest["records"]
        if str(item.get("cell_id", "")).startswith("c0007_a00013_")
    ]
    if len(records) != 3 or {item["date"] for item in records} != set(DATES):
        raise RuntimeError("Strict a13-only manifest did not resolve exactly three date-cells")
    manifest_by_date = {item["date"]: item for item in records}

    candidates = pyogrio.read_dataframe(
        CANDIDATE,
        layer="spatial_alignment_candidates",
        where="cell_id LIKE 'c0007_a00013_%'",
    )
    if len(candidates) != 3 or set(str(value) for value in candidates["actual_date"]) != set(DATES):
        raise RuntimeError("Strict a13-only candidate query did not resolve exactly three dates")
    candidate_by_date = {str(row["actual_date"]): row for _, row in candidates.iterrows()}
    for date in DATES:
        candidate = candidate_by_date[date]
        record = manifest_by_date[date]
        dx, dy = VECTORS[date]
        if str(candidate["cell_id"]) != f"c0007_a00013_{date}":
            raise RuntimeError(f"Unexpected cell ID for {date}")
        if float(candidate["candidate_dx_m"]) != dx or float(candidate["candidate_dy_m"]) != dy:
            raise RuntimeError(f"Frozen candidate vector mismatch for {date}")
        if float(record["candidate_dx_m"]) != dx or float(record["candidate_dy_m"]) != dy:
            raise RuntimeError(f"Preview candidate vector mismatch for {date}")
        if str(candidate["status"]) != "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY":
            raise RuntimeError(f"Candidate is not screen-only for {date}")
        if any(int(candidate[field]) != 0 for field in ("reviewed", "ground_evidence", "applied")):
            raise RuntimeError(f"Candidate review/application flags are not zero for {date}")
        if hashlib.sha256(candidate.geometry.wkb).hexdigest() != EXPECTED_SUPPORT_SHA:
            raise RuntimeError(f"Frozen support WKB mismatch for {date}")
        if not candidate.geometry.is_valid:
            raise RuntimeError(f"Invalid exact support geometry for {date}")
        if date != DATES[0] and candidate.geometry.wkb != candidate_by_date[DATES[0]].geometry.wkb:
            raise RuntimeError("Support geometry differs across the selected dates")

    wanted_names = {
        f"gesh_{x}_{y}_20.jpg"
        for x in range(420797, 420801)
        for y in range(312173, 312177)
    }
    date_rows: dict[tuple[str, str], dict[str, str]] = {}
    scene_to_date = {scene: date for date, scene in SCENES.items()}
    with DATE_TABLE.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("task_name") not in scene_to_date or row.get("filename") not in wanted_names:
                continue
            date = scene_to_date[row["task_name"]]
            key = date, row["filename"]
            if key in date_rows:
                raise RuntimeError(f"Duplicate actual-date evidence row for {key}")
            if (
                row.get("actual_capture_date") != date
                or row.get("date_conflict") != "0"
                or row.get("tile_present") != "1"
                or row.get("tile_status") != "actual_date_recovered"
            ):
                raise RuntimeError(f"Unresolved/conflicting actual date for {key}")
            date_rows[key] = row
    if len(date_rows) != 48 or any(sum(k[0] == date for k in date_rows) != 16 for date in DATES):
        raise RuntimeError("Expected exactly 16 actual-date evidence rows for each of three dates")

    RUN.mkdir(parents=True)
    (PACKET / "assets").mkdir(parents=True)
    (PACKET / "sidecars").mkdir(parents=True)
    (PACKET / "blank_tables").mkdir(parents=True)

    support = candidate_by_date[DATES[0]].geometry
    minx0, miny0, maxx0, maxy0 = support.bounds
    minx = math.floor((minx0 - DISPLAY_MARGIN_M) / DISPLAY_RES_M) * DISPLAY_RES_M
    miny = math.floor((miny0 - DISPLAY_MARGIN_M) / DISPLAY_RES_M) * DISPLAY_RES_M
    maxx = math.ceil((maxx0 + DISPLAY_MARGIN_M) / DISPLAY_RES_M) * DISPLAY_RES_M
    maxy = math.ceil((maxy0 + DISPLAY_MARGIN_M) / DISPLAY_RES_M) * DISPLAY_RES_M
    width = int(round((maxx - minx) / DISPLAY_RES_M))
    height = int(round((maxy - miny) / DISPLAY_RES_M))
    dst_transform = from_origin(minx, maxy, DISPLAY_RES_M, DISPLAY_RES_M)

    with rasterio.open(REFERENCE) as ref:
        if (
            ref.crs.to_string() != "EPSG:32649"
            or ref.count < 3
            or ref.transform.a != 10.0
            or ref.transform.e != -10.0
        ):
            raise RuntimeError("Reference is not the audited EPSG:32649 native 10 m RGB raster")
        float_window = from_bounds(minx, miny, maxx, maxy, transform=ref.transform)
        col0, row0 = max(0, math.floor(float_window.col_off)), max(0, math.floor(float_window.row_off))
        col1 = min(ref.width, math.ceil(float_window.col_off + float_window.width))
        row1 = min(ref.height, math.ceil(float_window.row_off + float_window.height))
        native_window = Window(col0, row0, col1 - col0, row1 - row0)
        native_rgb = np.transpose(ref.read(indexes=(1, 2, 3), window=native_window), (1, 2, 0))
        native_png = PACKET / "assets" / "reference_native_10m_pixels.png"
        Image.fromarray(native_rgb, mode="RGB").save(native_png, format="PNG", optimize=False)

        ref_display_bands = np.zeros((3, height, width), dtype=np.uint8)
        for band in range(1, 4):
            reproject(
                source=rasterio.band(ref, band),
                destination=ref_display_bands[band - 1],
                src_transform=ref.transform,
                src_crs=ref.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:32649",
                resampling=Resampling.nearest,
            )
        ref_display = np.transpose(ref_display_bands, (1, 2, 0))
        reference_display_png = PACKET / "assets" / "reference_display_nearest_0p25m.png"
        Image.fromarray(ref_display, mode="RGB").save(
            reference_display_png, format="PNG", optimize=False
        )
        native_transform = ref.window_transform(native_window)
        reference_meta = {
            "source_path": str(REFERENCE),
            "source_sha256": inputs[str(REFERENCE)],
            "crs": ref.crs.to_string(),
            "source_transform": list(ref.transform),
            "source_width_height": [ref.width, ref.height],
            "band_order": [1, 2, 3],
            "native_pixel_size_m": [10.0, 10.0],
            "acquisition_date_status": "UNKNOWN_ANNUAL_FILE_LABEL_ONLY",
            "native_crop_png": relative(native_png),
            "native_crop_rows_cols": [int(native_rgb.shape[0]), int(native_rgb.shape[1])],
            "native_crop_row_col_height_width": [
                int(row0), int(col0), int(row1 - row0), int(col1 - col0)
            ],
            "native_crop_transform": list(native_transform),
            "display_png": relative(reference_display_png),
            "display_resampling": "nearest",
            "display_grid_pixel_size_m": [DISPLAY_RES_M, DISPLAY_RES_M],
            "display_extent_epsg32649": [minx, miny, maxx, maxy],
            "display_resolution_does_not_change_native_evidence": True,
        }
    write_json(PACKET / "assets" / "reference_display_metadata.json", reference_meta)

    support_coords = [
        [float(x), float(y)] for x, y in list(support.exterior.coords)[:-1]
    ]
    write_json(
        PACKET / "support_geometry_EPSG32649.json",
        {
            "schema_version": "xian_r5.frozen_review_support_geometry.v1",
            "coordinate_crs": "EPSG:32649",
            "candidate_source": str(CANDIDATE),
            "source_layer": "spatial_alignment_candidates",
            "source_cell_ids": [str(candidate_by_date[date]["cell_id"]) for date in DATES],
            "source_wkb_sha256": EXPECTED_SUPPORT_SHA,
            "geometry": candidates.loc[
                candidates["actual_date"].astype(str) == DATES[0]
            ].iloc[0].geometry.__geo_interface__,
        },
    )
    cell_configs = []
    tile_provenance = []
    for date in DATES:
        record = manifest_by_date[date]
        candidate = candidate_by_date[date]
        tile_dir = PACKET / "assets" / "ge_raw_tiles" / date
        tile_dir.mkdir(parents=True)
        mosaic = np.zeros((1024, 1024, 3), dtype=np.uint8)
        seen: set[tuple[int, int]] = set()
        for tile in record["source_tiles"]:
            source_path = Path(tile["path"])
            source_hash = sha256(source_path)
            if source_hash.lower() != str(tile["sha256"]).lower():
                raise RuntimeError(f"Manifest/source tile hash mismatch: {source_path}")
            match = re.search(r"gesh_(\d+)_(\d+)_20\.jpg$", source_path.name)
            if not match:
                raise RuntimeError(f"Unexpected GE source-tile name: {source_path.name}")
            tile_x, tile_y = map(int, match.groups())
            if not (420797 <= tile_x <= 420800 and 312173 <= tile_y <= 312176):
                raise RuntimeError(f"Tile outside frozen 4x4 window: {source_path}")
            if (tile_x, tile_y) in seen:
                raise RuntimeError(f"Duplicate tile: {source_path}")
            seen.add((tile_x, tile_y))
            date_row = date_rows.get((date, source_path.name))
            if date_row is None:
                raise RuntimeError(f"No actual-date log row for {date} {source_path.name}")
            with Image.open(source_path) as image:
                tile_array = np.asarray(image.convert("RGB"))
            if tile_array.shape != (256, 256, 3):
                raise RuntimeError(f"Expected original 256x256 RGB tile: {source_path}")
            copy_path = tile_dir / f"x{tile_x}_y{tile_y}_z20.jpg"
            shutil.copyfile(source_path, copy_path)
            if sha256(copy_path) != source_hash:
                raise RuntimeError(f"Source tile copy hash mismatch: {copy_path}")
            col, row = (tile_x - 420797) * 256, (312176 - tile_y) * 256
            mosaic[row:row + 256, col:col + 256] = tile_array
            tile_provenance.append({
                "actual_date": date,
                "scene_id": record["scene_id"],
                "tile_x": tile_x,
                "tile_y": tile_y,
                "zoom": 20,
                "source_path": str(source_path),
                "source_sha256": source_hash,
                "packet_copy_path": copy_path.relative_to(PACKET).as_posix(),
                "packet_copy_sha256": sha256(copy_path),
                "actual_capture_date": date_row["actual_capture_date"],
                "date_source": date_row["date_source"],
                "tile_status": date_row["tile_status"],
                "date_conflict": date_row["date_conflict"],
                "source_log": date_row["source_log"],
                "source_log_line_number": int(date_row["line_number"]),
            })
        if len(seen) != 16:
            raise RuntimeError(f"Expected 16 source tiles for {date}")
        raw_mosaic_path = PACKET / "assets" / f"ge_source_tiles_unresampled_{date}.png"
        Image.fromarray(mosaic, mode="RGB").save(raw_mosaic_path, format="PNG", optimize=False)

        west = ORIGIN_LON + (420797 - ORIGIN_X) * STEP_DEG
        north = ORIGIN_LAT + ((312176 + 1) - ORIGIN_Y) * STEP_DEG
        source_transform = from_origin(west, north, STEP_DEG / 256.0, STEP_DEG / 256.0)
        ge_bands = np.zeros((3, height, width), dtype=np.uint8)
        for band in range(3):
            reproject(
                source=mosaic[:, :, band],
                destination=ge_bands[band],
                src_transform=source_transform,
                src_crs="EPSG:4326",
                dst_transform=dst_transform,
                dst_crs="EPSG:32649",
                resampling=Resampling.nearest,
            )
        ge_display = np.transpose(ge_bands, (1, 2, 0))
        ge_png = PACKET / "assets" / f"ge_display_epsg32649_{date}.png"
        Image.fromarray(ge_display, mode="RGB").save(ge_png, format="PNG", optimize=False)

        dx, dy = float(candidate["candidate_dx_m"]), float(candidate["candidate_dy_m"])
        shifted_transform = Affine.translation(dx, dy) * dst_transform
        shifted = np.zeros_like(ge_bands)
        for band in range(3):
            reproject(
                source=ge_bands[band],
                destination=shifted[band],
                src_transform=shifted_transform,
                src_crs="EPSG:32649",
                dst_transform=dst_transform,
                dst_crs="EPSG:32649",
                resampling=Resampling.nearest,
            )
        source_valid = np.ones((height, width), dtype=np.uint8)
        shifted_valid = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=source_valid,
            destination=shifted_valid,
            src_transform=shifted_transform,
            src_crs="EPSG:32649",
            dst_transform=dst_transform,
            dst_crs="EPSG:32649",
            resampling=Resampling.nearest,
        )
        shifted_rgb = np.transpose(shifted, (1, 2, 0))
        overlay = ref_display.copy()
        valid = shifted_valid.astype(bool)
        overlay[valid] = np.rint(
            0.5 * ref_display[valid].astype(np.float32)
            + 0.5 * shifted_rgb[valid].astype(np.float32)
        ).astype(np.uint8)
        overlay_png = PACKET / "assets" / f"candidate_shift_visual_aid_{date}.png"
        Image.fromarray(overlay, mode="RGB").save(overlay_png, format="PNG", optimize=False)

        def support_outline(array: np.ndarray) -> Image.Image:
            panel = Image.fromarray(array, mode="RGB").copy()
            draw = ImageDraw.Draw(panel)
            points = [
                (
                    round((x - minx) / DISPLAY_RES_M),
                    round((maxy - y) / DISPLAY_RES_M),
                )
                for x, y in list(candidate.geometry.exterior.coords)
            ]
            draw.line(points, fill=(255, 25, 35), width=3, joint="curve")
            return panel

        pane_images = [
            support_outline(ge_display),
            support_outline(ref_display),
            support_outline(overlay),
        ]
        header_height = 42
        contact = Image.new(
            "RGB", (width * 3, height + header_height), (248, 249, 251)
        )
        draw = ImageDraw.Draw(contact)
        for index, title in enumerate(
            (
                "GE source tiles; UTM display only",
                "Existing reference; native 10 m",
                "Frozen candidate shift; visual aid only",
            )
        ):
            draw.text((index * width + 10, 8), title, fill=(25, 35, 50))
            contact.paste(pane_images[index], (index * width, header_height))
        contact_path = PACKET / "assets" / f"review_contact_sheet_{date}.png"
        contact.save(contact_path, format="PNG", optimize=False)

        cell_configs.append({
            "date": date,
            "sceneId": record["scene_id"],
            "cellId": str(candidate["cell_id"]),
            "fieldId": str(candidate["field_id"]),
            "candidateDxM": dx,
            "candidateDyM": dy,
            "candidateStatus": str(candidate["status"]),
            "supportAreaM2": float(candidate.geometry.area),
            "supportWkbSha256": hashlib.sha256(candidate.geometry.wkb).hexdigest(),
            "supportCoordinates": support_coords,
            "images": {
                "ge": ge_png.relative_to(PACKET).as_posix(),
                "reference": reference_display_png.relative_to(PACKET).as_posix(),
                "overlay": overlay_png.relative_to(PACKET).as_posix(),
                "rawMosaic": raw_mosaic_path.relative_to(PACKET).as_posix(),
                "contactSheet": contact_path.relative_to(PACKET).as_posix(),
            },
        })

    provenance_path = PACKET / "tile_actual_date_and_hash_provenance.csv"
    provenance_fields = [
        "actual_date", "scene_id", "tile_x", "tile_y", "zoom", "source_path",
        "source_sha256", "packet_copy_path", "packet_copy_sha256",
        "actual_capture_date", "date_source", "tile_status", "date_conflict",
        "source_log", "source_log_line_number",
    ]
    with provenance_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=provenance_fields)
        writer.writeheader()
        writer.writerows(
            sorted(tile_provenance, key=lambda item: (
                item["actual_date"], item["tile_y"], item["tile_x"]
            ))
        )

    templates = []
    for cell in cell_configs:
        date, cell_id = cell["date"], cell["cellId"]
        sidecar = {
            "schema_version": "xian_r5.relative_alignment_tie_sidecar.v2",
            "review_contract_amendment": "R5_RELATIVE_ALIGNMENT_REVIEW_CONTRACT_AMENDMENT_V2",
            "uncertainty_rule": "R5_RELATIVE_ALIGNMENT_UNCERTAINTY_RULE_V1",
            "coordinates_crs": "EPSG:32649",
            "field_id": cell["fieldId"],
            "cell_id": cell_id,
            "moving_date": date,
            "candidate_source_ref": (
                str(CANDIDATE)
                + "#layer=spatial_alignment_candidates&cell_id="
                + cell_id
            ),
            "candidate_dx_m": cell["candidateDxM"],
            "candidate_dy_m": cell["candidateDyM"],
            "reference_source": str(REFERENCE),
            "reference_source_sha256": inputs[str(REFERENCE)],
            "support_geometry_sha256": cell["supportWkbSha256"],
            "fit_max_adjusted_bound_m": None,
            "holdout_max_adjusted_bound_m": None,
            "relative_registration_uncertainty_m": None,
            "positional_uncertainty_m": None,
            "uncertainty_semantics": "RELATIVE_IMAGE_GRID_ONLY",
            "absolute_position_accuracy": "UNKNOWN",
            "ties": [],
        }
        sidecar_path = PACKET / "sidecars" / f"{cell_id}.tie_sidecar.v2.template.json"
        write_json(sidecar_path, sidecar)
        blank_csv = PACKET / "blank_tables" / f"{cell_id}.tie_input.v2.template.csv"
        with blank_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerow(TIE_FIELDS)
        templates.append({
            "cell_id": cell_id,
            "actual_date": date,
            "sidecar_template": sidecar_path.relative_to(PACKET).as_posix(),
            "sidecar_template_status": "TEMPLATE_ONLY_NOT_INGESTIBLE",
            "ties_created": 0,
            "reviewer_id_present": False,
            "reviewed_at_present": False,
            "blank_csv": blank_csv.relative_to(PACKET).as_posix(),
            "blank_csv_data_rows": 0,
        })

    config = {
        "scope": "a13_only_three_actual_date_cells",
        "coordinateCrs": "EPSG:32649",
        "displayGrid": {
            "width": width, "height": height, "resolutionM": DISPLAY_RES_M,
            "minX": minx, "maxX": maxx, "minY": miny, "maxY": maxy,
        },
        "reference": {
            "path": str(REFERENCE),
            "sha256": inputs[str(REFERENCE)],
            "nativePixelSizeM": [10.0, 10.0],
            "nativeTransform": [10.0, 0.0, 293300.0, 0.0, -10.0, 3812400.0],
            "nativeWidth": 3885,
            "nativeHeight": 3086,
            "nativeDateStatus": "UNKNOWN_ANNUAL_FILE_LABEL_ONLY",
            "nativePixelsPng": reference_meta["native_crop_png"],
        },
        "dateCells": cell_configs,
    }
    config_json = json.dumps(config, ensure_ascii=False, separators=(",", ":"))
    html_doc = HTML_TEMPLATE.replace("__CONFIG__", config_json)
    (PACKET / "review_packet.html").write_text(html_doc, encoding="utf-8")
    (PACKET / "REVIEW_INSTRUCTIONS.md").write_text(REVIEW_INSTRUCTIONS, encoding="utf-8")
    ledger = LEDGER_TEMPLATE.format(
        date_table=DATE_TABLE,
        date_table_sha=inputs[str(DATE_TABLE)],
        tile_grid_source=TILE_GRID_SOURCE,
        tile_grid_sha=inputs[str(TILE_GRID_SOURCE)],
        origin_x=ORIGIN_X,
        origin_y=ORIGIN_Y,
        origin_lon=ORIGIN_LON,
        origin_lat=ORIGIN_LAT,
        step=STEP_DEG,
        reference=REFERENCE,
        reference_sha=inputs[str(REFERENCE)],
        candidate=CANDIDATE,
        candidate_sha=inputs[str(CANDIDATE)],
        support_sha=EXPECTED_SUPPORT_SHA,
        resolution=DISPLAY_RES_M,
    )
    (PACKET / "SOURCE_AND_COORDINATE_LEDGER.md").write_text(ledger, encoding="utf-8")
    scope = {
        "schema_version": "xian_r5.a13_human_review_packet_scope.v1",
        "run_id": "B04_20261001_02",
        "freeze_id": "XIAN-R5-GROUNDREG-FREEZE-0929",
        "artifact_type": "HUMAN_REVIEW_PACKET_PREPARATION_ONLY",
        "expected_date_cell_count": 3,
        "resolved_date_cell_count": len(cell_configs),
        "exact_scope_required": True,
        "fallback_to_other_cells": False,
        "date_cells": [
            {"cell_id": cell["cellId"], "base_cell_id": "c0007_a00013", "actual_date": cell["date"]}
            for cell in cell_configs
        ],
        "candidate_vectors_frozen": True,
        "candidate_geometry_modified": False,
        "support_geometry_modified": False,
        "candidate_applied_to_production": False,
        "display_only_candidate_overlay": True,
        "native_reference_pixel_size_m": [10.0, 10.0],
        "display_interpolation": "nearest",
        "tie_records_created": 0,
        "reviewer_id_created": False,
        "reviewed_at_created": False,
        "formal_review_completed": False,
        "sidecar_template_status": "TEMPLATE_ONLY_NOT_INGESTIBLE",
        "sidecar_templates": templates,
        "absolute_position_accuracy": "UNKNOWN",
    }
    write_json(PACKET / "REVIEW_SCOPE.json", scope)

    # Recheck every named input and selected tile before writing the append-only receipt.
    inputs_after = {path: sha256(Path(path)) for path in inputs}
    if inputs_after != inputs:
        raise RuntimeError("A source artifact changed while the packet was being prepared")
    for item in tile_provenance:
        if sha256(Path(item["source_path"])) != item["source_sha256"]:
            raise RuntimeError("A raw source tile changed during packet preparation")
        if sha256(PACKET / item["packet_copy_path"]) != item["source_sha256"]:
            raise RuntimeError("A copied raw source tile failed final hash verification")

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    state = {
        "FORMAL_FIT_HOLDOUT_REVIEWS_COMPLETED": 0,
        "ACCEPTED_ALIGNMENT_CELLS": 0,
        "REJECTED_ALIGNMENT_CELLS": 0,
        "UNRESOLVED_ALIGNMENT_CELLS": 9,
        "REGISTERED_ROOFS": 0,
        "PHYSICAL_IDS": 0,
        "CANONICAL_GEOMETRIES": 0,
        "GROUND_FOOTPRINTS": 0,
        "ABSOLUTE_POSITION_ACCURACY": "UNKNOWN",
        "READY_FOR_CITYWIDE": "NO",
        "ACCEPTED_CELL_PRE_APPLICATION_REGRESSION_GATE": "NOT_RUN",
    }
    report = {
        "schema_version": "xian_r5.formal_human_fit_holdout_packet_report.v1",
        "artifact_type": "PACKET_PREPARED_WAITING_FOR_REAL_REVIEWER_INPUT",
        "run_id": "B04_20261001_02",
        "generated_at_utc": generated_at,
        "scope": "a13_only_three_actual_date_cells",
        "packet_path": str(PACKET),
        "review_interface": str(PACKET / "review_packet.html"),
        "preparation": {
            "date_cells_prepared": 3,
            "source_ge_tiles_verified_and_copied": 48,
            "actual_date_matches": "48/48",
            "native_reference_pixel_size_m": [10.0, 10.0],
            "candidate_overlay": "DISPLAY_ONLY_FROZEN_VECTOR_NO_PRODUCTION_APPLICATION",
            "native_coordinate_readout": "EPSG:32649 display coordinates; no accuracy claim",
            "editable_tie_table": True,
            "initial_tie_rows": 0,
            "versioned_sidecar_templates": 3,
            "templates_ingestible": False,
            "reviewer_input_received": False,
        },
        "non_mutation": {
            "candidate_vector_modified": False,
            "candidate_geometry_modified": False,
            "support_geometry_modified": False,
            "source_tiles_modified": False,
            "reference_modified": False,
            "registered_roof_application": False,
            "identity_run": False,
            "canonical_geometry_run": False,
            "ground_footprint_run": False,
            "change_run": False,
            "building_tif_product_run": False,
            "citywide_run": False,
            "application_branch_test_run": False,
        },
        "state": state,
        "lineage_parent_run": "B04_20261001_01",
        "lineage_parent_artifact_hashes": {
            path: digest for path, digest in inputs.items() if path.startswith(str(PARENT))
        },
        "input_hashes_before": inputs,
        "input_hashes_after": inputs_after,
        "tile_count_by_date": {
            date: sum(item["actual_date"] == date for item in tile_provenance)
            for date in DATES
        },
        "tie_sidecars": templates,
    }
    write_json(REPORT_PATH, report)
    packet_hashes = {
        path.relative_to(RUN).as_posix(): sha256(path)
        for path in sorted(item for item in PACKET.rglob("*") if item.is_file())
    }
    output_manifest = {
        "schema_version": "xian_r5.b04_packet_output_sha256_manifest.v1",
        "run_id": "B04_20261001_02",
        "files": packet_hashes,
        "report_path": REPORT_PATH.relative_to(RUN).as_posix(),
        "report_sha256": sha256(REPORT_PATH),
        "receipt_excluded": True,
    }
    write_json(OUTPUT_MANIFEST_PATH, output_manifest)
    receipt = {
        "schema_version": "xian_r5.formal_human_fit_holdout_packet_receipt.v1",
        "artifact_type": "APPEND_ONLY_PACKET_PREPARATION_RECEIPT",
        "run_id": "B04_20261001_02",
        "generated_at_utc": generated_at,
        "exact_scope": scope["date_cells"],
        "scope_count_expected": 3,
        "scope_count_resolved": 3,
        "strict_scope_check": "PASS",
        "candidate_input_sha256_before_after": {
            "before": inputs[str(CANDIDATE)],
            "after": inputs_after[str(CANDIDATE)],
            "unchanged": True,
        },
        "reference_input_sha256_before_after": {
            "before": inputs[str(REFERENCE)],
            "after": inputs_after[str(REFERENCE)],
            "unchanged": True,
        },
        "actual_date_table_sha256_before_after": {
            "before": inputs[str(DATE_TABLE)],
            "after": inputs_after[str(DATE_TABLE)],
            "unchanged": True,
        },
        "exact_support_geometry_wkb_sha256": {
            cell["cellId"]: cell["supportWkbSha256"] for cell in cell_configs
        },
        "source_tile_evidence": {
            "verified_count": len(tile_provenance),
            "verified_actual_date_match_count": 48,
            "raw_packet_copy_count": 48,
            "all_source_and_copy_hashes_match": True,
            "provenance_csv": provenance_path.relative_to(RUN).as_posix(),
            "provenance_csv_sha256": sha256(provenance_path),
        },
        "review_state": {
            "real_reviewer_input_received": False,
            "formal_reviews_completed": 0,
            "ties_created": 0,
            "reviewer_id_created": False,
            "reviewed_at_created": False,
            "accepted": 0,
            "rejected": 0,
            "unresolved": 9,
            "ACCEPTED_CELL_PRE_APPLICATION_REGRESSION_GATE": "NOT_RUN",
        },
        "downstream_products": {
            "registered_roofs": 0,
            "physical_ids": 0,
            "canonical_geometries": 0,
            "ground_footprints": 0,
            "change_products": 0,
            "building_tif_products": 0,
            "citywide": False,
            "absolute_position_accuracy": "UNKNOWN",
            "ready_for_citywide": "NO",
        },
        "append_only_report_path": str(REPORT_PATH),
        "report_sha256": sha256(REPORT_PATH),
        "packet_output_manifest_path": str(OUTPUT_MANIFEST_PATH),
        "packet_output_manifest_sha256": sha256(OUTPUT_MANIFEST_PATH),
        "packet_file_count": len(packet_hashes),
        "packet_files_sha256": packet_hashes,
        "non_mutation": {
            "source_candidate_geometry_vector_support_unchanged": True,
            "reference_and_ge_tiles_unchanged": True,
            "no_application_branch_or_existing_tests_run": True,
            "no_registered_roof_identity_canonical_footprint_change_tif_or_citywide_run": True,
        },
    }
    write_json(RECEIPT_PATH, receipt)
    print(json.dumps({
        "run_id": "B04_20261001_02",
        "packet": str(PACKET),
        "report": str(REPORT_PATH),
        "receipt": str(RECEIPT_PATH),
        "scope": scope["date_cells"],
        "raw_tiles": len(tile_provenance),
        "display_extent_epsg32649": [minx, miny, maxx, maxy],
        "display_shape_rows_cols": [height, width],
        "reference_native_crop_shape_rows_cols": reference_meta["native_crop_rows_cols"],
        "packet_file_count": len(packet_hashes),
        "formal_reviews": 0,
        "ties": 0,
        "accepted": 0,
        "rejected": 0,
        "unresolved": 9,
    }, ensure_ascii=False, indent=2))


REVIEW_INSTRUCTIONS = """# R5 a13 FIT/HOLDOUT 人工证据包

## 当前状态

这是供真实 reviewer 使用的准备包，不是 review 结果。仅包含 c0007_a00013 的 2018-04-29、2018-11-01、2019-12-02 三个 date-cell。正式 FIT/HOLDOUT review 仍为 0；没有 tie、reviewer identity、review timestamp 或 disposition。所有 9 个冻结 date-cell 仍为 UNRESOLVED，本包没有准备其他 cell。

打开 review_packet.html（离线本地页面）。三期各自展示 GE 源影像、现有 reference、仅供判断的 candidate-shift overlay、冻结 support、多图同步坐标读数及手工 tie 表。候选位移只能用于画面叠图，任何原始瓦片、candidate vector、candidate geometry、support geometry 都没有被修改。

## Reviewer 操作

1. 选择 actual date；查看上方三面板和对应期未重采样 4×4 原始 GE tile 拼图。原始 JPEG 也按期复制在 assets/ge_raw_tiles/。
2. 仅选择稳定的非建筑地面特征。建筑屋顶、建筑轮廓层、移动阴影/车辆，以及无法论证稳定性的表面不可作为正式 tie。
3. 在 GE 和 reference 面板分别点击同一可识别特征，读取 EPSG:32649 米制坐标；也可以直接编辑表格。点击“使用最近点”前先选定目标行。工具只保存真实 reviewer 点击或手输的坐标，不会自动放置点。
4. 对每一 tie 手工填写 ID、FIT/HOLDOUT、feature type、两侧坐标、两侧 finite nonnegative localization bound、具体 localization basis、独立性组及 evidence annotation/path/hash。
5. 证据不能支持可信界限时，人工填写 precision_eligible=false、TIE_NOT_PRECISION_ELIGIBLE 及解释；不得从 10 m pixel size、zoom、nearest display、模型 confidence 或 candidate residual 反推界限。confidence 标签不能代替距离依据。
6. independence_group 要表示真实独立地物。一个路口、连续路缘/道路边、相邻像素或同一 reference mixed pixel 不可拆为多个独立 tie。FIT/HOLDOUT 两组都至少 3 个 precision-eligible 独立 ties，所有两侧坐标都须位于红色 exact support 内。
7. 表格可以导出 reviewer 输入 CSV；该导出不含 disposition 或 acceptance statistic。不要把本模板直接提交给 adapter：需由正式流程形成完整 v2 sidecar，并由 adapter 从 tie 记录确定性复算残差和统计量。真实 reviewer 的 reviewer_id 与带时区 reviewed_at 必须由 reviewer 自行记录；本页面不生成这些字段，也不生成 ACCEPT/REJECT。

## 分辨率与坐标解释

- Reference 原生像元：10 m；界面显示插值为 nearest。放大显示不增加原生定位证据。原始 reference 文件 acquisition date 与 per-pixel lineage 仍未知。
- GE 原始 z20 tile 按原始 JPEG 像素拼图附带。UTM 面板按已有 R5 tile-grid 锚点做 EPSG:32649 相对可视化；0.25 m 是显示网格，不代表 GE 原生精度或地面准确度。
- candidate-shift overlay 把 GE 显示内容按冻结 (dx,dy) 在图层上平移并混合显示；这是辅助可视化，不是配准接受、地面足迹恢复或 production application。
- 所有绝对地面位置精度继续为 UNKNOWN。相对 image-grid review 不等于 ground control、roof-to-ground correction、physical identity 或 confirmed change。
"""


LEDGER_TEMPLATE = """# Source and coordinate ledger

## Scope

- Run: B04_20261001_02
- Frozen project: XIAN-R5-GROUNDREG-FREEZE-0929
- Allowlist for this packet only: c0007_a00013 × 2018-04-29; c0007_a00013 × 2018-11-01; c0007_a00013 × 2019-12-02.
- Resolved cells: 3 exactly. No fallback, no date expansion, no other cell processing.

## GE source imagery

- Source: 48 original Google Earth z20 JPEG tiles, 16 per actual date; source bytes are copied unchanged under assets/ge_raw_tiles/.
- Actual-date audit: {date_table} SHA-256 {date_table_sha}. 48/48 tile records have a recovered actual capture date, date_conflict=0, and match the requested date. Per-tile source path, source/copy SHA-256, source log, and line number are in tile_actual_date_and_hash_provenance.csv.
- Source tile grid transform: anchor tile ({origin_x},{origin_y}) at lon/lat ({origin_lon:.14f},{origin_lat:.14f}); tile step {step:.16f} degrees; source pixels are 256×256. This is the existing local R5 tile-grid mapping documented in {tile_grid_source} (SHA-256 {tile_grid_sha}). It is a relative display georeference, not survey control or an absolute accuracy statement.
- Native tile content is also presented as an original-pixel 4×4 mosaic (1024×1024, no resampling). The common UTM panel uses nearest sampling at {resolution} m display grid solely for aligned viewing.

## Relative reference

- Source: {reference}
- SHA-256: {reference_sha}
- Native CRS: EPSG:32649; pixel size 10 m × 10 m; raster size 3885×3086; RGB bands 1/2/3.
- The file carries an annual label only. Exact acquisition date and per-pixel source-component lineage are unknown. The reference is used as a relative image grid only.
- Enlarged display resampling: nearest. Native pixel evidence remains 10 m regardless of displayed scale.

## Candidate and support geometry

- Candidate source: {candidate}, layer spatial_alignment_candidates, SHA-256 {candidate_sha}.
- Candidate status is SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY; reviewed=0, ground_evidence=0, applied=0.
- The exact support geometry is read from the selected a13 feature in EPSG:32649. Per-cell WKB SHA-256: {support_sha}; area: 18702.86331515698 m². Candidate vectors are frozen and shown with full precision in the page and sidecars.
- Red outlines reproduce the exact support geometry. Overlay image only shifts GE pixels by the frozen vector for reviewer visual comparison; no source or product geometry is changed.

## Coordinate/provenance caveats

- Native coordinate CRS shown by the interface: EPSG:32649, metres. The 0.25 m panel pixel and browser cursor readout are display coordinates, not localization bounds or accuracy claims.
- Do not infer any bound from the 10 m reference pixel size, zoom, display interpolation, confidence, or candidate residual.
- ABSOLUTE_POSITION_ACCURACY = UNKNOWN.
"""


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R5 a13 FIT/HOLDOUT human evidence packet</title>
<style>
:root{--ink:#1d2a3a;--muted:#59697d;--line:#d7e0ea;--bg:#f3f6fa;--card:#fff;--red:#d42332}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}header{padding:20px 24px;background:#12263a;color:white}header h1{font-size:22px;margin:0 0 7px}header p{margin:0;color:#dce7f1}.wrap{max-width:1600px;margin:18px auto;padding:0 16px}.banner{padding:14px 16px;border-left:5px solid #d08b00;background:#fff7df;margin-bottom:14px}.status{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}.pill{border:1px solid var(--line);border-radius:999px;background:#fff;padding:6px 10px;font-size:13px}.tabs{display:flex;gap:8px;margin:14px 0}.tabs button,button{border:1px solid #becbd9;background:white;color:var(--ink);border-radius:8px;padding:8px 12px;cursor:pointer;font:inherit}.tabs button.active{background:#14385c;color:white;border-color:#14385c}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 4px #0e2b4610}.cell-head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center}.cell-head h2{margin:0;font-size:19px}.sub{color:var(--muted);font-size:13px}.maps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:14px}.map-card{border:1px solid var(--line);border-radius:10px;padding:9px;background:#fbfcfe;min-width:0}.map-card h3{font-size:14px;margin:0 0 7px}.map-wrap{position:relative;width:100%;line-height:0;cursor:crosshair;background:#dbe2e9;overflow:hidden}.map-wrap img{display:block;width:100%;height:auto;user-select:none}.map-wrap svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.support{fill:none;stroke:var(--red);stroke-width:2.4;vector-effect:non-scaling-stroke}.point{fill:#00e3ff;stroke:#002e44;stroke-width:2;vector-effect:non-scaling-stroke}.info{font-size:12px;color:var(--muted);min-height:34px;margin-top:7px}.readout{font-variant-numeric:tabular-nums;color:#12385d}.raw{max-width:100%;height:auto;image-rendering:auto;border:1px solid var(--line)}.warning{background:#fff2ed;border-left:4px solid #c74828;padding:10px 12px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;min-width:1900px;width:100%;font-size:12px}th,td{border-bottom:1px solid #e6ebf0;padding:5px;vertical-align:top}th{position:sticky;top:0;background:#eef3f8;text-align:left;z-index:1}input,select,textarea{font:inherit;font-size:12px;border:1px solid #bac6d3;border-radius:5px;padding:5px;width:130px}textarea{width:180px;min-height:38px}input.coord{width:110px}input.bound{width:105px}td.statuscell{min-width:190px}.actions{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.small{font-size:12px;color:var(--muted)}details{margin:10px 0}code,.mono{font-family:Consolas,monospace;font-variant-numeric:tabular-nums}footer{padding:18px;color:var(--muted);font-size:12px}@media(max-width:980px){.maps{grid-template-columns:1fr}.map-card{max-width:780px}.wrap{padding:0 10px}}
</style></head><body>
<header><h1>R5 相对配准 FIT / HOLDOUT 人工证据包</h1><p>仅限 c0007_a00013 · 三个真实观测日期 · 候选向量冻结 · 当前尚未开始正式人审</p></header>
<div class="wrap">
<div class="banner"><strong>证据分辨率与状态边界</strong><br>参考影像原生像元为 10 m。页面放大采用 nearest display sampling，不会增加原生证据分辨率。候选平移仅为审核辅助叠图，不是已接受配准或生产应用。ABSOLUTE_POSITION_ACCURACY = UNKNOWN。</div>
<div class="status"><span class="pill">FIT/HOLDOUT 正式审核：0</span><span class="pill">ACCEPT：0</span><span class="pill">REJECT：0</span><span class="pill">UNRESOLVED：9（沿用冻结基线）</span><span class="pill">本包 tie rows：0</span></div>
<div class="card"><div class="cell-head"><h2>选择审核日期</h2><span class="sub">日期来自逐瓦片实际日期证据，不按名义年份推断</span></div><div class="tabs" id="tabs"></div><div id="cellView"></div></div>
<div class="card"><h2>人工 tie 输入表</h2><p class="warning"><strong>只由真实 reviewer 手工添加。</strong>本包没有自动创建 tie、坐标、定位界限、FIT/HOLDOUT 角色、reviewer_id、reviewed_at 或 disposition。每条 eligible tie 必须有可追溯 evidence artifact；localization_basis 要解释特征为何能在所填米制距离内定位。</p>
<p class="small">道路/路口或硬化面不能仅因“看起来对应”就计为 precision tie。不得从 10 m 像元、zoom、nearest display、模型信心或 candidate residual 推导 localization bounds。confidence 标签不能代替距离依据。相邻采样或同一连续地物不可拆成独立 tie。</p>
<div class="actions"><button id="addRow">添加空白 tie 行</button><button id="exportCsv">导出当前人工输入 CSV（不含 disposition / acceptance statistic）</button><span class="small">导出只包含 reviewer 已输入字段；adapter 后续从完整 sidecar 复算残差与不确定性。</span></div>
<div class="table-wrap"><table id="ties"><thead><tr><th>删除</th><th>tie_id（人工）</th><th>FIT/HOLDOUT（人工）</th><th>feature_type</th><th>ge_x / ge_y</th><th>reference_x / reference_y</th><th>GE localization bound (m)</th><th>Reference localization bound (m)</th><th>localization_basis</th><th>independence_group</th><th>evidence_annotation</th><th>evidence_path</th><th>evidence_sha256</th><th>precision_eligible</th><th>ineligibility reason</th><th>stable non-building feature class</th><th>stable evidence</th><th>stability_basis</th><th>exact support check</th></tr></thead><tbody></tbody></table></div>
<div class="small">坐标 CRS：EPSG:32649，单位 m。先在 GE 或 reference 面板点击，再对目标 tie 行显式按“使用最近点”；也可以直接手工编辑。exact-support 状态由冻结 support 几何确定性检查，adapter 仍会独立复算。</div></div>
<div class="card"><h2>原始 GE 影像来源</h2><p>每期附 16 张原始 z20 JPEG 和按原始像素拼接的 4×4 无重采样图。下图保留源像素。</p><div id="rawBlock"></div></div>
<div class="card"><h2>Reference 来源</h2><p>Sentinel-derived RGB，EPSG:32649，原生 10 m。精确 acquisition date 与 per-pixel source lineage 未知。它用于相对 image-grid review，不是绝对地面控制。</p><p class="small">原始参考文件 SHA-256：<span class="mono" id="refHash"></span>。原生像素小窗和 nearest 显示随包提供。</p></div>
<footer>本地离线页面；不联网、不运行模型、不应用 candidate 到生产数据。没有真实 reviewer 输入，因此正式审核仍为 0；未运行 ACCEPTED_CELL_PRE_APPLICATION_REGRESSION_GATE。</footer>
</div>
<script>
const CONFIG = __CONFIG__;
const fields=['tie_id','FIT_or_HOLDOUT','feature_type','ge_x','ge_y','reference_x','reference_y','ge_localization_bound_m','reference_localization_bound_m','localization_basis','independence_group','evidence_annotation','evidence_path','evidence_sha256','precision_eligible','precision_ineligibility_reason','feature_class','stable_nonbuilding_ground_evidence','stability_basis'];
const states={};let activeDate=CONFIG.dateCells[0].date,lastGe=null,lastReference=null;
function stateFor(d){if(!states[d])states[d]=[];return states[d]}
function esc(s){return String(s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})}
function cell(){return CONFIG.dateCells.find(function(c){return c.date===activeDate})}
function pathFor(c){return c.supportCoordinates.map(function(p,i){const x=(p[0]-CONFIG.displayGrid.minX)/CONFIG.displayGrid.resolutionM,y=(CONFIG.displayGrid.maxY-p[1])/CONFIG.displayGrid.resolutionM;return(i?'L':'M')+x.toFixed(3)+' '+y.toFixed(3)}).join(' ')+' Z'}
function inside(x,y,pts){let yes=false;for(let i=0,j=pts.length-1;i<pts.length;j=i++){const a=pts[i],b=pts[j],cross=(x-a[0])*(b[1]-a[1])-(y-a[1])*(b[0]-a[0]);if(Math.abs(cross)<1e-7&&x>=Math.min(a[0],b[0])-1e-7&&x<=Math.max(a[0],b[0])+1e-7&&y>=Math.min(a[1],b[1])-1e-7&&y<=Math.max(a[1],b[1])+1e-7)return true;const hit=((a[1]>y)!==(b[1]>y))&&(x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]);if(hit)yes=!yes}return yes}
function tabs(){const e=document.getElementById('tabs');e.innerHTML='';CONFIG.dateCells.forEach(function(c){const b=document.createElement('button');b.textContent=c.date;b.className=c.date===activeDate?'active':'';b.onclick=function(){activeDate=c.date;lastGe=null;lastReference=null;tabs();renderCell()};e.appendChild(b)})}
function mapCard(layer,title,src,note){return '<div class="map-card"><h3>'+title+'</h3><div class="map-wrap" data-layer="'+layer+'"><img src="'+src+'" draggable="false" alt="'+title+'"><svg viewBox="0 0 '+CONFIG.displayGrid.width+' '+CONFIG.displayGrid.height+'" preserveAspectRatio="none"><path class="support" d="'+pathFor(cell())+'"></path><circle class="point" data-point="'+layer+'" cx="-20" cy="-20" r="5"></circle></svg></div><div class="info">'+note+'</div></div>'}
function renderCell(){const c=cell();document.getElementById('cellView').innerHTML='<div class="cell-head"><h2>'+c.date+' · '+c.cellId+'</h2><div><span class="pill">Frozen dx = '+c.candidateDxM.toFixed(6)+' m</span> <span class="pill">Frozen dy = '+c.candidateDyM.toFixed(6)+' m</span> <span class="pill">support = '+c.supportAreaM2.toFixed(2)+' m²</span></div></div><p class="sub">Candidate status: '+esc(c.candidateStatus)+' · 仅在视觉层平移 GE 内容，candidate 几何和向量不变。</p><div class="maps">'+mapCard('ge','GE 高分影像 · EPSG:32649 display',c.images.ge,'点击读取 GE 点坐标。')+mapCard('reference','Existing reference · native 10 m',c.images.reference,'nearest display；点击显示 native reference 行/列和像元中心。')+mapCard('overlay','Candidate-applied overlay · 仅辅助',c.images.overlay,'冻结向量平移后的 GE 与 reference 混合叠图；不是已注册产品。')+'</div><div id="liveReadout" class="info readout">鼠标移到影像查看 EPSG:32649 坐标。显示网格不代表影像定位精度。</div><details><summary>原生 reference 像元小窗（未经放大，1 pixel = 10 m）</summary><p><img class="raw" src="'+CONFIG.reference.nativePixelsPng+'" alt="Native 10 m reference pixels"><br><span class="small">页面缩放不改变原生 10 m 证据分辨率。</span></p></details>';document.querySelectorAll('.map-wrap').forEach(function(w){w.addEventListener('mousemove',move);w.addEventListener('click',clickMap)});renderTable();renderRaw()}
function at(ev,w){const r=w.getBoundingClientRect(),px=Math.max(0,Math.min(CONFIG.displayGrid.width,(ev.clientX-r.left)/r.width*CONFIG.displayGrid.width)),py=Math.max(0,Math.min(CONFIG.displayGrid.height,(ev.clientY-r.top)/r.height*CONFIG.displayGrid.height));return{x:CONFIG.displayGrid.minX+px*CONFIG.displayGrid.resolutionM,y:CONFIG.displayGrid.maxY-py*CONFIG.displayGrid.resolutionM,px:px,py:py}}
function refPixel(x,y){const t=CONFIG.reference.nativeTransform,col=Math.floor((x-t[2])/t[0]),row=Math.floor((t[5]-y)/(-t[4]));if(col<0||row<0||col>=CONFIG.reference.nativeWidth||row>=CONFIG.reference.nativeHeight)return null;return{col:col,row:row,cx:t[2]+(col+.5)*t[0],cy:t[5]+(row+.5)*t[4]}}
function move(ev){const w=ev.currentTarget,p=at(ev,w),layer=w.dataset.layer;let extra='';if(layer==='reference'){const q=refPixel(p.x,p.y);extra=q?' · native pixel col='+q.col+', row='+q.row+'; center=('+q.cx.toFixed(2)+', '+q.cy.toFixed(2)+')':' · outside reference raster'}document.getElementById('liveReadout').textContent=layer.toUpperCase()+' cursor: X='+p.x.toFixed(3)+' m, Y='+p.y.toFixed(3)+' m (EPSG:32649)'+extra+' · display grid 0.25 m; no accuracy claim.'}
function clickMap(ev){const w=ev.currentTarget,p=at(ev,w),layer=w.dataset.layer;if(layer==='ge')lastGe={x:p.x,y:p.y};else if(layer==='reference')lastReference={x:p.x,y:p.y};const mark=w.querySelector('[data-point="'+layer+'"]');mark.setAttribute('cx',p.px);mark.setAttribute('cy',p.py);move(ev)}
function input(field,value){const v=value||'';if(field==='FIT_or_HOLDOUT')return '<select data-field="'+field+'"><option value="">Select role…</option><option>FIT</option><option>HOLDOUT</option></select>';if(field==='precision_eligible'||field==='stable_nonbuilding_ground_evidence')return '<select data-field="'+field+'"><option value="">Select…</option><option value="true">true</option><option value="false">false</option></select>';if(field==='feature_class')return '<select data-field="'+field+'"><option value="">Select feature class…</option><option>STABLE_NONBUILDING_GROUND_FEATURE</option></select>';if(['ge_x','ge_y','reference_x','reference_y','ge_localization_bound_m','reference_localization_bound_m'].includes(field))return '<input class="'+(field.includes('_bound_')?'bound':'coord')+'" type="number" step="0.001" data-field="'+field+'" value="'+esc(v)+'">';if(['localization_basis','evidence_annotation','stability_basis'].includes(field))return '<textarea data-field="'+field+'">'+esc(v)+'</textarea>';return '<input type="text" data-field="'+field+'" value="'+esc(v)+'">'}
function supportState(r){if(r.ge_x===''||r.ge_y===''||r.reference_x===''||r.reference_y==='')return '';return inside(Number(r.ge_x),Number(r.ge_y),cell().supportCoordinates)&&inside(Number(r.reference_x),Number(r.reference_y),cell().supportCoordinates)?'true':'false'}
function updateSupportText(tr,r){const q=supportState(r);const el=tr.querySelector('.statuscell');if(el)el.textContent=q===''?'待填写两侧坐标':q==='true'?'两侧点均在 exact support 内':'至少一侧点在 exact support 外'}
function renderTable(){const b=document.querySelector('#ties tbody');b.innerHTML='';stateFor(activeDate).forEach(function(r,i){const tr=document.createElement('tr');let s='<td><button data-del="'+i+'">×</button></td>';fields.forEach(function(f){let ctl=input(f,r[f]);if(f==='ge_x'||f==='ge_y')ctl+=' <button data-use="ge" data-i="'+i+'">用最近 GE 点</button>';if(f==='reference_x'||f==='reference_y')ctl+=' <button data-use="reference" data-i="'+i+'">用最近 Reference 点</button>';s+='<td>'+ctl+'</td>'});const q=supportState(r);s+='<td class="statuscell">'+(q===''?'待填写两侧坐标':q==='true'?'两侧点均在 exact support 内':'至少一侧点在 exact support 外')+'</td>';tr.innerHTML=s;b.appendChild(tr);tr.querySelectorAll('[data-field]').forEach(function(el){el.addEventListener('input',function(){r[el.dataset.field]=el.value;updateSupportText(tr,r)});el.addEventListener('change',function(){r[el.dataset.field]=el.value;updateSupportText(tr,r)})});tr.querySelector('[data-del]').onclick=function(){stateFor(activeDate).splice(i,1);renderTable()};tr.querySelectorAll('[data-use]').forEach(function(btn){btn.onclick=function(){const p=btn.dataset.use==='ge'?lastGe:lastReference;if(!p){alert('先在对应影像面板手工点击一个特征位置。');return}r[btn.dataset.use+'_x']=p.x.toFixed(3);r[btn.dataset.use+'_y']=p.y.toFixed(3);renderTable()}})})}
function renderRaw(){const c=cell();document.getElementById('rawBlock').innerHTML='<p><strong>'+c.date+' · '+c.sceneId+'</strong><br><img class="raw" src="'+c.images.rawMosaic+'" alt="Original unresampled 4 by 4 GE tile mosaic"><br><span class="small">原始像素拼接 1024×1024，未重采样。单瓦片文件及哈希见 tile_actual_date_and_hash_provenance.csv。</span></p><p><a href="'+c.images.contactSheet+'" target="_blank">静态三面板 contact sheet</a> · <a href="'+c.images.rawMosaic+'" target="_blank">原始 GE 拼图</a></p>'}
function exportCsv(){const cols=fields.concat(['within_exact_support']),rows=stateFor(activeDate),q=function(v){return '"'+String(v==null?'':v).replace(/"/g,'""')+'"'};let lines=[cols.map(q).join(',')];rows.forEach(function(r){lines.push(cols.map(function(k){return q(k==='within_exact_support'?supportState(r):(r[k]||''))}).join(','))});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'}));a.download='reviewer-entered-ties_'+activeDate+'.csv';a.click();setTimeout(function(){URL.revokeObjectURL(a.href)},1000)}
document.getElementById('addRow').onclick=function(){const r={};fields.forEach(function(f){r[f]=''});stateFor(activeDate).push(r);renderTable()};document.getElementById('exportCsv').onclick=exportCsv;document.getElementById('refHash').textContent=CONFIG.reference.sha256;tabs();renderCell();
</script></body></html>"""


if __name__ == "__main__":
    main()
