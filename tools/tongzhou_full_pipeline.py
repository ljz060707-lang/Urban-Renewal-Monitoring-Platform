
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml

from matplotlib.patches import Patch
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import shape


# ============================================================
# CONFIG
# ============================================================

PROJECT = Path(
    r"C:\Users\ljz52\Documents\Codex\2026-07-18\urban-renewal-phase1-starter-v0-2\outputs\urban-renewal-phase1-starter"
)

PYTHON = PROJECT / ".venv-gpu" / "Scripts" / "python.exe"

SAT_ROOT = Path(
    r"D:\Data Set\Satellite\beijing_tongzhou"
)

INFER_CONFIG = (
    PROJECT
    / "configs"
    / "change_public_v2_full20_inference_windows.yaml"
)

OUTPUT_ROOT = (
    PROJECT
    / "outputs"
    / "inference"
    / "beijing_tongzhou_to_latest"
)

SUMMARY_ROOT = OUTPUT_ROOT / "SUMMARY"

EXPECTED_FULL20_SHA = (
    "70040ca0e688537fd63479589690d5be5"
    "eefef859f39763c6b45a09852bc2621"
)

# Beijing: WGS84 / UTM zone 50N
METRIC_CRS = "EPSG:32650"

MIN_MAPPING_UNIT_M2 = 10.0

MAX_DISPLAY_WIDTH = 2200


# ============================================================
# BASIC UTILITIES
# ============================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(16 * 1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def parse_date(path: Path) -> datetime:
    """
    Accept examples such as:

      200803.tif       -> 2020-08-03
      210414_xxx.tif   -> 2021-04-14
      20230713.tif     -> 2023-07-13

    First tries an 8-digit YYYYMMDD token,
    then a 6-digit YYMMDD token.
    """

    stem = path.stem

    eight = re.findall(
        r"(?<!\d)(20\d{6})(?!\d)",
        stem,
    )

    for token in eight:
        try:
            return datetime.strptime(
                token,
                "%Y%m%d",
            )
        except ValueError:
            pass

    six = re.findall(
        r"(?<!\d)(\d{6})(?!\d)",
        stem,
    )

    for token in six:
        try:
            return datetime.strptime(
                token,
                "%y%m%d",
            )
        except ValueError:
            pass

    raise ValueError(
        "Cannot infer acquisition date from filename: "
        f"{path.name}"
    )


def safe_tag(tags: dict, *keys: str) -> str:
    for key in keys:
        value = tags.get(key)

        if value not in (None, ""):
            return str(value)

    return ""


def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CHECK FULL20 CONFIG
# ============================================================

def verify_full20() -> dict:
    if not INFER_CONFIG.is_file():
        raise FileNotFoundError(
            f"Inference config missing: {INFER_CONFIG}"
        )

    cfg = yaml.safe_load(
        INFER_CONFIG.read_text(
            encoding="utf-8-sig"
        )
    )

    checkpoint_value = (
        cfg
        .get("inference", {})
        .get("checkpoint")
    )

    if not checkpoint_value:
        raise RuntimeError(
            "Config has no inference.checkpoint."
        )

    checkpoint = Path(checkpoint_value)

    if not checkpoint.is_absolute():
        checkpoint = (
            PROJECT / checkpoint
        ).resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Full20 checkpoint missing: {checkpoint}"
        )

    digest = sha256(checkpoint)

    if digest != EXPECTED_FULL20_SHA:
        raise RuntimeError(
            "Full20 SHA mismatch.\n"
            f"Observed: {digest}\n"
            f"Expected: {EXPECTED_FULL20_SHA}"
        )

    print("=" * 100)
    print("FULL20 CHECKPOINT")
    print("=" * 100)
    print("checkpoint:", checkpoint)
    print("sha256    :", digest)
    print("status    : VERIFIED")
    print()

    return cfg


# ============================================================
# TIFF AUDIT
# ============================================================

def audit_tiffs() -> tuple[list[dict], list[dict]]:
    files = sorted([
        p
        for p in SAT_ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {
            ".tif",
            ".tiff",
        }
    ])

    if len(files) < 2:
        raise RuntimeError(
            f"Need at least two TIFFs under {SAT_ROOT}"
        )

    records: list[dict] = []

    print("=" * 100)
    print("TONGZHOU SATELLITE AUDIT")
    print("=" * 100)

    for path in files:
        date = parse_date(path)
        digest = sha256(path)

        with rasterio.open(path) as ds:
            tags = ds.tags()

            record = {
                "file":
                    path.name,

                "path":
                    str(path.resolve()),

                "date":
                    date.strftime("%Y-%m-%d"),

                "_date":
                    date,

                "sha256":
                    digest,

                "width":
                    ds.width,

                "height":
                    ds.height,

                "bands":
                    ds.count,

                "dtype":
                    ";".join(ds.dtypes),

                "crs":
                    str(ds.crs),

                "res_x":
                    ds.res[0],

                "res_y":
                    ds.res[1],

                "left":
                    ds.bounds.left,

                "bottom":
                    ds.bounds.bottom,

                "right":
                    ds.bounds.right,

                "top":
                    ds.bounds.top,

                "transform":
                    tuple(ds.transform),

                "metadata_datetime":
                    safe_tag(
                        tags,
                        "TIFFTAG_DATETIME",
                        "DATETIME",
                        "ACQUISITION_DATE",
                        "ACQUISITIONDATETIME",
                    ),
            }

        records.append(record)

        print()
        print(path.name)
        print(" date :", record["date"])
        print(" hash :", digest)
        print(
            " size :",
            record["width"],
            "x",
            record["height"],
        )
        print(" bands:", record["bands"])
        print(" crs  :", record["crs"])
        print(
            " res  :",
            record["res_x"],
            record["res_y"],
        )
        print(
            " metadata date:",
            record["metadata_datetime"]
            or "(not embedded)",
        )

    # --------------------------------------------------------
    # Save complete TIFF audit
    # --------------------------------------------------------

    audit_path = (
        OUTPUT_ROOT
        / "TONGZHOU_TIF_AUDIT.csv"
    )

    export_fields = [
        key
        for key in records[0]
        if not key.startswith("_")
        and key != "transform"
    ]

    with audit_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=export_fields,
        )

        writer.writeheader()

        for row in records:
            writer.writerow({
                key: row[key]
                for key in export_fields
            })

    # --------------------------------------------------------
    # Exact duplicate audit
    # --------------------------------------------------------

    by_hash: dict[str, list[dict]] = defaultdict(list)

    for row in records:
        by_hash[row["sha256"]].append(row)

    duplicate_rows = []

    canonical: list[dict] = []

    for digest, group in by_hash.items():
        group = sorted(
            group,
            key=lambda x: x["_date"],
        )

        dates = sorted({
            x["date"]
            for x in group
        })

        date_conflict = (
            len(dates) > 1
        )

        duplicate_rows.append({
            "sha256":
                digest,

            "files":
                ";".join(
                    x["file"]
                    for x in group
                ),

            "dates":
                ";".join(dates),

            "count":
                len(group),

            "duplicate":
                len(group) > 1,

            "date_conflict":
                date_conflict,

            "action":
                (
                    "EXCLUDED_DATE_CONFLICT"
                    if date_conflict
                    else "KEEP_ONE"
                ),
        })

        # Important:
        # identical bytes labelled as different dates cannot
        # be treated as valid temporal observations.
        if date_conflict:
            print()
            print(
                "[WARNING] exact duplicate content "
                "has conflicting dates:"
            )

            for item in group:
                print(
                    " ",
                    item["file"],
                    item["date"],
                )

            print(
                " This content group is excluded "
                "from temporal comparisons."
            )

            continue

        representative = dict(group[0])

        representative["aliases"] = (
            ";".join(
                x["file"]
                for x in group
            )
        )

        canonical.append(
            representative
        )

    dup_path = (
        OUTPUT_ROOT
        / "DUPLICATE_AUDIT.csv"
    )

    with dup_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sha256",
                "files",
                "dates",
                "count",
                "duplicate",
                "date_conflict",
                "action",
            ],
        )

        writer.writeheader()
        writer.writerows(
            duplicate_rows
        )

    if len(canonical) < 2:
        raise RuntimeError(
            "After removing duplicate-date conflicts, "
            "fewer than two unique valid observations remain."
        )

    canonical.sort(
        key=lambda x: x["_date"]
    )

    print()
    print("=" * 100)
    print("VALID UNIQUE OBSERVATIONS")
    print("=" * 100)

    for row in canonical:
        print(
            row["date"],
            row["file"],
            "aliases=",
            row.get("aliases", row["file"]),
        )

    return records, canonical


# ============================================================
# GRID CHECK + PAIR PLAN
# ============================================================

def same_grid(
    a: dict,
    b: dict,
) -> bool:

    if a["crs"] != b["crs"]:
        return False

    if a["width"] != b["width"]:
        return False

    if a["height"] != b["height"]:
        return False

    ta = np.asarray(
        a["transform"],
        dtype=float,
    )

    tb = np.asarray(
        b["transform"],
        dtype=float,
    )

    return bool(
        np.allclose(
            ta,
            tb,
            rtol=0,
            atol=1e-12,
        )
    )


def build_pair_plan(
    canonical: list[dict],
) -> list[dict]:

    latest = max(
        canonical,
        key=lambda x: x["_date"],
    )

    print()
    print("=" * 100)
    print("LATEST VALID UNIQUE IMAGE")
    print("=" * 100)

    print(
        latest["date"],
        latest["file"],
    )

    pairs: list[dict] = []

    for earlier in canonical:
        if (
            earlier["sha256"]
            == latest["sha256"]
        ):
            continue

        grid_ok = same_grid(
            earlier,
            latest,
        )

        pair = {
            "pair_name":
                f"{earlier['_date'].strftime('%Y%m%d')}"
                f"_to_"
                f"{latest['_date'].strftime('%Y%m%d')}",

            "t1_file":
                earlier["file"],

            "t1_path":
                earlier["path"],

            "t1_date":
                earlier["date"],

            "t1_sha256":
                earlier["sha256"],

            "t1_aliases":
                earlier.get(
                    "aliases",
                    earlier["file"],
                ),

            "t2_file":
                latest["file"],

            "t2_path":
                latest["path"],

            "t2_date":
                latest["date"],

            "t2_sha256":
                latest["sha256"],

            "grid_match":
                grid_ok,

            "eligible":
                grid_ok,
        }

        pairs.append(pair)

    plan_path = (
        OUTPUT_ROOT
        / "PAIR_PLAN.csv"
    )

    fields = list(
        pairs[0].keys()
    )

    with plan_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(pairs)

    print()
    print("=" * 100)
    print("PAIR PLAN")
    print("=" * 100)

    for pair in pairs:
        state = (
            "PASS"
            if pair["eligible"]
            else "SKIP_GRID_MISMATCH"
        )

        print(
            state,
            pair["t1_date"],
            "->",
            pair["t2_date"],
            pair["pair_name"],
        )

    eligible = [
        p
        for p in pairs
        if p["eligible"]
    ]

    if not eligible:
        raise RuntimeError(
            "No historical image shares the exact "
            "CRS/extent/grid with the latest image."
        )

    return eligible


# ============================================================
# FULL20 INFERENCE
# ============================================================

def inference_complete(
    output: Path,
) -> bool:

    required = [
        output
        / "rasters"
        / "any_change_probability.tif",

        output
        / "rasters"
        / "any_change_mask.tif",

        output
        / "rasters"
        / "new_building_candidate.tif",

        output
        / "rasters"
        / "new_building_mask.tif",

        output
        / "rasters"
        / "demolition_candidate.tif",

        output
        / "rasters"
        / "demolition_mask.tif",

        output
        / "preview.png",
    ]

    return all(
        p.is_file()
        and p.stat().st_size > 0
        for p in required
    )


def run_inference(
    pairs: list[dict],
) -> None:

    print()
    print("=" * 100)
    print("FULL20 BATCH INFERENCE")
    print("=" * 100)

    for index, pair in enumerate(
        pairs,
        start=1,
    ):

        output = (
            OUTPUT_ROOT
            / pair["pair_name"]
        )

        print()
        print("=" * 100)
        print(
            f"[{index}/{len(pairs)}] "
            f"{pair['pair_name']}"
        )
        print("=" * 100)

        if inference_complete(output):
            print(
                "Existing complete inference found."
            )
            print("SKIP GPU inference.")
            continue

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        cmd = [
            str(PYTHON),

            str(
                PROJECT / "run.py"
            ),

            "urban-renewal-infer",

            "--config",
            str(INFER_CONFIG),

            "--t1",
            pair["t1_path"],

            "--t2",
            pair["t2_path"],

            "--output",
            str(output),

            "--device",
            "cuda:0",
        ]

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT),
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Full20 inference failed: "
                f"{pair['pair_name']}"
            )

        if not inference_complete(output):
            raise RuntimeError(
                "Inference returned success but "
                "required outputs are missing: "
                f"{pair['pair_name']}"
            )

        print("INFERENCE: PASS")


# ============================================================
# METRIC VECTORISATION
# ============================================================

def metric_vectorize(
    mask_path: Path,
    output_path: Path,
    layer: str,
) -> tuple[gpd.GeoDataFrame, int]:

    with rasterio.open(
        mask_path
    ) as ds:

        mask = ds.read(1)

        source_crs = ds.crs

        geometries = []

        for geom, value in shapes(
            mask.astype(np.uint8),
            mask=(mask > 0),
            transform=ds.transform,
        ):

            if int(value) != 1:
                continue

            geometries.append(
                shape(geom)
            )

    raw_count = len(
        geometries
    )

    if not geometries:
        empty = gpd.GeoDataFrame(
            {
                "class": [],
                "area_m2": [],
            },
            geometry=[],
            crs=METRIC_CRS,
        )

        return empty, 0

    gdf = gpd.GeoDataFrame(
        {
            "class":
                [layer]
                * len(geometries)
        },
        geometry=geometries,
        crs=source_crs,
    )

    metric = gdf.to_crs(
        METRIC_CRS
    )

    metric[
        "area_m2"
    ] = metric.geometry.area

    metric = (
        metric[
            metric["area_m2"]
            >= MIN_MAPPING_UNIT_M2
        ]
        .copy()
        .reset_index(drop=True)
    )

    metric[
        "object_id"
    ] = [
        f"{layer}_{i+1:05d}"
        for i in range(
            len(metric)
        )
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not metric.empty:
        metric.to_file(
            output_path,
            layer=layer,
            driver="GPKG",
        )

    return metric, raw_count


def vectorize_all(
    pairs: list[dict],
) -> dict[str, dict]:

    results = {}

    print()
    print("=" * 100)
    print("METRIC OBJECT REFINEMENT")
    print("=" * 100)

    for pair in pairs:

        pair_root = (
            OUTPUT_ROOT
            / pair["pair_name"]
        )

        rasters = (
            pair_root
            / "rasters"
        )

        metric_root = (
            pair_root
            / "vectors_metric"
        )

        new_gdf, new_raw = metric_vectorize(
            rasters
            / "new_building_mask.tif",

            metric_root
            / "new_building_metric.gpkg",

            "new_building",
        )

        demo_gdf, demo_raw = metric_vectorize(
            rasters
            / "demolition_mask.tif",

            metric_root
            / "demolition_metric.gpkg",

            "demolition",
        )

        results[
            pair["pair_name"]
        ] = {
            "new":
                new_gdf,

            "demo":
                demo_gdf,

            "new_raw":
                new_raw,

            "demo_raw":
                demo_raw,
        }

        print()
        print(pair["pair_name"])
        print(
            " new:",
            new_raw,
            "raw ->",
            len(new_gdf),
            ">= 10 m2",
        )

        print(
            " demo:",
            demo_raw,
            "raw ->",
            len(demo_gdf),
            ">= 10 m2",
        )

    return results


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def read_rgb(
    path: Path,
    max_width: int = MAX_DISPLAY_WIDTH,
):

    with rasterio.open(path) as ds:

        scale = min(
            1.0,
            max_width / ds.width,
        )

        width = max(
            1,
            int(ds.width * scale),
        )

        height = max(
            1,
            int(ds.height * scale),
        )

        band_count = min(
            3,
            ds.count,
        )

        array = ds.read(
            list(
                range(
                    1,
                    band_count + 1,
                )
            ),
            out_shape=(
                band_count,
                height,
                width,
            ),
            resampling=(
                Resampling.bilinear
            ),
        )

        if band_count == 1:
            array = np.repeat(
                array,
                3,
                axis=0,
            )

        array = np.moveaxis(
            array,
            0,
            -1,
        ).astype(
            np.float32
        )

        if array.max() > 1:
            if array.max() <= 255:
                array /= 255.0
            else:
                low = np.nanpercentile(
                    array,
                    2,
                )

                high = np.nanpercentile(
                    array,
                    98,
                )

                array = np.clip(
                    (
                        array
                        - low
                    )
                    / max(
                        high - low,
                        1e-6,
                    ),
                    0,
                    1,
                )

        extent = [
            ds.bounds.left,
            ds.bounds.right,
            ds.bounds.bottom,
            ds.bounds.top,
        ]

        return (
            array,
            extent,
            ds.crs,
        )


def read_display_raster(
    path: Path,
    shape_: tuple[int, int],
    nearest=False,
):

    with rasterio.open(path) as ds:

        method = (
            Resampling.nearest
            if nearest
            else Resampling.bilinear
        )

        return ds.read(
            1,
            out_shape=shape_,
            resampling=method,
        )


def decorate_map(
    ax,
    extent,
):

    left, right, bottom, top = extent

    ax.annotate(
        "N",
        xy=(0.95, 0.93),
        xytext=(0.95, 0.82),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        arrowprops=dict(
            facecolor="black",
            edgecolor="black",
            width=2.5,
            headwidth=9,
        ),
    )

    geod = Geod(
        ellps="WGS84"
    )

    lon0 = (
        left
        + (right-left)*0.06
    )

    lat0 = (
        bottom
        + (top-bottom)*0.07
    )

    lon1, _, _ = geod.fwd(
        lon0,
        lat0,
        90,
        500,
    )

    ax.plot(
        [lon0, lon1],
        [lat0, lat0],
        color="black",
        linewidth=4,
    )

    ax.text(
        (lon0 + lon1)/2,
        lat0
        + (top-bottom)*0.015,
        "500 m",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )


# ============================================================
# PER-PAIR VISUALIZATION
# ============================================================

def visualize_pairs(
    pairs: list[dict],
    vector_results: dict[str, dict],
) -> None:

    print()
    print("=" * 100)
    print("PAIR VISUALIZATION")
    print("=" * 100)

    for index, pair in enumerate(
        pairs,
        start=1,
    ):

        pair_name = (
            pair["pair_name"]
        )

        print()
        print(
            f"[{index}/{len(pairs)}]",
            pair_name,
        )

        pair_root = (
            OUTPUT_ROOT
            / pair_name
        )

        rasters = (
            pair_root
            / "rasters"
        )

        viz = (
            pair_root
            / "visualization"
        )

        viz.mkdir(
            parents=True,
            exist_ok=True,
        )

        t1 = Path(
            pair["t1_path"]
        )

        t2 = Path(
            pair["t2_path"]
        )

        rgb1, extent, source_crs = read_rgb(
            t1
        )

        rgb2, _, _ = read_rgb(
            t2
        )

        h, w = rgb2.shape[:2]

        display_shape = (
            h,
            w,
        )

        any_prob = read_display_raster(
            rasters
            / "any_change_probability.tif",
            display_shape,
        )

        new_prob = read_display_raster(
            rasters
            / "new_building_candidate.tif",
            display_shape,
        )

        demo_prob = read_display_raster(
            rasters
            / "demolition_candidate.tif",
            display_shape,
        )

        new_mask = read_display_raster(
            rasters
            / "new_building_mask.tif",
            display_shape,
            nearest=True,
        )

        demo_mask = read_display_raster(
            rasters
            / "demolition_mask.tif",
            display_shape,
            nearest=True,
        )

        metric = vector_results[
            pair_name
        ]

        new_metric = metric[
            "new"
        ]

        demo_metric = metric[
            "demo"
        ]

        new_map = (
            new_metric.to_crs(
                source_crs
            )
            if not new_metric.empty
            else new_metric
        )

        demo_map = (
            demo_metric.to_crs(
                source_crs
            )
            if not demo_metric.empty
            else demo_metric
        )

        # ----------------------------------------------------
        # 01 BEFORE / AFTER
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(15, 7),
            constrained_layout=True,
        )

        axes[0].imshow(
            rgb1,
            extent=extent,
        )

        axes[0].set_title(
            f"T1 ? {pair['t1_date']}"
        )

        axes[1].imshow(
            rgb2,
            extent=extent,
        )

        axes[1].set_title(
            f"T2 ? {pair['t2_date']}"
        )

        for ax in axes:
            decorate_map(
                ax,
                extent,
            )

        fig.suptitle(
            "Tongzhou Multi-temporal Satellite Comparison",
            fontsize=16,
        )

        fig.savefig(
            viz
            / "01_before_after.png",
            dpi=220,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # 02 RAW MASK CHANGE MAP
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(12, 9)
        )

        ax.imshow(
            rgb2,
            extent=extent,
        )

        overlay = np.zeros(
            (
                h,
                w,
                4,
            ),
            dtype=np.float32,
        )

        overlay[
            new_mask > 0
        ] = [
            1.0,
            0.08,
            0.08,
            0.62,
        ]

        overlay[
            demo_mask > 0
        ] = [
            0.05,
            0.65,
            1.0,
            0.66,
        ]

        ax.imshow(
            overlay,
            extent=extent,
        )

        decorate_map(
            ax,
            extent,
        )

        ax.legend(
            handles=[
                Patch(
                    facecolor="red",
                    label="New-building candidate",
                ),
                Patch(
                    facecolor="deepskyblue",
                    label="Demolition candidate",
                ),
            ],
            loc="lower right",
        )

        ax.set_title(
            "Full20 Raw Candidate Map\n"
            f"{pair['t1_date']} ? "
            f"{pair['t2_date']}"
        )

        fig.savefig(
            viz
            / "02_raw_change_map.png",
            dpi=250,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # 03 METRIC OBJECT MAP -- recommended
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(12, 9)
        )

        ax.imshow(
            rgb2,
            extent=extent,
        )

        if not new_map.empty:
            new_map.plot(
                ax=ax,
                facecolor="red",
                edgecolor="darkred",
                alpha=0.45,
                linewidth=0.8,
            )

        if not demo_map.empty:
            demo_map.plot(
                ax=ax,
                facecolor="deepskyblue",
                edgecolor="blue",
                alpha=0.45,
                linewidth=0.8,
            )

        decorate_map(
            ax,
            extent,
        )

        ax.legend(
            handles=[
                Patch(
                    facecolor="red",
                    label=(
                        "New candidate "
                        f"(n={len(new_metric)})"
                    ),
                ),
                Patch(
                    facecolor="deepskyblue",
                    label=(
                        "Demolition candidate "
                        f"(n={len(demo_metric)})"
                    ),
                ),
            ],
            loc="lower right",
        )

        ax.set_title(
            "Object-level Building Change Candidates\n"
            f"{pair['t1_date']} ? "
            f"{pair['t2_date']} | "
            "MMU ? 10 m?"
        )

        fig.savefig(
            viz
            / "03_object_change_map.png",
            dpi=260,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # 04-06 PROBABILITY MAPS
        # ----------------------------------------------------

        probability_maps = [
            (
                "04_any_change_probability.png",
                "Any-change probability",
                any_prob,
                "viridis",
            ),
            (
                "05_new_building_probability.png",
                "New-building candidate probability",
                new_prob,
                "magma",
            ),
            (
                "06_demolition_probability.png",
                "Demolition candidate probability",
                demo_prob,
                "Blues",
            ),
        ]

        for (
            filename,
            title,
            values,
            cmap,
        ) in probability_maps:

            fig, ax = plt.subplots(
                figsize=(12, 9)
            )

            ax.imshow(
                rgb2,
                extent=extent,
                alpha=0.48,
            )

            im = ax.imshow(
                values,
                extent=extent,
                cmap=cmap,
                alpha=0.72,
                vmin=0,
                vmax=1,
            )

            fig.colorbar(
                im,
                ax=ax,
                fraction=0.035,
                label="Probability",
            )

            decorate_map(
                ax,
                extent,
            )

            ax.set_title(
                f"{title}\n"
                f"{pair['t1_date']} ? "
                f"{pair['t2_date']}"
            )

            fig.savefig(
                viz / filename,
                dpi=230,
                bbox_inches="tight",
            )

            plt.close(fig)

        # ----------------------------------------------------
        # 07 HOTSPOT MAP
        # ----------------------------------------------------

        frames = []

        for gdf, category in [
            (
                new_map,
                "new",
            ),
            (
                demo_map,
                "demolition",
            ),
        ]:
            if gdf.empty:
                continue

            temp = gdf.copy()

            temp[
                "_category"
            ] = category

            frames.append(temp)

        fig, ax = plt.subplots(
            figsize=(12, 9)
        )

        ax.imshow(
            rgb2,
            extent=extent,
            alpha=0.52,
        )

        if frames:
            combined = gpd.GeoDataFrame(
                __import__("pandas").concat(
                    frames,
                    ignore_index=True,
                ),
                crs=source_crs,
            )

            centroids = (
                combined
                .to_crs(METRIC_CRS)
                .centroid
            )

            centroids_geo = (
                gpd.GeoSeries(
                    centroids,
                    crs=METRIC_CRS,
                )
                .to_crs(source_crs)
            )

            xs = np.asarray([
                p.x
                for p in centroids_geo
            ])

            ys = np.asarray([
                p.y
                for p in centroids_geo
            ])

            heat, y_edges, x_edges = (
                np.histogram2d(
                    ys,
                    xs,
                    bins=30,
                    range=[
                        [
                            extent[2],
                            extent[3],
                        ],
                        [
                            extent[0],
                            extent[1],
                        ],
                    ],
                )
            )

            masked_heat = np.ma.masked_where(
                heat <= 0,
                heat,
            )

            im = ax.imshow(
                masked_heat,
                extent=extent,
                origin="lower",
                cmap="inferno",
                alpha=0.70,
            )

            fig.colorbar(
                im,
                ax=ax,
                fraction=0.035,
                label="Candidate density",
            )

        decorate_map(
            ax,
            extent,
        )

        ax.set_title(
            "Urban Change Candidate Hotspots\n"
            f"{pair['t1_date']} ? "
            f"{pair['t2_date']}"
        )

        fig.savefig(
            viz
            / "07_change_hotspot.png",
            dpi=250,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            "visualization:",
            viz
        )


# ============================================================
# RASTER STATISTICS
# ============================================================

def mask_fraction(
    path: Path,
) -> float:

    positive = 0
    total = 0

    with rasterio.open(path) as ds:

        for _, window in ds.block_windows(1):
            array = ds.read(
                1,
                window=window,
            )

            valid = np.isfinite(
                array
            )

            positive += int(
                np.sum(
                    (array > 0)
                    & valid
                )
            )

            total += int(
                np.sum(valid)
            )

    if total == 0:
        return float("nan")

    return (
        positive / total
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    pairs: list[dict],
    vector_results: dict[str, dict],
) -> list[dict]:

    summary = []

    for pair in pairs:

        pair_name = (
            pair["pair_name"]
        )

        pair_root = (
            OUTPUT_ROOT
            / pair_name
        )

        rasters = (
            pair_root
            / "rasters"
        )

        metric = vector_results[
            pair_name
        ]

        new_gdf = metric[
            "new"
        ]

        demo_gdf = metric[
            "demo"
        ]

        new_area = (
            float(
                new_gdf[
                    "area_m2"
                ].sum()
            )
            if not new_gdf.empty
            else 0.0
        )

        demo_area = (
            float(
                demo_gdf[
                    "area_m2"
                ].sum()
            )
            if not demo_gdf.empty
            else 0.0
        )

        summary.append({
            "pair":
                pair_name,

            "t1_date":
                pair["t1_date"],

            "t2_date":
                pair["t2_date"],

            "new_objects_mmu10":
                len(new_gdf),

            "demolition_objects_mmu10":
                len(demo_gdf),

            "new_area_m2":
                new_area,

            "demolition_area_m2":
                demo_area,

            "new_area_ha":
                new_area / 10000,

            "demolition_area_ha":
                demo_area / 10000,

            "any_change_pixel_share":
                mask_fraction(
                    rasters
                    / "any_change_mask.tif"
                ),

            "new_pixel_share":
                mask_fraction(
                    rasters
                    / "new_building_mask.tif"
                ),

            "demolition_pixel_share":
                mask_fraction(
                    rasters
                    / "demolition_mask.tif"
                ),

            "t1_aliases":
                pair["t1_aliases"],

            "metric_crs":
                METRIC_CRS,

            "mmu_m2":
                MIN_MAPPING_UNIT_M2,
        })

    out = (
        SUMMARY_ROOT
        / "PAIR_SUMMARY.csv"
    )

    with out.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary
        )

    return summary


# ============================================================
# SUMMARY MAPS
# ============================================================

def summary_figures(
    pairs: list[dict],
    vector_results: dict[str, dict],
    summary: list[dict],
) -> None:

    latest_path = Path(
        pairs[0]["t2_path"]
    )

    latest_rgb, extent, source_crs = (
        read_rgb(
            latest_path,
            max_width=1800,
        )
    )

    # --------------------------------------------------------
    # 01 ALL PERIOD MAP PANELS
    # --------------------------------------------------------

    n = len(pairs)

    cols = 2

    rows = math.ceil(
        n / cols
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            14,
            7 * rows,
        ),
        squeeze=False,
        constrained_layout=True,
    )

    for index, pair in enumerate(
        pairs
    ):

        row = index // cols
        col = index % cols

        ax = axes[
            row,
            col,
        ]

        ax.imshow(
            latest_rgb,
            extent=extent,
        )

        metric = vector_results[
            pair["pair_name"]
        ]

        new_gdf = metric[
            "new"
        ]

        demo_gdf = metric[
            "demo"
        ]

        if not new_gdf.empty:
            new_gdf.to_crs(
                source_crs
            ).plot(
                ax=ax,
                facecolor="red",
                edgecolor="darkred",
                alpha=0.48,
                linewidth=0.7,
            )

        if not demo_gdf.empty:
            demo_gdf.to_crs(
                source_crs
            ).plot(
                ax=ax,
                facecolor="deepskyblue",
                edgecolor="blue",
                alpha=0.48,
                linewidth=0.7,
            )

        decorate_map(
            ax,
            extent,
        )

        ax.set_title(
            f"{pair['t1_date']} ? "
            f"{pair['t2_date']}\n"
            f"New={len(new_gdf)}, "
            f"Demolition={len(demo_gdf)}"
        )

    for blank in range(
        n,
        rows * cols,
    ):
        row = blank // cols
        col = blank % cols

        axes[
            row,
            col
        ].axis("off")

    fig.legend(
        handles=[
            Patch(
                facecolor="red",
                label="New-building candidate",
            ),
            Patch(
                facecolor="deepskyblue",
                label="Demolition candidate",
            ),
        ],
        loc="upper center",
        ncol=2,
    )

    fig.suptitle(
        "Beijing Tongzhou ? Building Change "
        "Relative to Latest Observation",
        fontsize=17,
    )

    fig.savefig(
        SUMMARY_ROOT
        / "01_all_periods_change_maps.png",
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 02 OBJECT COUNT
    # --------------------------------------------------------

    labels = [
        row["t1_date"]
        for row in summary
    ]

    new_count = [
        row["new_objects_mmu10"]
        for row in summary
    ]

    demo_count = [
        row[
            "demolition_objects_mmu10"
        ]
        for row in summary
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width/2,
        new_count,
        width,
        label="New-building candidates",
    )

    ax.bar(
        x + width/2,
        demo_count,
        width,
        label="Demolition candidates",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_xlabel(
        "Baseline observation"
    )

    ax.set_ylabel(
        "Object count (MMU ? 10 m?)"
    )

    ax.set_title(
        "Object-level Building Change "
        "Relative to Latest Observation"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SUMMARY_ROOT
        / "02_object_counts.png",
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 03 CHANGE AREA
    # --------------------------------------------------------

    new_area = [
        row["new_area_ha"]
        for row in summary
    ]

    demo_area = [
        row["demolition_area_ha"]
        for row in summary
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width/2,
        new_area,
        width,
        label="New-building candidates",
    )

    ax.bar(
        x + width/2,
        demo_area,
        width,
        label="Demolition candidates",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_xlabel(
        "Baseline observation"
    )

    ax.set_ylabel(
        "Candidate area (ha)"
    )

    ax.set_title(
        "Candidate Building-change Area "
        "Relative to Latest Observation"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SUMMARY_ROOT
        / "03_change_area.png",
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dirs()

    verify_full20()

    _, canonical = (
        audit_tiffs()
    )

    pairs = build_pair_plan(
        canonical
    )

    run_inference(
        pairs
    )

    vector_results = vectorize_all(
        pairs
    )

    visualize_pairs(
        pairs,
        vector_results,
    )

    summary = build_summary(
        pairs,
        vector_results,
    )

    summary_figures(
        pairs,
        vector_results,
        summary,
    )

    report = {
        "status":
            "success",

        "satellite_root":
            str(SAT_ROOT),

        "output_root":
            str(OUTPUT_ROOT),

        "latest_image":
            pairs[0]["t2_path"],

        "latest_date":
            pairs[0]["t2_date"],

        "pair_count":
            len(pairs),

        "metric_crs":
            METRIC_CRS,

        "mmu_m2":
            MIN_MAPPING_UNIT_M2,

        "full20_sha256":
            EXPECTED_FULL20_SHA,

        "test_images_read":
            0,

        "outputs": {
            "tif_audit":
                str(
                    OUTPUT_ROOT
                    / "TONGZHOU_TIF_AUDIT.csv"
                ),

            "duplicate_audit":
                str(
                    OUTPUT_ROOT
                    / "DUPLICATE_AUDIT.csv"
                ),

            "pair_plan":
                str(
                    OUTPUT_ROOT
                    / "PAIR_PLAN.csv"
                ),

            "summary":
                str(
                    SUMMARY_ROOT
                    / "PAIR_SUMMARY.csv"
                ),

            "summary_map":
                str(
                    SUMMARY_ROOT
                    / "01_all_periods_change_maps.png"
                ),
        },
    }

    (
        OUTPUT_ROOT
        / "TONGZHOU_BATCH_MANIFEST.json"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("TONGZHOU PIPELINE COMPLETE")
    print("=" * 100)

    print(
        "pairs   :",
        len(pairs)
    )

    print(
        "summary :",
        SUMMARY_ROOT
        / "PAIR_SUMMARY.csv"
    )

    print(
        "map     :",
        SUMMARY_ROOT
        / "01_all_periods_change_maps.png"
    )

    print()
    print("NO TEST DATA WAS USED.")


if __name__ == "__main__":
    main()
