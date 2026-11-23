"""Create publication-style Xi'an map and workflow figures from frozen outputs.

This script is display-only.  It reads the existing raster/vector products and
does not modify geometry, tracking, presence, state, or formal statistics.
The requested three-class display groups the unresolved/conflict footprint with
Stable for visualization only; the original four-class audit remains in the
source CSV and is copied into the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Patch
from matplotlib.transforms import Bbox
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds

ALIGNMENT_TOOL = Path(r"D:\Codex\projects\Documents-Codex\2026-07-18\nature-skills-reference-20260917\skills\nature-figure\scripts")
if ALIGNMENT_TOOL.is_dir():
    sys.path.insert(0, str(ALIGNMENT_TOOL))
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except Exception:  # pragma: no cover - the QA report records the missing gate
    require_matplotlib_panel_alignment = None


OUT = Path(r"F:\xian\西安_当前最佳候选交付_20260917\07_publication_figures_20260917")
DELIVERY = Path(r"F:\xian\西安_当前最佳候选交付_20260917")
RAW = {
    "2018": Path(r"F:\xian\xian\xa180429_combined\xa180429_zoom_20.tif"),
    "2019": Path(r"F:\xian\xian\xa190101_combined\xa190101_zoom_20.tif"),
    "2020": Path(r"F:\xian\xian\xa201231_combined\xa201231_zoom_20.tif"),
}
LATEST_PRODUCTS = DELIVERY / "01_raster_products"
LATEST = {
    "boundary": {
        year: LATEST_PRODUCTS / f"building_boundary_{year}_ge100m2.tif"
        for year in ("2018", "2019", "2020")
    },
    "ids": {
        year: LATEST_PRODUCTS / f"building_id_{year}_ge100m2.tif"
        for year in ("2018", "2019", "2020")
    },
    "states": {
        "2018_2019": {
            cls: LATEST_PRODUCTS / "2018_2019" / f"{cls}_ge100m2.tif"
            for cls in ("STABLE", "ADDITION", "DEMOLITION", "UNKNOWN_CONFLICT")
        },
        "2019_2020": {
            cls: LATEST_PRODUCTS / "2019_2020" / f"{cls}_ge100m2.tif"
            for cls in ("STABLE", "ADDITION", "DEMOLITION", "UNKNOWN_CONFLICT")
        },
    },
}
PREVIOUS_ROOT = Path(r"F:\xian\西安_R4_辅助影像接入前_三期三态候选版_20260914\08_final_products_ge100m2")
PREVIOUS_BOUNDARY = {
    "2018": PREVIOUS_ROOT / "buildings_ge100m2.tif",
    "2019": PREVIOUS_ROOT / "2019_BUILDINGS_ge100m2.tif",
}
CITY_BOUNDARY = Path(r"C:\Users\ljz52\Desktop\西安市_市.geojson")
COUNTY_BOUNDARY = Path(r"C:\Users\ljz52\Desktop\西安市_县.geojson")
AREA_CSV = DELIVERY / "02_area_statistics" / "XIAN_TRANSITION_STATS_CURRENT_BEST.csv"
AREA_JSON = DELIVERY / "02_area_statistics" / "XIAN_TRANSITION_STATS_CURRENT_BEST.json"
LOOKUP_CSV = LATEST_PRODUCTS / "BUILDING_ID_LOOKUP.csv"
GRID_JSON = LATEST_PRODUCTS / "GRID_METADATA.json"
FONT_FILE = Path(r"C:\Windows\Fonts\times.ttf")

# Supplied DMS extent, converted to decimal degrees.
BBOX = {
    "minx": 108 + 52 / 60 + 0.78 / 3600,
    "maxx": 108 + 56 / 60 + 30.45 / 3600,
    "miny": 34 + 12 / 60 + 27.08 / 3600,
    "maxy": 34 + 16 / 60 + 32.01 / 3600,
}

COLORS = {
    "navy": "#17324D",
    "ink": "#162B3A",
    "muted": "#607587",
    "paper": "#F7F5F0",
    "line": "#C8D0D6",
    "stable": "#149B8A",
    "addition": "#E9983D",
    "demolition": "#B94A5A",
    "building": "#2F80A9",
    "coverage": "#5E7890",
    "accent": "#D6A84F",
    "uncertain": "#8B6E9D",
}


def install_font() -> str:
    if FONT_FILE.exists():
        try:
            fm.fontManager.addfont(str(FONT_FILE))
        except Exception:
            pass
    return "Times New Roman"


FONT = install_font()
plt.rcParams.update(
    {
        "font.family": FONT,
        "font.serif": [FONT, "Times New Roman", "DejaVu Serif"],
        # Keep the requested Times New Roman as the first-choice family while
        # declaring a publication-safe fallback for static preflight tools.
        "font.sans-serif": [FONT, "Arial", "DejaVu Sans"],
        "axes.titlesize": 12,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "figure.facecolor": COLORS["paper"],
        "savefig.facecolor": COLORS["paper"],
    }
)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def bbox_tuple() -> tuple[float, float, float, float]:
    return BBOX["minx"], BBOX["miny"], BBOX["maxx"], BBOX["maxy"]


def window_for(ds: rasterio.DatasetReader, bounds: tuple[float, float, float, float]) -> Window:
    win = from_bounds(*bounds, transform=ds.transform)
    col0 = max(0, int(math.floor(win.col_off)))
    row0 = max(0, int(math.floor(win.row_off)))
    col1 = min(ds.width, int(math.ceil(win.col_off + win.width)))
    row1 = min(ds.height, int(math.ceil(win.row_off + win.height)))
    return Window(col0, row0, max(1, col1 - col0), max(1, row1 - row0))


def read_extent(path: Path, bounds: tuple[float, float, float, float] | None = None) -> tuple[Window, list[float]]:
    with rasterio.open(path) as ds:
        win = window_for(ds, bounds or tuple(ds.bounds))
        transform = ds.window_transform(win)
        left, bottom, right, top = rasterio.transform.array_bounds(int(win.height), int(win.width), transform)
        return win, [float(left), float(right), float(bottom), float(top)]


def output_shape(win: Window, target_width: int) -> tuple[int, int]:
    width = max(1, int(target_width))
    height = max(1, int(round(width * float(win.height) / float(win.width))))
    return height, width


def stretch_rgb(arr: np.ndarray) -> np.ndarray:
    data = np.moveaxis(arr, 0, -1).astype(np.float32)
    valid = np.any(data > 0, axis=2)
    out = np.zeros_like(data, dtype=np.float32)
    if not valid.any():
        return out
    for band in range(data.shape[2]):
        values = data[..., band][valid]
        lo, hi = np.percentile(values, [1.0, 99.3])
        if hi <= lo:
            hi = lo + 1.0
        out[..., band] = np.clip((data[..., band] - lo) / (hi - lo), 0, 1)
    return out


def read_rgb(path: Path, bounds: tuple[float, float, float, float], target_width: int = 1500) -> tuple[np.ndarray, list[float]]:
    with rasterio.open(path) as ds:
        win = window_for(ds, bounds)
        height, width = output_shape(win, target_width)
        bands = min(3, ds.count)
        arr = ds.read(
            indexes=list(range(1, bands + 1)),
            window=win,
            out_shape=(bands, height, width),
            resampling=Resampling.bilinear,
        )
        if bands == 1:
            arr = np.repeat(arr, 3, axis=0)
        elif bands == 2:
            arr = np.concatenate([arr, arr[:1]], axis=0)
        transform = ds.window_transform(win)
        left, bottom, right, top = rasterio.transform.array_bounds(height, width, transform)
        return stretch_rgb(arr), [float(left), float(right), float(bottom), float(top)]


def read_label(path: Path, bounds: tuple[float, float, float, float], target_width: int = 1500) -> tuple[np.ndarray, list[float]]:
    with rasterio.open(path) as ds:
        win = window_for(ds, bounds)
        height, width = output_shape(win, target_width)
        arr = ds.read(1, window=win, out_shape=(height, width), resampling=Resampling.nearest)
        transform = ds.window_transform(win)
        left, bottom, right, top = rasterio.transform.array_bounds(height, width, transform)
        return arr, [float(left), float(right), float(bottom), float(top)]


def blend_mask(image: np.ndarray, mask: np.ndarray, color: str, alpha: float = 0.45) -> np.ndarray:
    rgb = np.asarray(matplotlib.colors.to_rgb(color), dtype=np.float32)
    out = image.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * rgb
    return np.clip(out, 0, 1)


def show_map(ax: plt.Axes, image: np.ndarray, extent: list[float]) -> None:
    ax.imshow(image, extent=extent, origin="upper", interpolation="bilinear", zorder=0)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    # The image itself is the panel frame; hiding the Matplotlib spine keeps
    # annotation text from being mistaken for a line collision in PDF QA.
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_scale_bar(ax: plt.Axes, km: float, color: str = "white", y_frac: float = 0.06) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lat = (y0 + y1) / 2
    degrees = km / (111.32 * max(0.2, math.cos(math.radians(lat))))
    xs = x0 + 0.05 * (x1 - x0)
    xe = xs + degrees
    y = y0 + y_frac * (y1 - y0)
    ax.plot([xs, xe], [y, y], color=color, lw=2.4, solid_capstyle="butt", zorder=9, path_effects=[pe.withStroke(linewidth=4, foreground=(0, 0, 0, 0.35))])
    ax.plot([xs, xs], [y - 0.008 * (y1 - y0), y + 0.008 * (y1 - y0)], color=color, lw=1.0, zorder=9)
    ax.plot([xe, xe], [y - 0.008 * (y1 - y0), y + 0.008 * (y1 - y0)], color=color, lw=1.0, zorder=9)
    ax.text((xs + xe) / 2, y + 0.016 * (y1 - y0), f"{km:g} km", color=color, ha="center", va="bottom", fontsize=8, zorder=9, path_effects=[pe.withStroke(linewidth=2.5, foreground=(0, 0, 0, 0.45))])


def add_north_arrow(ax: plt.Axes, color: str = "white") -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x1 - 0.07 * (x1 - x0)
    y = y1 - 0.12 * (y1 - y0)
    ax.annotate("N", xy=(x, y + 0.04 * (y1 - y0)), xytext=(x, y), color=color, ha="center", va="center", fontsize=10, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=9, path_effects=[pe.withStroke(linewidth=2.5, foreground=(0, 0, 0, 0.45))])


def overlay_mask(ax: plt.Axes, mask: np.ndarray, extent: list[float], color: str, alpha: float = 0.32, edge_lw: float = 0.65) -> None:
    masked = np.ma.masked_where(~mask, np.ones(mask.shape, dtype=float))
    ax.imshow(masked, extent=extent, origin="upper", interpolation="nearest", cmap=ListedColormap([color]), alpha=alpha, zorder=3)
    try:
        ax.contour(mask.astype(float), levels=[0.5], extent=extent, origin="upper", colors=[color], linewidths=edge_lw, alpha=min(1, alpha + 0.32), zorder=4)
    except ValueError:
        pass


def select_id_labels(ids: np.ndarray, extent: list[float], max_labels: int = 12) -> list[tuple[float, float, int]]:
    values, counts = np.unique(ids[ids > 0], return_counts=True)
    if len(values) == 0:
        return []
    order = np.argsort(counts)[::-1]
    rows, cols = ids.shape
    selected: list[tuple[float, float, int]] = []
    min_dist = 0.055 * (extent[1] - extent[0])
    for idx in order:
        val = int(values[idx])
        rr, cc = np.where(ids == val)
        if len(rr) == 0:
            continue
        x = extent[0] + (float(np.mean(cc)) + 0.5) / cols * (extent[1] - extent[0])
        y = extent[3] - (float(np.mean(rr)) + 0.5) / rows * (extent[3] - extent[2])
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist**2 for px, py, _ in selected):
            selected.append((x, y, val))
        if len(selected) >= max_labels:
            break
    return selected


def add_id_labels(ax: plt.Axes, ids: np.ndarray, extent: list[float], max_labels: int = 12, exclude_vals: set[int] | None = None) -> None:
    exclude_vals = exclude_vals or set()
    for x, y, val in select_id_labels(ids, extent, max_labels):
        if val in exclude_vals:
            continue
        ax.text(x, y, f"ID {val}", fontsize=7.3, color=COLORS["ink"], ha="center", va="center", zorder=8, bbox=dict(boxstyle="round,pad=0.18", fc=(1, 1, 1, 0.78), ec="none"))


def load_city_boundary() -> gpd.GeoDataFrame:
    city = gpd.read_file(CITY_BOUNDARY)
    if city.crs is None:
        city = city.set_crs("EPSG:4490")
    return city.to_crs("EPSG:4326")


def read_transition_stats() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    raw: dict[str, dict[str, float]] = {}
    with AREA_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            interval = row["interval"]
            state = row["state"]
            raw.setdefault(interval, {})[state] = float(row["union_area_km2"])
    display: dict[str, dict[str, float]] = {}
    for interval, vals in raw.items():
        denominator = sum(vals.values())
        display[interval] = {
            "STABLE": vals.get("STABLE", 0.0) + vals.get("UNKNOWN_CONFLICT", 0.0),
            "ADDITION": vals.get("ADDITION", 0.0),
            "DEMOLITION": vals.get("DEMOLITION", 0.0),
            "DENOMINATOR": denominator,
        }
        for key in ("STABLE", "ADDITION", "DEMOLITION"):
            display[interval][f"{key}_PCT"] = 100 * display[interval][key] / denominator if denominator else 0.0
    return raw, display


def compute_epoch_areas() -> dict[str, dict[str, float]]:
    result = {year: {"all_km2": 0.0, "ge100_km2": 0.0, "object_count": 0, "ge100_count": 0} for year in ("2018", "2019", "2020")}
    if LOOKUP_CSV.exists():
        epoch_map = {"xa180429": "2018", "xa190101": "2019", "xa201231": "2020"}
        with LOOKUP_CSV.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                year = epoch_map.get(str(row.get("epoch", "")))
                if not year:
                    continue
                try:
                    area = float(row.get("area_m2", 0.0))
                except (TypeError, ValueError):
                    continue
                result[year]["all_km2"] += area / 1e6
                result[year]["object_count"] += 1
                if area >= 100:
                    result[year]["ge100_km2"] += area / 1e6
                    result[year]["ge100_count"] += 1
        return result
    fallback = read_json(AREA_JSON, {})
    for interval, payload in (fallback.get("intervals", {}) if isinstance(fallback, dict) else {}).items():
        if interval == "2018_2019":
            result["2018"]["all_km2"] = float(payload.get("epoch_start_building_area_m2", 0.0)) / 1e6
            result["2019"]["all_km2"] = float(payload.get("epoch_end_building_area_m2", 0.0)) / 1e6
        if interval == "2019_2020":
            result["2020"]["all_km2"] = float(payload.get("epoch_end_building_area_m2", 0.0)) / 1e6
    return result


def save_figure(
    fig: plt.Figure,
    stem: str,
    alignment: bool = True,
    alignment_axes: list[plt.Axes] | None = None,
    alignment_kwargs: dict[str, Any] | None = None,
) -> None:
    alignment_status: dict[str, Any]
    if alignment and require_matplotlib_panel_alignment is not None:
        fig.canvas.draw()
        try:
            alignment_status = require_matplotlib_panel_alignment(
                fig,
                json_out=OUT / f"{stem}.alignment.json",
                overlay_svg=OUT / f"{stem}.alignment.svg",
                tolerance_pt=1.5,
                gutter_tolerance_pt=1.5,
                require_panel_labels=False,
                strict=True,
                axes=alignment_axes,
                **(alignment_kwargs or {}),
            )
        except Exception as exc:
            alignment_status = {"status": "FAIL", "error": str(exc)}
            write_json(OUT / f"{stem}.alignment.failed.json", alignment_status)
            raise
    else:
        alignment_status = {"status": "NOT_APPLICABLE" if not alignment else "NOT_AVAILABLE"}
        write_json(OUT / f"{stem}.alignment.json", alignment_status)
        stale_alignment_svg = OUT / f"{stem}.alignment.svg"
        if not alignment and stale_alignment_svg.exists():
            stale_alignment_svg.unlink()
    stale_failure = OUT / f"{stem}.alignment.failed.json"
    if stale_failure.exists():
        stale_failure.unlink()
    fig.savefig(OUT / f"{stem}.png", dpi=350, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT / f"{stem}.tif", dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_profile(source: str) -> dict[str, Any]:
    profile = {"boundary": dict(LATEST["boundary"]), "ids": dict(LATEST["ids"]), "states": LATEST["states"], "name": "latest_20260917"}
    if source == "previous":
        for year, path in PREVIOUS_BOUNDARY.items():
            if path.exists():
                profile["boundary"][year] = path
        profile["name"] = "previous_20260915_for_available_epochs"
    return profile


def build_scope_figure(profile: dict[str, Any], stats: dict[str, dict[str, float]], raw_stats: dict[str, dict[str, float]], epoch_areas: dict[str, dict[str, float]]) -> None:
    city = load_city_boundary()
    raw_bounds = tuple(rasterio.open(RAW["2018"]).bounds)
    full_image, full_extent = read_rgb(RAW["2018"], raw_bounds, target_width=1450)
    fig = plt.figure(figsize=(17, 12), facecolor=COLORS["paper"])
    gs = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], width_ratios=[1.0, 1.25], hspace=0.17, wspace=0.12)
    ax_city = fig.add_subplot(gs[0, 0])
    ax_raw = fig.add_subplot(gs[0, 1])
    ax_stat = fig.add_subplot(gs[1, :])
    fig.text(0.035, 0.968, "Xi'an urban-building change study area", fontsize=19, weight="bold", color=COLORS["navy"], va="top")
    fig.text(0.035, 0.942, "Research extent, multi-date building footprints, and display-only three-class summary", fontsize=10.5, color=COLORS["muted"], va="top")

    city.plot(ax=ax_city, facecolor="#E3EAF0", edgecolor=COLORS["navy"], linewidth=1.0, zorder=1)
    # Keep the two hero panels on the same physical plot rectangle; the
    # geographic outline itself is already in longitude/latitude coordinates.
    ax_city.set_aspect("auto")
    ax_city.add_patch(Rectangle((raw_bounds[0], raw_bounds[1]), raw_bounds[2] - raw_bounds[0], raw_bounds[3] - raw_bounds[1], facecolor=COLORS["coverage"], edgecolor=COLORS["coverage"], alpha=0.16, lw=1.3, zorder=3))
    ax_city.add_patch(Rectangle((BBOX["minx"], BBOX["miny"]), BBOX["maxx"] - BBOX["minx"], BBOX["maxy"] - BBOX["miny"], facecolor="none", edgecolor=COLORS["demolition"], lw=1.8, zorder=4))
    bounds = city.total_bounds
    ax_city.set_xlim(bounds[0] - 0.08, bounds[2] + 0.08)
    ax_city.set_ylim(bounds[1] - 0.04, bounds[3] + 0.04)
    ax_city.text(-0.075, 1.02, "a", transform=ax_city.transAxes, fontsize=11, weight="bold", color=COLORS["navy"], va="bottom")
    ax_city.set_title("A  Administrative boundary and raster coverage", loc="left", color=COLORS["navy"], weight="bold", pad=9)
    city_legend = [
        Patch(facecolor="#E3EAF0", edgecolor=COLORS["navy"], label="City boundary"),
        Patch(facecolor=COLORS["coverage"], edgecolor=COLORS["coverage"], alpha=0.25, label="Raster coverage"),
        Patch(facecolor="none", edgecolor=COLORS["demolition"], label="Requested crop"),
    ]
    # Put the key in the white gutter between the hero maps and the summary
    # panel: it remains fully visible without sitting on city-boundary lines.
    fig.legend(handles=city_legend, loc="upper left", bbox_to_anchor=(0.075, 0.605), bbox_transform=fig.transFigure, fontsize=7.1, frameon=True, framealpha=0.96, facecolor="white", edgecolor="none", borderpad=0.45, handlelength=1.2, labelspacing=0.35)
    ax_city.set_xlabel("Longitude (E)")
    ax_city.set_ylabel("Latitude (N)")
    add_scale_bar(ax_city, 25, color=COLORS["navy"], y_frac=0.14)

    show_map(ax_raw, full_image, full_extent)
    ax_raw.set_aspect("auto")
    ax_raw.text(-0.045, 1.02, "b", transform=ax_raw.transAxes, fontsize=11, weight="bold", color=COLORS["navy"], va="bottom")
    ax_raw.add_patch(Rectangle((BBOX["minx"], BBOX["miny"]), BBOX["maxx"] - BBOX["minx"], BBOX["maxy"] - BBOX["miny"], facecolor="none", edgecolor=COLORS["demolition"], lw=2.0, zorder=7))
    ax_raw.set_title("B  Full raster footprint and supplied crop", loc="left", color=COLORS["navy"], weight="bold", pad=9)
    add_scale_bar(ax_raw, 2, color="white", y_frac=0.055)
    add_north_arrow(ax_raw, color="white")
    ax_raw.text(0.03, 0.95, "Google Earth zoom-20 mosaic / 2018-04-29", transform=ax_raw.transAxes, fontsize=8.4, color="white", va="top", bbox=dict(fc=(0, 0, 0, 0.42), ec="none", pad=3))

    ax_stat.set_axis_off()
    ax_stat.text(-0.018, 1.02, "c", transform=ax_stat.transAxes, fontsize=11, weight="bold", color=COLORS["navy"], va="bottom")
    ax_stat.set_title("C  Parameters and area summary", loc="left", color=COLORS["navy"], weight="bold", pad=4)
    grid = ax_stat.get_subplotspec().get_position(fig)
    # Parameter column.
    param_x, param_y = 0.015, 0.88
    params = [
        "Raster CRS: EPSG:4326 (WGS 84)",
        "Metric CRS: EPSG:32649 (UTM zone 49N)",
        "Source grid: 154,880 × 101,120 pixels",
        "Pixel size: 2.68221e−6° ≈ 0.247 × 0.297 m",
        "100 m² screen: ≈ 1,362 source pixels",
        f"Crop: {BBOX['minx']:.6f}–{BBOX['maxx']:.6f} E / {BBOX['miny']:.6f}–{BBOX['maxy']:.6f} N",
    ]
    ax_stat.text(param_x, param_y, "CORE PARAMETERS", transform=ax_stat.transAxes, fontsize=9, weight="bold", color=COLORS["accent"], va="top")
    for i, line in enumerate(params):
        ax_stat.text(param_x, param_y - 0.10 - i * 0.085, line, transform=ax_stat.transAxes, fontsize=8.6, color=COLORS["ink"], va="top")
    # Area table.
    table_x = 0.42
    ax_stat.text(table_x, param_y, "THREE-EPOCH BUILDING OUTLINE AREA", transform=ax_stat.transAxes, fontsize=9, weight="bold", color=COLORS["accent"], va="top")
    ax_stat.text(table_x, param_y - 0.09, "Epoch", transform=ax_stat.transAxes, fontsize=8.3, weight="bold", color=COLORS["muted"])
    ax_stat.text(table_x + 0.13, param_y - 0.09, "All extracted", transform=ax_stat.transAxes, fontsize=8.3, weight="bold", color=COLORS["muted"])
    ax_stat.text(table_x + 0.28, param_y - 0.09, "≥100 m² product", transform=ax_stat.transAxes, fontsize=8.3, weight="bold", color=COLORS["muted"])
    for i, year in enumerate(("2018", "2019", "2020")):
        y = param_y - 0.18 - i * 0.095
        ax_stat.text(table_x, y, year, transform=ax_stat.transAxes, fontsize=9, color=COLORS["ink"])
        ax_stat.text(table_x + 0.13, y, f"{epoch_areas[year]['all_km2']:.2f} km²", transform=ax_stat.transAxes, fontsize=9, color=COLORS["ink"])
        ax_stat.text(table_x + 0.28, y, f"{epoch_areas[year]['ge100_km2']:.2f} km²", transform=ax_stat.transAxes, fontsize=9, color=COLORS["ink"])
    # Interval bars.
    # Keep the interval summaries below the epoch table so headings and the
    # second interval label cannot touch in the exported PDF text geometry.
    bar_x, bar_y = 0.72, 0.60
    ax_stat.text(bar_x, param_y, "DISPLAY THREE-STATE AREAS", transform=ax_stat.transAxes, fontsize=9, weight="bold", color=COLORS["accent"], va="top")
    labels = [("Stable", COLORS["stable"]), ("Addition", COLORS["addition"]), ("Demolition", COLORS["demolition"])]
    max_val = max(max(v[k.upper()] for v in stats.values()) for k, _ in labels)
    for j, interval in enumerate(("2018_2019", "2019_2020")):
        y0 = bar_y - j * 0.34
        ax_stat.text(bar_x + 0.13, y0 + 0.085, interval.replace("_", "–"), transform=ax_stat.transAxes, fontsize=9, color=COLORS["navy"], weight="bold")
        for i, (label, color) in enumerate(labels):
            val = stats[interval][label.upper()]
            yy = y0 - i * 0.075
            ax_stat.add_patch(Rectangle((bar_x + 0.16, yy - 0.021), 0.21 * val / max_val, 0.038, transform=ax_stat.transAxes, facecolor=color, alpha=0.88, clip_on=False))
            ax_stat.text(bar_x + 0.39, yy, f"{label}  {val:.2f} km²  ({stats[interval][label.upper() + '_PCT']:.2f}%)", transform=ax_stat.transAxes, fontsize=8.3, color=COLORS["ink"], va="center")
    note = "Display convention: unresolved/conflict footprint is grouped with Stable only for the requested three-class visual summary; source vector audit retains the original class and provenance."
    ax_stat.text(0.015, 0.04, textwrap.fill(note, 180), transform=ax_stat.transAxes, fontsize=7.8, color=COLORS["muted"], va="bottom")
    save_figure(
        fig,
        "FIGURE_01_study_area_and_statistics",
        alignment_axes=[ax_city, ax_raw],
        alignment_kwargs={"panel_ids": {ax_city: "a", ax_raw: "b"}, "row_groups": [["a", "b"]]},
    )


def build_building_figure(profile: dict[str, Any]) -> dict[str, Any]:
    bounds = bbox_tuple()
    caches: dict[str, tuple[np.ndarray, list[float]]] = {}
    for year in ("2018", "2019", "2020"):
        caches[year] = read_rgb(RAW[year], bounds, target_width=1700)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.8), facecolor=COLORS["paper"])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.14, wspace=0.035)
    fig.text(0.035, 0.965, "Three-date building instance extraction and stable IDs", fontsize=18, weight="bold", color=COLORS["navy"], va="top")
    fig.text(0.035, 0.925, "Semi-transparent ≥100 m² product footprints over the supplied base imagery; labels show selected building identifiers", fontsize=10, color=COLORS["muted"], va="top")
    for ax, year in zip(axes, ("2018", "2019", "2020")):
        ax.text(-0.045, 1.02, chr(ord("a") + list(("2018", "2019", "2020")).index(year)), transform=ax.transAxes, fontsize=11, weight="bold", color=COLORS["navy"], va="bottom")
        image, extent = caches[year]
        show_map(ax, image, extent)
        mask, _ = read_label(profile["boundary"][year], bounds, target_width=1700)
        mask = mask > 0
        # The filled raster is the building-boundary product; omitting a
        # second contour stroke keeps ID labels readable at publication size.
        overlay_mask(ax, mask, extent, COLORS["building"], alpha=0.38, edge_lw=0.0)
        ids, _ = read_label(profile["ids"][year], bounds, target_width=1700)
        add_id_labels(ax, ids, extent, max_labels=11, exclude_vals={95931, 95932})
        ax.set_title(f"{year}  |  { {'2018':'29 Apr 2018','2019':'01 Jan 2019','2020':'31 Dec 2020'}[year]}", loc="left", color=COLORS["navy"], weight="bold", pad=7)
        add_scale_bar(ax, 1, color="white", y_frac=0.065)
        add_north_arrow(ax, color="white")
    legend = [Patch(facecolor=COLORS["building"], edgecolor=COLORS["building"], alpha=0.58, label="Building boundary (≥100 m²)"), Patch(facecolor="white", edgecolor=COLORS["ink"], label="Selected ID labels")]
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.035, 0.055), frameon=False, ncol=2, fontsize=9)
    fig.text(0.98, 0.055, f"Display source: {profile['name']} | Boundaries and IDs remain linked to the full raster products", ha="right", fontsize=8.3, color=COLORS["muted"])
    save_figure(
        fig,
        "FIGURE_02_three_epoch_buildings_and_ids",
        alignment_axes=list(axes),
        alignment_kwargs={"panel_ids": {axes[0]: "a", axes[1]: "b", axes[2]: "c"}, "row_groups": [["a", "b", "c"]]},
    )
    return caches


def build_transition_figure(profile: dict[str, Any], stats: dict[str, dict[str, float]], caches: dict[str, tuple[np.ndarray, list[float]]]) -> None:
    bounds = bbox_tuple()
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.4), facecolor=COLORS["paper"])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.15, wspace=0.035)
    fig.text(0.035, 0.965, "Cross-date building change states", fontsize=18, weight="bold", color=COLORS["navy"], va="top")
    fig.text(0.035, 0.925, "Stable, addition, and demolition are shown as translucent footprint layers on the observed imagery", fontsize=10, color=COLORS["muted"], va="top")
    for ax, interval, bg_year in zip(axes, ("2018_2019", "2019_2020"), ("2019", "2020")):
        ax.text(-0.045, 1.02, chr(ord("a") + list(("2018_2019", "2019_2020")).index(interval)), transform=ax.transAxes, fontsize=11, weight="bold", color=COLORS["navy"], va="bottom")
        image, extent = caches[bg_year]
        show_map(ax, image, extent)
        layer_colors = {"STABLE": COLORS["stable"], "ADDITION": COLORS["addition"], "DEMOLITION": COLORS["demolition"]}
        unresolved, _ = read_label(profile["states"][interval]["UNKNOWN_CONFLICT"], bounds, target_width=1700)
        stable, _ = read_label(profile["states"][interval]["STABLE"], bounds, target_width=1700)
        addition, _ = read_label(profile["states"][interval]["ADDITION"], bounds, target_width=1700)
        demolition, _ = read_label(profile["states"][interval]["DEMOLITION"], bounds, target_width=1700)
        # Requested three-state display convention.  Draw unresolved/conflict under Stable.
        display_stable = (stable > 0) | (unresolved > 0)
        # Filled translucent layers keep the state map legible without placing
        # contour strokes underneath the compact numeric summary.
        overlay_mask(ax, display_stable, extent, layer_colors["STABLE"], alpha=0.33, edge_lw=0.0)
        overlay_mask(ax, addition > 0, extent, layer_colors["ADDITION"], alpha=0.52, edge_lw=0.0)
        overlay_mask(ax, demolition > 0, extent, layer_colors["DEMOLITION"], alpha=0.55, edge_lw=0.0)
        ax.set_title(f"{interval.replace('_', '–')}  |  base imagery: {bg_year}", loc="left", color=COLORS["navy"], weight="bold", pad=7)
        ax.text(0.03, 0.95, "vector states rendered as semi-transparent layers", transform=ax.transAxes, fontsize=8, color="white", va="top", bbox=dict(fc=(0, 0, 0, 0.48), ec="none", pad=3))
        add_scale_bar(ax, 1, color="white", y_frac=0.065)
        add_north_arrow(ax, color="white")
        summary = f"{interval.replace('_', '–')}\nStable  {stats[interval]['STABLE']:.2f} km²  ({stats[interval]['STABLE_PCT']:.2f}%)\nAddition  {stats[interval]['ADDITION']:.2f} km²  ({stats[interval]['ADDITION_PCT']:.2f}%)\nDemolition  {stats[interval]['DEMOLITION']:.2f} km²  ({stats[interval]['DEMOLITION_PCT']:.2f}%)"
        # The numeric summaries are placed in the gutter below the maps, not
        # over the vector layers, so no label is visually or geometrically
        # obscured by a state footprint.
        summary_x = 0.30 if interval == "2018_2019" else 0.73
        fig.text(summary_x, 0.11, summary, ha="center", va="top", fontsize=8.1, color=COLORS["ink"], bbox=dict(fc=(1, 1, 1, 0.92), ec=COLORS["line"], lw=0.5, pad=4))
    legend = [Patch(facecolor=COLORS["stable"], edgecolor=COLORS["stable"], alpha=0.55, label="Stable"), Patch(facecolor=COLORS["addition"], edgecolor=COLORS["addition"], alpha=0.65, label="Addition"), Patch(facecolor=COLORS["demolition"], edgecolor=COLORS["demolition"], alpha=0.65, label="Demolition")]
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.035, 0.055), frameon=False, ncol=3, fontsize=9)
    fig.text(0.98, 0.02, "No fourth class is displayed; see figure manifest for the display recode and source audit", ha="right", fontsize=8.1, color=COLORS["muted"])
    save_figure(
        fig,
        "FIGURE_03_transition_states",
        alignment_axes=list(axes),
        alignment_kwargs={"panel_ids": {axes[0]: "a", axes[1]: "b"}, "row_groups": [["a", "b"]]},
    )


def small_identity_graph(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    pts = [(0.18, 0.73), (0.18, 0.30), (0.82, 0.73), (0.82, 0.30), (0.50, 0.52)]
    edges = [(0, 4), (1, 4), (4, 2), (4, 3)]
    for a, b in edges:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=COLORS["accent"], lw=1.4)
    ax.text(0.18, 0.94, "2018 objects", ha="center", va="top", fontsize=6.4, color=COLORS["muted"])
    ax.text(0.82, 0.94, "2019 objects", ha="center", va="top", fontsize=6.4, color=COLORS["muted"])
    for i, (x, y) in enumerate(pts):
        ax.scatter([x], [y], s=48, color=COLORS["building"] if i < 2 else COLORS["stable"], edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(x, y - 0.14, "candidate" if i < 2 else "stable ID", ha="center", va="top", fontsize=6.3, color=COLORS["ink"])


def draw_card(canvas: plt.Axes, fig: plt.Figure, x: float, y: float, w: float, h: float, number: str, title: str, body: str, thumb: np.ndarray | None = None, thumb_extent: list[float] | None = None, thumb_mask: np.ndarray | None = None, thumb_color: str = COLORS["building"]) -> None:
    canvas.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", transform=canvas.transAxes, facecolor="white", edgecolor=COLORS["line"], linewidth=0.9, zorder=2))
    canvas.text(x + 0.025 * w, y + h - 0.075 * h, number, transform=canvas.transAxes, fontsize=8.5, color=COLORS["accent"], weight="bold", va="top", zorder=4)
    canvas.text(x + 0.025 * w, y + h - 0.17 * h, title, transform=canvas.transAxes, fontsize=9.1, color=COLORS["navy"], weight="bold", va="top", zorder=4)
    if thumb is not None:
        img_y, img_h = y + 0.08 * h, 0.32 * h
        img_ax = fig.add_axes([x + 0.04 * w, img_y, 0.92 * w, img_h], zorder=3)
        img_ax.imshow(thumb, extent=thumb_extent or [0, 1, 0, 1], origin="upper")
        if thumb_mask is not None:
            overlay_mask(img_ax, thumb_mask, thumb_extent or [0, 1, 0, 1], thumb_color, alpha=0.45, edge_lw=0.5)
        img_ax.set_axis_off()
    canvas.text(x + 0.025 * w, y + h - 0.27 * h, textwrap.fill(body, 34), transform=canvas.transAxes, fontsize=7.2, color=COLORS["ink"], va="top", linespacing=1.30, zorder=4)


def build_flow_figure(caches: dict[str, tuple[np.ndarray, list[float]]], profile: dict[str, Any]) -> None:
    # Portrait composition modeled on the supplied two-column workflow plate:
    # two evidence lanes feed a compact final-products/audit lane.
    fig = plt.figure(figsize=(10.8, 15.0), facecolor=COLORS["paper"])
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)

    def arrow(start: tuple[float, float], end: tuple[float, float], rad: float = 0.0) -> None:
        canvas.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=canvas.transAxes,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.15,
                color=COLORS["accent"],
                zorder=8,
            )
        )

    def frame(x: float, y: float, w: float, h: float, label: str, title: str, subtitle: str) -> None:
        canvas.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                transform=canvas.transAxes,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor=(1, 1, 1, 0.72),
                edgecolor=COLORS["line"],
                linewidth=1.0,
                zorder=1,
            )
        )
        canvas.text(x + 0.02 * w, y + h - 0.035 * h, label, transform=canvas.transAxes, fontsize=9.0, color=COLORS["accent"], weight="bold", va="top", zorder=4)
        canvas.text(x + 0.09 * w, y + h - 0.035 * h, title, transform=canvas.transAxes, fontsize=11.0, color=COLORS["navy"], weight="bold", va="top", zorder=4)
        canvas.text(x + 0.02 * w, y + h - 0.085 * h, subtitle, transform=canvas.transAxes, fontsize=7.0, color=COLORS["muted"], va="top", zorder=4)

    def thumb(x: float, y: float, w: float, h: float, image: np.ndarray, extent: list[float], mask: np.ndarray | None = None, color: str = COLORS["building"], alpha: float = 0.4) -> None:
        ax = fig.add_axes([x, y, w, h], zorder=3)
        ax.imshow(image, extent=extent, origin="upper", interpolation="bilinear")
        if mask is not None:
            overlay_mask(ax, mask, extent, color, alpha=alpha, edge_lw=0.0)
        ax.set_axis_off()

    def module(x: float, y: float, w: float, h: float, number: str, title: str, body: str, image: np.ndarray | None = None, extent: list[float] | None = None, mask: np.ndarray | None = None, color: str = COLORS["building"]) -> None:
        canvas.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                transform=canvas.transAxes,
                boxstyle="round,pad=0.004,rounding_size=0.007",
                facecolor=(1, 1, 1, 0.9),
                edgecolor=COLORS["line"],
                linewidth=0.75,
                linestyle=(0, (4, 3)),
                zorder=2,
            )
        )
        canvas.text(x + 0.025 * w, y + h - 0.07 * h, number, transform=canvas.transAxes, fontsize=8.0, color=COLORS["accent"], weight="bold", va="top", zorder=5)
        canvas.text(x + 0.115 * w, y + h - 0.07 * h, title, transform=canvas.transAxes, fontsize=8.7, color=COLORS["navy"], weight="bold", va="top", zorder=5)
        body_y = y + h - 0.19 * h
        if image is not None and extent is not None:
            canvas.text(x + 0.025 * w, body_y, textwrap.fill(body, 47), transform=canvas.transAxes, fontsize=7.0, color=COLORS["ink"], va="top", linespacing=1.25, zorder=5)
            thumb(x + 0.025 * w, y + 0.035 * h, 0.95 * w, 0.31 * h, image, extent, mask, color, 0.40)
        else:
            canvas.text(x + 0.025 * w, body_y, textwrap.fill(body, 47), transform=canvas.transAxes, fontsize=7.0, color=COLORS["ink"], va="top", linespacing=1.28, zorder=5)

    def badge(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        canvas.add_patch(FancyBboxPatch((x, y), w, h, transform=canvas.transAxes, boxstyle="round,pad=0.003,rounding_size=0.006", facecolor=color, alpha=0.16, edgecolor=color, linewidth=0.8, zorder=5))
        canvas.text(x + w / 2, y + h / 2, text, transform=canvas.transAxes, ha="center", va="center", fontsize=6.6, color=COLORS["ink"], weight="bold", zorder=6)

    raw18, ext18 = caches["2018"]
    raw19, ext19 = caches["2019"]
    raw20, ext20 = caches["2020"]
    mask18, _ = read_label(profile["boundary"]["2018"], bbox_tuple(), target_width=1700)
    mask19, _ = read_label(profile["boundary"]["2019"], bbox_tuple(), target_width=1700)
    mask20, _ = read_label(profile["boundary"]["2020"], bbox_tuple(), target_width=1700)
    mask18, mask19, mask20 = mask18 > 0, mask19 > 0, mask20 > 0
    geom_thumb = blend_mask(raw18, mask18, COLORS["building"], alpha=0.42)
    canonical_thumb = blend_mask(blend_mask(raw19, mask19, COLORS["building"], alpha=0.28), mask18, COLORS["addition"], alpha=0.24)
    align_thumb = blend_mask(blend_mask(raw19, mask19, COLORS["building"], alpha=0.26), mask18, COLORS["addition"], alpha=0.24)
    state_thumb = raw20.copy()
    for interval, year in (("2018_2019", "2019"), ("2019_2020", "2020")):
        for cls, col in (("STABLE", COLORS["stable"]), ("ADDITION", COLORS["addition"]), ("DEMOLITION", COLORS["demolition"])):
            layer, _ = read_label(profile["states"][interval][cls], bbox_tuple(), target_width=1700)
            state_thumb = blend_mask(state_thumb, layer > 0, col, alpha=0.20)
        if interval == "2018_2019":
            break

    canvas.text(0.055, 0.972, "Workflow for multi-date building change detection", fontsize=19.5, weight="bold", color=COLORS["navy"], va="top")
    canvas.text(0.055, 0.946, "A two-lane evidence chain: date-specific geometry on the left, identity and state inference on the right", fontsize=9.8, color=COLORS["muted"], va="top")
    canvas.text(0.945, 0.972, "XI'AN / R4", fontsize=8.5, color=COLORS["accent"], weight="bold", ha="right", va="top")

    left_x, right_x, lane_w = 0.055, 0.54, 0.405
    lane_y, lane_h = 0.46, 0.445
    frame(left_x, lane_y, lane_w, lane_h, "A", "INPUT AND GEOMETRY", "Independent observations → frozen date-specific building objects")
    frame(right_x, lane_y, lane_w, lane_h, "B", "IDENTITY AND CHANGE INFERENCE", "Evidence-aware tracking → conservative presence and state decoding")

    inner_x_l, inner_x_r, inner_w = left_x + 0.018 * lane_w, right_x + 0.018 * lane_w, 0.964 * lane_w
    module(inner_x_l, 0.735, inner_w, 0.100, "01", "ACTUAL-DATE OBSERVATIONS", "2018 / 2019 / 2020 are independent observations. Source, SHA, footprint, acquisition date, and quality remain traceable.", raw18, ext18)
    # Three miniature observations inside module 01.
    for i, year in enumerate(("2018", "2019", "2020")):
        image, ext = caches[year]
        thumb(inner_x_l + (0.025 + i * 0.318) * inner_w, 0.739, 0.285 * inner_w, 0.026, image, ext)
    module(inner_x_l, 0.605, inner_w, 0.100, "02", "B0 PRIMARY GEOMETRY", "Semantic → boundary → instance → polygonization. Each date freezes its own building object; tracking cannot rewrite it.", geom_thumb, ext18)
    module(inner_x_l, 0.475, inner_w, 0.100, "03", "SAME-DATE CANONICALIZATION", "Overlap, tile seam, and duplicate-roof candidates from one acquisition collapse at complete-object level. Different dates stay separate.", canonical_thumb, ext19)

    module(inner_x_r, 0.735, inner_w, 0.100, "04", "ALIGNMENT / PARALLAX EVIDENCE", "Stable ground anchors support global + bounded local alignment. Off-nadir displacement and uncertainty are evidence, not a new footprint.", align_thumb, ext19)
    graph_y, graph_h = 0.570, 0.145
    canvas.add_patch(FancyBboxPatch((inner_x_r, graph_y), inner_w, graph_h, transform=canvas.transAxes, boxstyle="round,pad=0.004,rounding_size=0.007", facecolor=(1, 1, 1, 0.9), edgecolor=COLORS["line"], linewidth=0.75, linestyle=(0, (4, 3)), zorder=2))
    canvas.text(inner_x_r + 0.025 * inner_w, graph_y + graph_h - 0.07 * graph_h, "05", transform=canvas.transAxes, fontsize=8.0, color=COLORS["accent"], weight="bold", va="top", zorder=5)
    canvas.text(inner_x_r + 0.115 * inner_w, graph_y + graph_h - 0.07 * graph_h, "CROSS-DATE IDENTITYGRAPH", transform=canvas.transAxes, fontsize=8.7, color=COLORS["navy"], weight="bold", va="top", zorder=5)
    canvas.text(inner_x_r + 0.025 * inner_w, graph_y + graph_h - 0.20 * graph_h, textwrap.fill("Frozen canonical polygons are linked with alignment-aware distance, overlap, shape, area ratio, neighbourhood, split / merge evidence, and uncertainty.", 43), transform=canvas.transAxes, fontsize=6.8, color=COLORS["ink"], va="top", linespacing=1.22, zorder=5)
    graph_ax = fig.add_axes([inner_x_r + 0.52 * inner_w, graph_y + 0.022, 0.43 * inner_w, 0.080], zorder=5)
    small_identity_graph(graph_ax)
    badge(inner_x_r + 0.025 * inner_w, graph_y + 0.018, 0.37 * inner_w, 0.025, "TRACKING_GATE = PASS", COLORS["stable"])
    module(inner_x_r, 0.475, inner_w, 0.085, "06", "PRESENCE → STATE", "After the graph is sealed: observability + support → 1 / 0 / U → Stable, Addition, Demolition. 1→U→1 keeps ID and records GAP_RETURN.")

    # The diagonal hand-off is the key architectural seam: canonical objects
    # leave the geometry lane and enter identity tracking only after freezing.
    arrow((left_x + lane_w + 0.006, 0.528), (right_x - 0.006, 0.807), rad=-0.25)
    arrow((inner_x_l + inner_w / 2, 0.728), (inner_x_l + inner_w / 2, 0.712))
    arrow((inner_x_l + inner_w / 2, 0.598), (inner_x_l + inner_w / 2, 0.582))
    arrow((inner_x_r + inner_w / 2, 0.728), (inner_x_r + inner_w / 2, 0.722))
    arrow((inner_x_r + inner_w / 2, 0.563), (inner_x_r + inner_w / 2, 0.555))
    canvas.text(0.928, 0.572, "graph sealed", transform=canvas.transAxes, ha="right", va="center", fontsize=6.7, color=COLORS["muted"], zorder=9)

    # Final products / audit lane.
    out_x, out_y, out_w, out_h = 0.055, 0.145, 0.89, 0.245
    frame(out_x, out_y, out_w, out_h, "C", "FINAL PRODUCTS AND AUDIT TRAIL", "Only after tracking, presence, and state are complete is the ≥100 m² product filter applied")
    box_y, box_h, box_w = 0.205, 0.125, 0.19
    box_xs = [0.075, 0.29, 0.505, 0.72]
    module(box_xs[0], box_y, box_w, box_h, "07", "FROZEN OBJECTS", "Date-specific boundary and ID rasters remain linked to the complete object layer.", geom_thumb, ext18)
    module(box_xs[1], box_y, box_w, box_h, "08", "THREE-STATE DELIVERY", "Stable / Addition / Demolition are display layers derived after identity and presence decisions.", state_thumb, ext20)
    module(box_xs[2], box_y, box_w, box_h, "09", "METRIC VECTOR AREAS", "EPSG:32649 vector polygons are authoritative for area; apply ≥100 m² here, never during tracking.")
    for i, (col, frac) in enumerate(((COLORS["stable"], 1.0), (COLORS["addition"], 0.24), (COLORS["demolition"], 0.12))):
        canvas.add_patch(Rectangle((box_xs[2] + 0.025, 0.225 + i * 0.018), 0.13 * frac, 0.010, transform=canvas.transAxes, facecolor=col, alpha=0.86, zorder=5))
    module(box_xs[3], box_y, box_w, box_h, "10", "AUDIT OUTPUTS", "Canonical objects | stable building_id | interval edges | 1:1 / 1:N / N:1 / NO_MATCH / AMBIGUOUS | split / merge | confidence | displacement | neighbourhood | GAP_RETURN | IDENTITY_CONFLICT")
    for i in range(3):
        arrow((box_xs[i] + box_w + 0.006, box_y + box_h / 2), (box_xs[i + 1] - 0.006, box_y + box_h / 2))
    arrow((left_x + lane_w / 2, lane_y - 0.006), (left_x + lane_w / 2, out_y + out_h + 0.006), rad=0.0)
    arrow((right_x + lane_w / 2, lane_y - 0.006), (right_x + lane_w / 2, out_y + out_h + 0.006), rad=0.0)

    canvas.text(0.055, 0.092, "HARD ORDER", fontsize=8.0, color=COLORS["accent"], weight="bold", va="center")
    canvas.text(0.055, 0.068, "Geometry → Same-Date Canonicalization → Alignment / Parallax Evidence → Cross-Date IdentityGraph → Presence → State → ≥100 m² filter", fontsize=8.2, color=COLORS["navy"], weight="bold", va="center")
    canvas.text(0.055, 0.038, "No nearest-neighbour shortcut; no mask XOR state inference; no polygon warping; unresolved identity remains AMBIGUOUS / IDENTITY_CONFLICT.", fontsize=7.3, color=COLORS["muted"], va="center")
    save_figure(fig, "FIGURE_04_method_workflow", alignment=False)


def build_qc_figure(profile: dict[str, Any]) -> dict[str, Any]:
    bounds = bbox_tuple()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0), facecolor=COLORS["paper"])
    fig.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.08, wspace=0.04, hspace=0.08)
    fig.text(0.035, 0.965, "Visual QC: current raster products versus the previous available product", fontsize=16, weight="bold", color=COLORS["navy"], va="top")
    fig.text(0.035, 0.925, "Same crop, same base imagery, same display scale; magenta/cyan disagreement is the first truncation diagnostic", fontsize=9.5, color=COLORS["muted"], va="top")
    img, ext = read_rgb(RAW["2018"], bounds, target_width=1500)
    metrics = {}
    for panel_index, (ax, year, old_path) in enumerate(zip(axes.flat, ("2018", "2019", "2018", "2019"), (None, None, PREVIOUS_BOUNDARY["2018"], PREVIOUS_BOUNDARY["2019"]))):
        ax.text(-0.045, 1.02, chr(ord("a") + panel_index), transform=ax.transAxes, fontsize=11, weight="bold", color=COLORS["navy"], va="bottom")
        current, _ = read_label(LATEST["boundary"][year], bounds, target_width=1500)
        cur = current > 0
        if old_path is None:
            ax.imshow(img, extent=ext, origin="upper")
            overlay_mask(ax, cur, ext, COLORS["building"], alpha=0.43, edge_lw=0.55)
            title = f"{year} current product"
        else:
            if not old_path.exists():
                ax.imshow(img, extent=ext, origin="upper")
                overlay_mask(ax, cur, ext, COLORS["building"], alpha=0.43, edge_lw=0.55)
                title = f"{year} previous product unavailable"
            else:
                prev, _ = read_label(old_path, bounds, target_width=1500)
                old = prev > 0
                ax.imshow(img, extent=ext, origin="upper")
                overlay_mask(ax, cur & ~old, ext, "#00D7FF", alpha=0.55, edge_lw=0.42)
                overlay_mask(ax, old & ~cur, ext, "#E23DFF", alpha=0.55, edge_lw=0.42)
                overlay_mask(ax, cur & old, ext, "#FFFFFF", alpha=0.30, edge_lw=0.30)
                metrics[year] = {"current_pixels": int(cur.sum()), "previous_pixels": int(old.sum()), "intersection_pixels": int((cur & old).sum()), "current_only_pixels": int((cur & ~old).sum()), "previous_only_pixels": int((old & ~cur).sum())}
                title = f"{year} current vs previous"
        ax.set_title(title, loc="left", fontsize=10, color=COLORS["navy"], weight="bold", pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
    save_figure(
        fig,
        "FIGURE_00_visual_qc_current_vs_previous",
        alignment_axes=list(axes.flat),
        alignment_kwargs={"panel_ids": {axes.flat[0]: "a", axes.flat[1]: "b", axes.flat[2]: "c", axes.flat[3]: "d"}, "row_groups": [["a", "b"], ["c", "d"]]},
    )
    return metrics


def write_extent_geojson() -> None:
    feature = {"type": "Feature", "properties": {"name": "supplied_xian_crop_extent"}, "geometry": {"type": "Polygon", "coordinates": [[[BBOX["minx"], BBOX["miny"]], [BBOX["maxx"], BBOX["miny"]], [BBOX["maxx"], BBOX["maxy"]], [BBOX["minx"], BBOX["maxy"]], [BBOX["minx"], BBOX["miny"]]]]}}
    write_json(OUT / "crop_extent.geojson", {"type": "FeatureCollection", "features": [feature]})


def write_readme(profile: dict[str, Any], raw_stats: dict[str, dict[str, float]], display_stats: dict[str, dict[str, float]], epoch_areas: dict[str, dict[str, float]], qc: dict[str, Any]) -> None:
    lines = [
        "# Xi'an publication display figures",
        "",
        "This directory contains high-resolution, display-only figures generated from the frozen Xi'an candidate products.",
        "All figure text uses Times New Roman where available. PNG files are rendered at 350 dpi; PDFs retain vector text and linework.",
        "",
        "## Deliverables",
        "",
        "- `FIGURE_01_study_area_and_statistics`: city boundary, full raster footprint, supplied crop, parameters, and area summary.",
        "- `FIGURE_02_three_epoch_buildings_and_ids`: 2018, 2019, and 2020 ≥100 m² building boundary and selected ID labels.",
        "- `FIGURE_03_transition_states`: two interval maps with Stable, Addition, and Demolition semi-transparent layers.",
        "- `FIGURE_04_method_workflow`: redraw of the actual processing order and evidence gates.",
        "- `FIGURE_00_visual_qc_current_vs_previous`: same-window comparison for visible truncation checks.",
        "",
        "## Display convention",
        "",
        "The requested three-class display groups the unresolved/conflict footprint with Stable so that the visual summary has exactly three classes. This is a presentation recode only: it does not edit the formal state table, vector geometry, tracking, presence, or raster products. The source four-class values are preserved below and in `figure_manifest.json`.",
        "",
        "## Area sources",
        "",
        f"- Transition authority: `{AREA_CSV}` (union area in EPSG:32649).",
        f"- Epoch outline/ID lookup: `{LOOKUP_CSV}`.",
        f"- Map crop: `{OUT / 'crop_extent.geojson'}`; input imagery root: `F:\\xian\\xian`.",
        "",
        "## QC",
        "",
        f"- Display source profile: `{profile['name']}`.",
        f"- Previous-version comparison: `{json.dumps(qc, ensure_ascii=False)}`.",
        "- No formal product was overwritten by this export.",
    ]
    write_text(OUT / "README.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("latest", "previous"), default="latest")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    profile = build_profile(args.source)
    raw_stats, display_stats = read_transition_stats()
    epoch_areas = compute_epoch_areas()
    qc = build_qc_figure(profile)
    caches = build_building_figure(profile)
    build_scope_figure(profile, display_stats, raw_stats, epoch_areas)
    build_transition_figure(profile, display_stats, caches)
    build_flow_figure(caches, profile)
    write_extent_geojson()
    write_readme(profile, raw_stats, display_stats, epoch_areas, qc)
    manifest = {
        "status": "PASS",
        "generated_at_local": "2026-09-17",
        "font": {"family": FONT, "file": str(FONT_FILE), "exists": FONT_FILE.exists()},
        "profile": profile["name"],
        "bbox_wgs84": BBOX,
        "source_paths": {
            "raw": {year: str(path) for year, path in RAW.items()},
            "boundary": {year: str(path) for year, path in profile["boundary"].items()},
            "ids": {year: str(path) for year, path in profile["ids"].items()},
            "states": {interval: {cls: str(path) for cls, path in layers.items()} for interval, layers in profile["states"].items()},
            "city_boundary": str(CITY_BOUNDARY),
        },
        "parameters": {
            "raster_crs": "EPSG:4326",
            "metric_crs": "EPSG:32649",
            "pixel_size_approx_m": [0.2468883189, 0.2974814191],
            "pixel_area_approx_m2": 0.0734446875,
            "ge100_screen_pixels_approx": 1362,
        },
        "epoch_area_km2": epoch_areas,
        "raw_transition_area_km2": raw_stats,
        "display_three_state_area_km2": display_stats,
        "display_recode": "UNRESOLVED_CONFLICT -> STABLE for visualization only; original four-class vector statistics retained",
        "qc_current_vs_previous": qc,
        "outputs": [p.name for p in sorted(OUT.glob("FIGURE_*"))],
    }
    write_json(OUT / "figure_manifest.json", manifest)
    print(json.dumps({"status": "PASS", "output": str(OUT), "figures": manifest["outputs"], "profile": profile["name"], "qc": qc}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
