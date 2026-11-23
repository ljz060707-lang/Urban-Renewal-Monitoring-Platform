"""Shared rendering toolkit for Xi'an tracking cases and publication figures.

Everything is rendered with Pillow only (this runtime has no matplotlib), using
Times New Roman with a serif fallback, English labels, and the real project
imagery plus the real candidate geometry.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "work" / "python-packages"))
spec = importlib.util.spec_from_file_location("xian_raster_core", TOOLS / "xian_rasterize_current_best.py")
core = importlib.util.module_from_spec(spec)
sys.modules["xian_raster_core"] = core
assert spec.loader is not None
spec.loader.exec_module(core)

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from shapely.ops import transform as shapely_transform  # noqa: E402

SCENE_IMAGE = {
    "xa180429": Path(r"F:\xian\xian\xa180429_combined\xa180429_zoom_20.tif"),
    "xa190101": Path(r"F:\xian\xian\xa190101_combined\xa190101_zoom_20.tif"),
    "xa201231": Path(r"F:\xian\xian\xa201231_combined\xa201231_zoom_20.tif"),
}
SCENE_LABEL = {"xa180429": "2018-04-29", "xa190101": "2019-01-01", "xa201231": "2020-12-31"}
SCENE_ORDER = ("xa180429", "xa190101", "xa201231")

STATE_COLOR = {
    "STABLE": (40, 130, 255),
    "ADDITION": (0, 190, 90),
    "DEMOLITION": (230, 40, 40),
    "UNKNOWN_CONFLICT": (255, 175, 0),
}

_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\timesbd.ttf",
    r"C:\Windows\Fonts\georgia.ttf",
    r"C:\Windows\Fonts\DejaVuSerif.ttf",
)


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    paths = ([r"C:\Windows\Fonts\timesbd.ttf"] if bold else []) + list(FONT_CANDIDATES)
    for p in paths:
        if Path(p).exists():
            try:
                _FONT_CACHE[key] = ImageFont.truetype(p, size)
                return _FONT_CACHE[key]
            except Exception:
                continue
    _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


class SceneReader:
    """Reads native-resolution RGB crops from a scene image, with a small cache."""

    def __init__(self) -> None:
        self._src: dict[str, rasterio.DatasetReader] = {}
        self._to_metric = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
        self._to_wgs = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)

    def src(self, scene: str):
        if scene not in self._src:
            self._src[scene] = rasterio.open(SCENE_IMAGE[scene])
        return self._src[scene]

    def close(self) -> None:
        for s in self._src.values():
            s.close()
        self._src.clear()

    def crop_px(self, scene: str, cx: float, cy: float, size: int) -> tuple[np.ndarray, Window]:
        """Crop `size` x `size` native pixels around an EPSG:32649 centroid."""
        lon, lat = self._to_wgs.transform(cx, cy)
        src = self.src(scene)
        col, row = (~src.transform) * (lon, lat)
        half = size // 2
        col0 = int(round(col)) - half
        row0 = int(round(row)) - half
        col0 = max(0, min(col0, src.width - size))
        row0 = max(0, min(row0, src.height - size))
        win = Window(col0, row0, size, size)
        return src.read(window=win), win

    def crop_aoi_px(self, scene: str, col0: int, row0: int, size: int) -> tuple[np.ndarray, Window]:
        src = self.src(scene)
        col0 = max(0, min(col0, src.width - size))
        row0 = max(0, min(row0, src.height - size))
        win = Window(col0, row0, size, size)
        return src.read(window=win), win

    def window_bounds_metric(self, scene: str, win: Window) -> tuple[float, float, float, float]:
        src = self.src(scene)
        lon0, lat1 = src.transform * (win.col_off, win.row_off)
        lon1, lat0 = src.transform * (win.col_off + win.width, win.row_off + win.height)
        xs, ys = [], []
        for lon, lat in ((lon0, lat0), (lon0, lat1), (lon1, lat0), (lon1, lat1)):
            x, y = self._to_metric.transform(lon, lat)
            xs.append(x)
            ys.append(y)
        return min(xs), min(ys), max(xs), max(ys)


def stretch_to_uint8(band: np.ndarray) -> np.ndarray:
    a = band.astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max()) or 1.0
    return np.clip((a - lo) / max(1e-6, hi - lo) * 255.0, 0, 255).astype(np.uint8)


def to_rgb(arr: np.ndarray) -> np.ndarray:
    """(bands, h, w) -> (h, w, 3) uint8."""
    if arr.shape[0] >= 3:
        return np.dstack([stretch_to_uint8(arr[i]) for i in range(3)])
    g = stretch_to_uint8(arr[0])
    return np.dstack([g, g, g])


def _px_of(geom):
    """Nearest-neighbour polygon coordinate -> pixel list, in source pixel space."""
    def _to_px(x, y, z=None):
        return (x, y)
    return _to_px


def draw_geometry_outline(
    ax: ImageDraw.ImageDraw,
    geom,
    window: Window,
    src_transform,
    *,
    colour,
    width: int = 2,
    fill_alpha: int = 0,
    base: Image.Image | None = None,
) -> None:
    """Draw an EPSG:4326 polygon into a crop whose window is given in source pixels."""
    inv = ~src_transform
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        rings = [poly.exterior] + list(poly.interiors)
        for ring in rings:
            pts = []
            for x, y in ring.coords:
                col, row = inv * (x, y)
                px = col - window.col_off
                py = row - window.row_off
                pts.append((px, py))
            if len(pts) >= 2:
                ax.line(pts, fill=colour, width=width, joint="curve")
        if fill_alpha and base is not None:
            pts = []
            for x, y in poly.exterior.coords:
                col, row = inv * (x, y)
                pts.append((col - window.col_off, row - window.row_off))
            if len(pts) >= 3:
                layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
                ImageDraw.Draw(layer).polygon(pts, fill=colour + (fill_alpha,))
                base.alpha_composite(layer)


def geom_to_wgs84(geom, to_wgs):
    return shapely_transform(to_wgs.transform, geom)


def render_label_block(
    size: tuple[int, int],
    lines: list[tuple[str, tuple[int, int, int]]],
    *,
    bg=(255, 255, 255),
    pad: int = 10,
    font_size: int = 17,
) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    y = pad
    for text, colour in lines:
        d.text((pad, y), text, font=font(font_size), fill=colour)
        y += font_size + 6
        if y > h - font_size:
            break
    return img


def add_scale_bar(
    img: Image.Image, pixel_m: float, length_m: int = 50, *, colour=(255, 255, 255), text_colour=(255, 255, 255)
) -> None:
    d = ImageDraw.Draw(img)
    w, h = img.size
    px = max(20, int(round(length_m / pixel_m)))
    x1 = w - 16
    x0 = max(16, x1 - px)
    y = h - 20
    d.line([(x0, y), (x1, y)], fill=(0, 0, 0), width=5)
    d.line([(x0, y), (x1, y)], fill=colour, width=3)
    for xx in (x0, x1):
        d.line([(xx, y - 5), (xx, y + 5)], fill=(0, 0, 0), width=4)
        d.line([(xx, y - 5), (xx, y + 5)], fill=colour, width=2)
    label = f"{length_m} m"
    f = font(14)
    tw = d.textlength(label, font=f)
    d.text((x1 - tw, y - 22), label, font=f, fill=(0, 0, 0))
    d.text((x1 - tw - 1, y - 23), label, font=f, fill=text_colour)


def add_north_arrow(img: Image.Image, *, colour=(255, 255, 255), size_px: int = 34) -> None:
    d = ImageDraw.Draw(img)
    x, y = 24, 24
    d.polygon([(x, y), (x - size_px // 3, y + size_px), (x + size_px // 3, y + size_px)], fill=(0, 0, 0))
    d.polygon(
        [(x, y + 3), (x - size_px // 4, y + size_px - 2), (x + size_px // 4, y + size_px - 2)], fill=colour
    )
    d.text((x + size_px // 2 + 2, y + 2), "N", font=font(15, True), fill=(0, 0, 0))
    d.text((x + size_px // 2 + 3, y + 3), "N", font=font(15, True), fill=colour)


def panel_label(img: Image.Image, text: str, *, bg=(255, 255, 255, 235), fg=(0, 0, 0)) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    f = font(16, True)
    tw = d.textlength(text, font=f)
    d.rectangle([6, 6, 14 + tw, 30], fill=bg)
    d.text((10, 9), text, font=f, fill=fg)
