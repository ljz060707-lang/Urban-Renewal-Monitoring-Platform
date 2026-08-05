from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import struct
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except Exception:  # pragma: no cover - the bundled runtime has Pillow
    Image = None


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "outputs" / "urrs" / "review" / "lite_review"
DB_PATH = DATA_ROOT / "review.sqlite"
EVENTS_PATH = DATA_ROOT / "review_events.jsonl"
BACKUP_ROOT = DATA_ROOT / "backups"
EXPORT_ROOT = DATA_ROOT / "exports"
CHIP_ROOT = DATA_ROOT / "chips"
APP_VERSION = "urrs-lite-review/0.1.0"
_WRITE_LOCK = threading.RLock()

NANJING_DATES = ["2015-01-04", "2016-08-18", "2017-12-11", "2018-02-08", "2019-03-16", "2022-12-02"]
TONGZHOU_DATES = ["2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16"]
NANJING_BOUNDS = (118.632431030273, 32.0237731933594, 118.65234375, 32.0416259765625)

NANJING_IMAGE_ROOT = Path(r"D:\Data Set\Satellite\nanjing_jiangbei 20")
TONGZHOU_IMAGE_ROOT = Path(r"D:\Data Set\Satellite\beijing_tongzhou")
NANJING_RUN = ROOT / "outputs" / "inference" / "nanjing_building_construction_integrated_20260804_233025"
NANJING_STABLE_RUN = ROOT / "outputs" / "inference" / "nanjing_stable_building_construction_20260805_105929"
TONGZHOU_RUN = ROOT / "outputs" / "inference" / "beijing_tongzhou_construction_images_best_20260805_110156"

ALLOWED_ACTIONS = {
    "accept",
    "reject",
    "delete",
    "correct",
    "edit",
    "split",
    "merge",
    "propagate",
    "copy_adjacent",
    "add",
    "ignore",
    "background_confirm",
    "complete",
    "save",
    "skip",
    "later",
    "undo",
    "redo",
}
TRAINING_IMPORTABLE = {"human_verified", "human_corrected"}
HOLDOUT_SPLIT = "holdout"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(value: Any) -> str:
    payload = value if isinstance(value, (bytes, bytearray)) else json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8", newline="")
    os.replace(part, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def row_dict(row: sqlite3.Row | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def _json(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list, tuple)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default if default is not None else value


def project_id(project: str) -> str:
    value = str(project).strip().lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_\-]+", "", value)
    aliases = {"nanjing_building_v1": "nanjing_building", "tongzhou_construction_v1": "tongzhou_construction"}
    return aliases.get(value, value)


def compute_spatial_block(x: float, y: float, project: str, grid: int = 8) -> str:
    """Deterministic block identity. Blocks, not individual objects, define splits."""
    bounds = NANJING_BOUNDS if project_id(project).startswith("nanjing") else None
    if bounds:
        minx, miny, maxx, maxy = bounds
        nx = max(0, min(grid - 1, int((x - minx) / max(maxx - minx, 1e-12) * grid)))
        ny = max(0, min(grid - 1, int((y - miny) / max(maxy - miny, 1e-12) * grid)))
    else:
        # For Tongzhou candidates the GPKG coordinates are still deterministic;
        # quantize after scaling to avoid any dependence on row order.
        nx = int(abs(x) * 1000) % grid
        ny = int(abs(y) * 1000) % grid
    return f"{project_id(project)}_b{nx:02d}_{ny:02d}"


def frozen_split(block_id: str, project: str) -> str:
    """Stable train/validation/holdout assignment with a sealed holdout."""
    digest = int(stable_hash(f"split:{project_id(project)}:{block_id}")[:8], 16) % 100
    if digest < 65:
        return "train"
    if digest < 82:
        return "validation"
    return "holdout"


def _ring_self_intersects(points: Sequence[Sequence[float]]) -> bool:
    if len(points) < 4:
        return False
    pts = [(float(p[0]), float(p[1])) for p in points]
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def onseg(a, b, c):
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    def hit(a, b, c, d):
        o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
        if o1 == 0 and onseg(a, b, c): return True
        if o2 == 0 and onseg(a, b, d): return True
        if o3 == 0 and onseg(c, d, a): return True
        if o4 == 0 and onseg(c, d, b): return True
        return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

    for i in range(len(pts) - 1):
        for j in range(i + 1, len(pts) - 1):
            if j in (i - 1, i, i + 1) or (i == 0 and j == len(pts) - 2):
                continue
            if hit(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                return True
    return False


def validate_geometry(geometry: Any, allow_empty: bool = False) -> tuple[bool, str]:
    if geometry in (None, ""):
        return (allow_empty, "geometry missing")
    if isinstance(geometry, str):
        try:
            geometry = json.loads(geometry)
        except Exception:
            return False, "geometry is not JSON"
    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon", "Point"}:
        return False, "only GeoJSON Point/Polygon/MultiPolygon is supported"
    try:
        if geometry["type"] == "Point":
            coords = geometry["coordinates"]
            if len(coords) < 2 or not all(math.isfinite(float(v)) for v in coords[:2]):
                return False, "invalid point"
        else:
            polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
            for poly in polygons:
                for ring in poly:
                    if len(ring) < 4 or ring[0] != ring[-1]:
                        return False, "polygon ring must be closed and contain four points"
                    if _ring_self_intersects(ring):
                        return False, "self-intersecting polygon"
                    for p in ring:
                        if len(p) < 2 or not all(math.isfinite(float(v)) for v in p[:2]):
                            return False, "invalid polygon coordinate"
    except (KeyError, TypeError, ValueError):
        return False, "malformed geometry"
    return True, ""


def geometry_bbox(geometry: Any) -> tuple[float, float, float, float]:
    if isinstance(geometry, str):
        geometry = _json(geometry, {})
    coords: list[tuple[float, float]] = []
    def walk(v):
        if isinstance(v, (list, tuple)) and len(v) >= 2 and all(isinstance(x, (int, float)) for x in v[:2]):
            coords.append((float(v[0]), float(v[1])))
        elif isinstance(v, (list, tuple)):
            for x in v: walk(x)
    walk((geometry or {}).get("coordinates", []))
    if not coords:
        return (0.0, 0.0, 1.0, 1.0)
    xs, ys = zip(*coords)
    return min(xs), min(ys), max(xs), max(ys)


def geometry_centroid(geometry: Any) -> tuple[float, float]:
    b = geometry_bbox(geometry)
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def utm_to_wgs84(easting: float, northing: float, zone: int = 50) -> tuple[float, float]:
    """Small dependency-free inverse for the EPSG:326xx geometries in URRS GPKGs."""
    a = 6378137.0
    ecc_sq = 0.0066943799901413165
    k0 = 0.9996
    x = float(easting) - 500000.0
    y = float(northing)
    m = y / k0
    mu = m / (a * (1 - ecc_sq / 4 - 3 * ecc_sq**2 / 64 - 5 * ecc_sq**3 / 256))
    e1 = (1 - math.sqrt(1 - ecc_sq)) / (1 + math.sqrt(1 - ecc_sq))
    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512
    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)
    ecc_prime_sq = ecc_sq / (1 - ecc_sq)
    c1 = ecc_prime_sq * math.cos(fp) ** 2
    t1 = math.tan(fp) ** 2
    n1 = a / math.sqrt(1 - ecc_sq * math.sin(fp) ** 2)
    r1 = a * (1 - ecc_sq) / (1 - ecc_sq * math.sin(fp) ** 2) ** 1.5
    d = x / (n1 * k0)
    lat = fp - (n1 * math.tan(fp) / r1) * (d**2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ecc_prime_sq) * d**4 / 24 + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ecc_prime_sq - 3 * c1**2) * d**6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d**3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ecc_prime_sq + 24 * t1**2) * d**5 / 120) / math.cos(fp)
    return zone * 6 - 183 + math.degrees(lon), math.degrees(lat)


def transform_geometry(geometry: Any, to_wgs84: bool = True) -> Any:
    """Convert projected UTM coordinates to display coordinates; leave WGS84 unchanged."""
    if isinstance(geometry, str): geometry = _json(geometry, None)
    if not isinstance(geometry, dict) or not to_wgs84:
        return geometry
    def rec(v):
        if isinstance(v, (list, tuple)) and len(v) >= 2 and all(isinstance(x, (int, float)) for x in v[:2]):
            if abs(float(v[0])) > 1000 or abs(float(v[1])) > 1000:
                x, y = utm_to_wgs84(float(v[0]), float(v[1]))
                return [x, y] + list(v[2:])
            return list(v)
        if isinstance(v, (list, tuple)): return [rec(x) for x in v]
        return v
    result = dict(geometry)
    if "coordinates" in result: result["coordinates"] = rec(result["coordinates"])
    if "geometries" in result: result["geometries"] = [transform_geometry(x, to_wgs84) for x in result["geometries"]]
    return result


def _wkb_reader(data: bytes, offset: int = 0) -> tuple[dict[str, Any] | None, int]:
    if offset >= len(data):
        return None, offset
    endian = data[offset]
    offset += 1
    order = "<" if endian == 1 else ">"
    raw_type = struct.unpack_from(order + "I", data, offset)[0]
    offset += 4
    has_z = bool(raw_type & 0x80000000)
    has_m = bool(raw_type & 0x40000000)
    typ = raw_type & 0x0FFFFFFF
    dim = 2 + int(has_z) + int(has_m)
    if typ == 1:
        vals = struct.unpack_from(order + "d" * dim, data, offset); offset += 8 * dim
        return {"type": "Point", "coordinates": [vals[0], vals[1]]}, offset
    if typ == 3:
        n = struct.unpack_from(order + "I", data, offset)[0]; offset += 4
        rings = []
        for _ in range(n):
            count = struct.unpack_from(order + "I", data, offset)[0]; offset += 4
            ring = []
            for _ in range(count):
                vals = struct.unpack_from(order + "d" * dim, data, offset); offset += 8 * dim
                ring.append([vals[0], vals[1]])
            rings.append(ring)
        return {"type": "Polygon", "coordinates": rings}, offset
    if typ == 6:
        n = struct.unpack_from(order + "I", data, offset)[0]; offset += 4
        polys = []
        for _ in range(n):
            g, offset = _wkb_reader(data, offset)
            if g and g.get("type") == "Polygon": polys.append(g["coordinates"])
        return {"type": "MultiPolygon", "coordinates": polys}, offset
    if typ in (2, 4, 5, 7):  # line/multi/collection fallback
        if typ == 2:
            n = struct.unpack_from(order + "I", data, offset)[0]; offset += 4
            vals = []
            for _ in range(n):
                p, offset = _wkb_reader(data, offset); vals.append(p.get("coordinates", []) if p else [])
            return {"type": "LineString", "coordinates": vals}, offset
        n = struct.unpack_from(order + "I", data, offset)[0]; offset += 4
        geoms = []
        for _ in range(n):
            g, offset = _wkb_reader(data, offset)
            if g: geoms.append(g)
        if typ == 4: return {"type": "MultiPoint", "coordinates": [g.get("coordinates") for g in geoms]}, offset
        if typ == 5: return {"type": "MultiLineString", "coordinates": [g.get("coordinates") for g in geoms]}, offset
        return {"type": "GeometryCollection", "geometries": geoms}, offset
    return None, offset


def gpkg_geometry_to_geojson(blob: bytes | None) -> dict[str, Any] | None:
    if not blob:
        return None
    try:
        data = bytes(blob)
        if data[:2] == b"GP":
            flags = data[3]
            envelope_code = (flags >> 1) & 0x07
            order = "<" if flags & 1 else ">"
            offset = 8
            offset += {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(envelope_code, 0)
            data = data[offset:]
        geom, _ = _wkb_reader(data, 0)
        return geom
    except Exception:
        return None


def read_gpkg_rows(path: Path, limit: int | None = None, table: str | None = None) -> tuple[str, list[dict[str, Any]], tuple[float, float, float, float] | None]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        if table is None:
            row = conn.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features' ORDER BY table_name LIMIT 1").fetchone()
            if not row: return "", [], None
            table = row[0]
        content = conn.execute("SELECT min_x,min_y,max_x,max_y FROM gpkg_contents WHERE table_name=?", (table,)).fetchone()
        bounds = tuple(content) if content and all(v is not None for v in content) else None
        sql = f'SELECT * FROM "{table.replace(chr(34), chr(34)*2)}"'
        if limit: sql += f" LIMIT {int(limit)}"
        rows = []
        for row in conn.execute(sql):
            d = dict(row)
            if "geom" in d:
                d["geometry"] = gpkg_geometry_to_geojson(d.pop("geom"))
            rows.append(d)
        return table, rows, bounds
    finally:
        conn.close()


def image_for_date(project: str, scene_date: str) -> Path | None:
    if project_id(project).startswith("nanjing"):
        suffix = scene_date.replace("-", "")[2:]
        # Exact folder names are retained by the input inventory.
        for p in NANJING_IMAGE_ROOT.rglob("*.tif"):
            if scene_date.replace("-", "")[2:] in p.name:
                return p
    else:
        token = scene_date.replace("-", "")[2:]
        for p in TONGZHOU_IMAGE_ROOT.glob("*.tif"):
            if token in p.name:
                return p
    return None


def _image_bounds(project: str, fallback: tuple[float, float, float, float] | None = None) -> tuple[float, float, float, float]:
    if project_id(project).startswith("nanjing"):
        return NANJING_BOUNDS
    if fallback and all(v is not None for v in fallback):
        return tuple(float(v) for v in fallback)
    # Beijing GPKG extents are normally supplied by prepare_review_queues.
    return (116.90, 39.72, 117.15, 39.95)


def make_chip(source: Path, geometry: Any, output: Path, project: str, bounds: tuple[float, float, float, float] | None = None, size: int = 1024) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return {"path": str(output), "sha256": stable_hash(output.read_bytes()), "reused": True}
    if Image is None or not source.exists():
        return {"path": None, "error": "Pillow or source image unavailable"}
    try:
        with Image.open(source) as im:
            im.seek(0)
            width, height = im.size
            minx, miny, maxx, maxy = _image_bounds(project, bounds)
            cx, cy = geometry_centroid(geometry)
            px = int((cx - minx) / max(maxx - minx, 1e-12) * width)
            py = int((maxy - cy) / max(maxy - miny, 1e-12) * height)
            gx = geometry_bbox(geometry)
            span = max(abs(gx[2] - gx[0]), abs(gx[3] - gx[1]), (maxx - minx) / 256.0, (maxy - miny) / 256.0)
            halfx = max(96, int(span / max(maxx - minx, 1e-12) * width * 2.5))
            halfy = max(96, int(span / max(maxy - miny, 1e-12) * height * 2.5))
            left, top = max(0, px - halfx), max(0, py - halfy)
            right, bottom = min(width, px + halfx), min(height, py + halfy)
            crop = im.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
            crop.thumbnail((size, size), Image.Resampling.LANCZOS)
            if crop.mode not in {"RGB", "RGBA"}: crop = crop.convert("RGB")
            crop.save(output, format="PNG", optimize=True)
        return {"path": str(output), "sha256": stable_hash(output.read_bytes()), "width": crop.width, "height": crop.height}
    except Exception as exc:
        return {"path": None, "error": f"chip generation failed: {exc}"}


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (project_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT, dates_json TEXT, created_at TEXT NOT NULL, sealed INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS review_queues (queue_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, module TEXT NOT NULL, queue_name TEXT NOT NULL, target_ratio REAL, frozen_split INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS samples (sample_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, queue_id TEXT NOT NULL, module TEXT NOT NULL, split TEXT NOT NULL, spatial_block_id TEXT NOT NULL, scene_date TEXT NOT NULL, source_asset TEXT, chip_path TEXT, source_geometry_json TEXT, geometry_json TEXT, reference_json TEXT, metadata_json TEXT, status TEXT NOT NULL DEFAULT 'unreviewed', decision TEXT, error_type TEXT, negative_type TEXT, annotation_status TEXT NOT NULL DEFAULT 'draft', reviewer TEXT, notes TEXT, source_hash TEXT, blind_first INTEGER NOT NULL DEFAULT 0, model_visible INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS spatial_blocks (spatial_block_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, split TEXT NOT NULL, centroid_x REAL, centroid_y REAL, sealed INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scene_assets (asset_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, scene_date TEXT NOT NULL, asset_path TEXT NOT NULL, width INTEGER, height INTEGER, band_count INTEGER, sha256 TEXT, bounds_json TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS model_proposals (proposal_id TEXT PRIMARY KEY, sample_id TEXT NOT NULL, class_name TEXT, geometry_json TEXT, confidence REAL, source TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'reference', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS human_annotations (annotation_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, sample_id TEXT NOT NULL, spatial_block_id TEXT NOT NULL, scene_date TEXT NOT NULL, class_name TEXT, geometry_json TEXT, geometry_wkt TEXT, source TEXT NOT NULL, annotation_status TEXT NOT NULL, annotator TEXT, reviewer TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, confidence REAL, notes TEXT);
CREATE TABLE IF NOT EXISTS valid_masks (mask_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, sample_id TEXT NOT NULL, geometry_json TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ignore_masks (mask_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, sample_id TEXT NOT NULL, geometry_json TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS review_decisions (sample_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, decision TEXT NOT NULL, error_type TEXT, negative_type TEXT, complete_background_confirmed INTEGER NOT NULL DEFAULT 0, notes TEXT, reviewer TEXT, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS review_events (event_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, reviewer TEXT, project_id TEXT NOT NULL, module TEXT NOT NULL, sample_id TEXT NOT NULL, spatial_block_id TEXT, scene_date TEXT, split TEXT, action TEXT NOT NULL, old_value TEXT, new_value TEXT, geometry_before TEXT, geometry_after TEXT, app_version TEXT NOT NULL, payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS review_progress (project_id TEXT NOT NULL, module TEXT NOT NULL, total INTEGER NOT NULL DEFAULT 0, reviewed INTEGER NOT NULL DEFAULT 0, complete_blocks INTEGER NOT NULL DEFAULT 0, train_blocks INTEGER NOT NULL DEFAULT 0, validation_blocks INTEGER NOT NULL DEFAULT 0, holdout_blocks INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, PRIMARY KEY(project_id,module));
CREATE TABLE IF NOT EXISTS split_registry (spatial_block_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, split TEXT NOT NULL, locked INTEGER NOT NULL DEFAULT 1, assigned_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS training_gates (gate_id TEXT PRIMARY KEY, gate_name TEXT NOT NULL, status TEXT NOT NULL, details_json TEXT NOT NULL, checked_at TEXT NOT NULL, frozen INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS export_history (export_id TEXT PRIMARY KEY, project_id TEXT, export_path TEXT NOT NULL, status TEXT NOT NULL, manifest_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_samples_queue ON samples(queue_id, split, status);
CREATE INDEX IF NOT EXISTS idx_samples_block ON samples(spatial_block_id, split);
CREATE INDEX IF NOT EXISTS idx_events_sample ON review_events(sample_id, timestamp);
"""


class ReviewDB:
    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVENTS_PATH.touch(exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._set_default("app_version", APP_VERSION)
        self._set_default("holdout_sealed", "1")
        self._set_default("external_network", "disabled")
        self.conn.commit()

    def close(self):
        self.conn.close()

    def _set_default(self, key: str, value: str):
        self.conn.execute("INSERT OR IGNORE INTO app_settings(key,value,updated_at) VALUES(?,?,?)", (key, value, now_iso()))

    def setting(self, key: str, default: str | None = None) -> str | None:
        r = self.conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
        return r[0] if r else default

    def add_project(self, project: str, name: str, dates: Sequence[str], description: str = "") -> str:
        pid = project_id(project)
        self.conn.execute("INSERT OR IGNORE INTO projects(project_id,name,description,dates_json,created_at,sealed) VALUES(?,?,?,?,?,0)", (pid, name, description, json.dumps(list(dates)), now_iso()))
        self.conn.commit()
        return pid

    def queue_id(self, project: str, module: str) -> str:
        return f"{project_id(project)}:{module}"

    def add_queue(self, project: str, module: str, name: str, ratio: float | None = None) -> str:
        qid = self.queue_id(project, module)
        self.conn.execute("INSERT OR IGNORE INTO review_queues(queue_id,project_id,module,queue_name,target_ratio,frozen_split,created_at) VALUES(?,?,?,?,?,?,?)", (qid, project_id(project), module, name, ratio, 1, now_iso()))
        self.conn.commit()
        return qid

    def add_asset(self, project: str, scene_date: str, path: Path, bounds: tuple[float, float, float, float] | None = None):
        aid = stable_hash([project_id(project), scene_date, str(path.resolve())])[:24]
        width = height = bands = None
        if Image and path.exists():
            try:
                with Image.open(path) as im:
                    width, height = im.size; bands = len(im.getbands())
            except Exception:
                pass
        self.conn.execute("INSERT OR IGNORE INTO scene_assets(asset_id,project_id,scene_date,asset_path,width,height,band_count,sha256,bounds_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)", (aid, project_id(project), scene_date, str(path), width, height, bands, stable_hash(path.read_bytes()) if path.exists() else None, json.dumps(bounds) if bounds else None, now_iso()))
        self.conn.commit()

    def add_block(self, project: str, block: str, x: float, y: float, split_override: str | None = None) -> str:
        pid = project_id(project)
        split = split_override or frozen_split(block, pid)
        if split not in {"train", "validation", HOLDOUT_SPLIT}:
            raise ValueError(f"unsupported frozen split: {split}")
        existing = self.conn.execute("SELECT project_id,split FROM spatial_blocks WHERE spatial_block_id=?", (block,)).fetchone()
        if existing and (existing[0] != pid or existing[1] != split):
            raise ValueError(f"spatial block split is already locked: {block} -> {existing[1]}")
        self.conn.execute("INSERT OR IGNORE INTO spatial_blocks(spatial_block_id,project_id,split,centroid_x,centroid_y,sealed,created_at) VALUES(?,?,?,?,?,?,?)", (block, pid, split, x, y, 1, now_iso()))
        self.conn.execute("INSERT OR IGNORE INTO split_registry(spatial_block_id,project_id,split,locked,assigned_at) VALUES(?,?,?,?,?)", (block, pid, split, 1, now_iso()))
        self.conn.commit()
        return split

    def add_sample(self, *, sample_id: str, project: str, module: str, scene_date: str, block: str, source_asset: Path | None, chip_path: Path | None, source_geometry: Any, reference: Mapping[str, Any] | None = None, metadata: Mapping[str, Any] | None = None, status: str = "unreviewed", negative_type: str | None = None, blind_first: bool = False, model_visible: bool = True, bounds: tuple[float, float, float, float] | None = None, proposal: Mapping[str, Any] | None = None, split_override: str | None = None) -> bool:
        pid = project_id(project); qid = self.add_queue(pid, module, module.replace("_", " ").title())
        cx, cy = geometry_centroid(source_geometry); split = self.add_block(pid, block, cx, cy, split_override)
        inserted = self.conn.execute("INSERT OR IGNORE INTO samples(sample_id,project_id,queue_id,module,split,spatial_block_id,scene_date,source_asset,chip_path,source_geometry_json,geometry_json,reference_json,metadata_json,status,negative_type,blind_first,model_visible,created_at,updated_at,source_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (sample_id, pid, qid, module, split, block, scene_date, str(source_asset) if source_asset else None, str(chip_path) if chip_path else None, json.dumps(source_geometry, ensure_ascii=False), json.dumps(None), json.dumps(reference or {}, ensure_ascii=False), json.dumps(metadata or {}, ensure_ascii=False), status, negative_type, int(blind_first), int(model_visible), now_iso(), now_iso(), stable_hash([str(source_asset), source_geometry])))
        if not inserted.rowcount:
            # Queue regeneration is idempotent but may repair visibility flags
            # for samples created by an interrupted preparation run. Reviewed
            # records are never rewritten.
            self.conn.execute("UPDATE samples SET blind_first=?,model_visible=? WHERE sample_id=? AND status IN ('unreviewed','draft')", (int(blind_first), int(model_visible), sample_id))
        if inserted.rowcount and proposal:
            self.conn.execute("INSERT OR IGNORE INTO model_proposals(proposal_id,sample_id,class_name,geometry_json,confidence,source,status,created_at) VALUES(?,?,?,?,?,?,?,?)", (proposal.get("proposal_id") or stable_hash([sample_id, "proposal"])[:24], sample_id, proposal.get("class_name"), json.dumps(source_geometry), proposal.get("confidence"), proposal.get("source", "model_reference"), "reference", now_iso()))
        self.conn.commit()
        return bool(inserted.rowcount)

    def sample(self, sample_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
        if not r: return None
        d = dict(r)
        for key in ("source_geometry_json", "geometry_json", "reference_json", "metadata_json"):
            d[key[:-5] if key.endswith("_json") else key] = _json(d.pop(key), {})
        d["annotations"] = [dict(x) for x in self.conn.execute("SELECT * FROM human_annotations WHERE sample_id=? ORDER BY created_at", (sample_id,))]
        d["valid_masks"] = [dict(x) for x in self.conn.execute("SELECT * FROM valid_masks WHERE sample_id=? ORDER BY created_at", (sample_id,))]
        d["ignore_masks"] = [dict(x) for x in self.conn.execute("SELECT * FROM ignore_masks WHERE sample_id=? ORDER BY created_at", (sample_id,))]
        d["events"] = [dict(x) for x in self.conn.execute("SELECT * FROM review_events WHERE sample_id=? ORDER BY timestamp", (sample_id,))]
        # Validation is blind-first: model references stay in the database for
        # provenance but are not returned to the page until the first human
        # decision has been recorded.
        if d.get("blind_first") and d.get("status") in {"unreviewed", "draft"}:
            d["reference_hidden"] = True
            d["reference"] = {}
        return d

    def queue(self, *, project: str | None = None, module: str | None = None, split: str | None = None, status: str | None = None, error_type: str | None = None, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        clauses, params = [], []
        if project: clauses.append("project_id=?"); params.append(project_id(project))
        if module: clauses.append("module=?"); params.append(module)
        if split: clauses.append("split=?"); params.append(split)
        if status: clauses.append("status=?"); params.append(status)
        if error_type: clauses.append("(error_type=? OR negative_type=? )"); params.extend([error_type, error_type])
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        total = self.conn.execute("SELECT count(*) FROM samples" + where, params).fetchone()[0]
        offset = max(0, (int(page) - 1) * min(500, max(1, int(page_size))))
        limit = min(500, max(1, int(page_size)))
        rows = [dict(r) for r in self.conn.execute("SELECT sample_id,project_id,module,split,spatial_block_id,scene_date,chip_path,status,decision,error_type,negative_type,blind_first,model_visible,updated_at FROM samples" + where + " ORDER BY split,spatial_block_id,scene_date,sample_id LIMIT ? OFFSET ?", params + [limit, offset])]
        return {"items": rows, "total": total, "page": int(page), "page_size": limit}

    def progress(self, project: str | None = None) -> dict[str, Any]:
        clauses, params = ("WHERE project_id=?", [project_id(project)]) if project else ("", [])
        rows = [dict(r) for r in self.conn.execute("SELECT project_id,split,status,decision,module,spatial_block_id FROM samples " + clauses, params)]
        result: dict[str, Any] = {"projects": {}}
        for pid in sorted({r["project_id"] for r in rows}):
            rs = [r for r in rows if r["project_id"] == pid]
            blocks = {s: {r["spatial_block_id"] for r in rs if r["split"] == s} for s in ("train", "validation", "holdout")}
            complete_blocks = {s: {r["spatial_block_id"] for r in rs if r["split"] == s and r["decision"] in {"complete_prediction_correct", "prediction_needs_correction", "image_quality_ignore"}} for s in blocks}
            result["projects"][pid] = {"total": len(rs), "reviewed": sum(bool(r["decision"]) for r in rs), "by_split": {s: {"samples": sum(1 for r in rs if r["split"] == s), "blocks": len(blocks[s]), "complete_blocks": len(complete_blocks[s])} for s in blocks}, "decision_counts": {k: sum(1 for r in rs if r["decision"] == k) for k in sorted({r["decision"] for r in rs if r["decision"]})}}
        return result

    def _append_event_jsonl(self, event: Mapping[str, Any]) -> None:
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS_PATH.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            fh.flush(); os.fsync(fh.fileno())

    def record_review(self, sample_id: str, action: str, payload: Mapping[str, Any] | None = None, reviewer: str = "local_reviewer") -> dict[str, Any]:
        payload = dict(payload or {})
        if action not in ALLOWED_ACTIONS: raise ValueError(f"unsupported review action: {action}")
        with _WRITE_LOCK:
            sample = self.conn.execute("SELECT * FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
            if not sample: raise KeyError(sample_id)
            geometry = payload.get("geometry", _json(sample["geometry_json"], None) or _json(sample["source_geometry_json"], None))
            ok, reason = validate_geometry(geometry, allow_empty=action in {"skip", "later", "background_confirm", "complete", "accept", "reject", "delete", "ignore", "save"})
            if action in {"add", "edit", "correct"} and not ok: raise ValueError(reason)
            if action in {"complete", "background_confirm"} and not bool(payload.get("complete_background_confirmed", action == "background_confirm")):
                raise ValueError("complete-background confirmation is required")
            if action in {"reject", "delete"} and not (payload.get("error_type") or payload.get("notes")):
                raise ValueError("deleting/rejecting a model proposal requires an error type or note")
            old_decision = sample["decision"]
            old_geom = _json(sample["geometry_json"], None)
            decision = payload.get("decision")
            status = "draft"
            annotation_status = "draft"
            source = "model_reference"
            if action in {"accept", "background_confirm"}:
                decision = decision or ("complete_prediction_correct" if action == "background_confirm" else "accepted")
                status, annotation_status = "reviewed", "human_verified"
            elif action in {"correct", "edit", "split", "merge", "add"}:
                decision = decision or "prediction_needs_correction"
                status, annotation_status, source = "reviewed", "human_corrected", ("human_drawn" if action == "add" else "model_corrected_by_human")
            elif action in {"propagate", "copy_adjacent"}:
                decision, status, annotation_status, source = "draft_propagated", "draft", "draft", "draft_propagated"
            elif action in {"reject", "delete"}:
                decision, status, annotation_status = "rejected", "reviewed", "rejected"
            elif action == "ignore":
                decision, status, annotation_status = "ignore", "reviewed", "ignore"
            elif action in {"later", "skip"}:
                decision, status = "needs_later_review", "deferred"
            elif action in {"save", "undo", "redo"}:
                decision = decision or old_decision
                status = sample["status"]
                annotation_status = sample["annotation_status"]
            elif action == "complete":
                decision, status, annotation_status = payload.get("decision", "complete_prediction_correct"), "reviewed", "human_verified"
            event_id = uuid.uuid4().hex
            timestamp = now_iso()
            with self.conn:
                self.conn.execute("UPDATE samples SET geometry_json=?,status=?,decision=?,error_type=?,negative_type=?,annotation_status=?,reviewer=?,notes=?,updated_at=?,model_visible=? WHERE sample_id=?", (json.dumps(geometry, ensure_ascii=False), status, decision, payload.get("error_type"), payload.get("negative_type"), annotation_status, reviewer, payload.get("notes"), timestamp, int(payload.get("model_visible", sample["model_visible"])), sample_id))
                self.conn.execute("INSERT OR REPLACE INTO review_decisions(sample_id,project_id,decision,error_type,negative_type,complete_background_confirmed,notes,reviewer,updated_at) VALUES(?,?,?,?,?,?,?,?,?)", (sample_id, sample["project_id"], decision or "draft", payload.get("error_type"), payload.get("negative_type"), int(bool(payload.get("complete_background_confirmed", action in {"background_confirm", "complete"}))), payload.get("notes"), reviewer, timestamp))
                # A complete-background confirmation is a negative/background
                # decision and must not be exported as a positive polygon.
                if action in {"accept", "correct", "edit", "split", "merge", "add"}:
                    aid = uuid.uuid4().hex
                    self.conn.execute("INSERT INTO human_annotations(annotation_id,project_id,sample_id,spatial_block_id,scene_date,class_name,geometry_json,geometry_wkt,source,annotation_status,annotator,reviewer,created_at,updated_at,confidence,notes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (aid, sample["project_id"], sample_id, sample["spatial_block_id"], sample["scene_date"], payload.get("class_name", "building" if sample["project_id"].startswith("nanjing") else "construction_site"), json.dumps(geometry, ensure_ascii=False), None, source, annotation_status, reviewer, reviewer, timestamp, timestamp, payload.get("confidence"), payload.get("notes")))
                ignore_geometry = payload.get("ignore_mask") or (geometry if action == "ignore" else None)
                if ignore_geometry is not None:
                    ok_ignore, reason_ignore = validate_geometry(ignore_geometry, allow_empty=False)
                    if not ok_ignore:
                        raise ValueError(f"invalid ignore mask: {reason_ignore}")
                    self.conn.execute("INSERT INTO ignore_masks(mask_id,project_id,sample_id,geometry_json,status,created_at) VALUES(?,?,?,?,?,?)", (uuid.uuid4().hex, sample["project_id"], sample_id, json.dumps(ignore_geometry, ensure_ascii=False), "human_verified", timestamp))
                valid_geometry = payload.get("valid_mask") or (geometry if action in {"background_confirm", "complete"} else None)
                if valid_geometry is not None:
                    ok_valid, reason_valid = validate_geometry(valid_geometry, allow_empty=False)
                    if not ok_valid:
                        raise ValueError(f"invalid valid mask: {reason_valid}")
                    self.conn.execute("INSERT OR REPLACE INTO valid_masks(mask_id,project_id,sample_id,geometry_json,status,created_at) VALUES(?,?,?,?,?,?)", (stable_hash([sample_id, "valid"])[:32], sample["project_id"], sample_id, json.dumps(valid_geometry, ensure_ascii=False), "complete_background_confirmed", timestamp))
                event = {"event_id": event_id, "timestamp": timestamp, "reviewer": reviewer, "project": sample["project_id"], "module": sample["module"], "sample_id": sample_id, "spatial_block_id": sample["spatial_block_id"], "scene_date": sample["scene_date"], "split": sample["split"], "action": action, "old_value": {"status": sample["status"], "decision": old_decision}, "new_value": {"status": status, "decision": decision, "error_type": payload.get("error_type"), "negative_type": payload.get("negative_type")}, "geometry_before": old_geom, "geometry_after": geometry, "app_version": APP_VERSION, "payload": payload}
                self.conn.execute("INSERT INTO review_events(event_id,timestamp,reviewer,project_id,module,sample_id,spatial_block_id,scene_date,split,action,old_value,new_value,geometry_before,geometry_after,app_version,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (event_id, timestamp, reviewer, sample["project_id"], sample["module"], sample_id, sample["spatial_block_id"], sample["scene_date"], sample["split"], action, json.dumps(event["old_value"]), json.dumps(event["new_value"], ensure_ascii=False), json.dumps(old_geom, ensure_ascii=False), json.dumps(geometry, ensure_ascii=False), APP_VERSION, json.dumps(event, ensure_ascii=False)))
                revision = int(self.setting("database_revision", "0") or 0) + 1
                self.conn.execute("INSERT OR REPLACE INTO app_settings(key,value,updated_at) VALUES(?,?,?)", ("database_revision", str(revision), timestamp))
            self._append_event_jsonl(event)
            return {"success": True, "saved_at": timestamp, "event_id": event_id, "database_revision": revision}

    def backup(self) -> dict[str, Any]:
        with _WRITE_LOCK:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = BACKUP_ROOT / stamp
            target.mkdir(parents=True, exist_ok=True)
            self.conn.execute("PRAGMA wal_checkpoint(FULL)")
            shutil.copy2(self.path, target / self.path.name)
            if EVENTS_PATH.exists(): shutil.copy2(EVENTS_PATH, target / EVENTS_PATH.name)
            receipt = {"success": True, "backup_path": str(target), "created_at": now_iso(), "database_sha256": stable_hash((target / self.path.name).read_bytes()), "event_log_sha256": stable_hash((target / EVENTS_PATH.name).read_bytes()) if (target / EVENTS_PATH.name).exists() else None}
            atomic_write_json(target / "DATABASE_BACKUP_RECEIPT.json", receipt)
            return receipt


def safe_data_path(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    allowed = [DATA_ROOT.resolve(), ROOT.resolve(), NANJING_IMAGE_ROOT.resolve(), TONGZHOU_IMAGE_ROOT.resolve()]
    if not any(candidate == base or base in candidate.parents for base in allowed):
        raise PermissionError("path is outside the approved local data roots")
    return candidate


def write_geojson(path: Path, features: Iterable[Mapping[str, Any]]) -> int:
    fs = list(features)
    atomic_write_json(path, {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": f.get("geometry"), "properties": {k: v for k, v in f.items() if k != "geometry"}} for f in fs]})
    return len(fs)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "\n"); return 0
    cols = sorted({k for r in rows for k in r})
    part = path.with_name(path.name + ".part")
    with part.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols); writer.writeheader(); writer.writerows({k: r.get(k) for k in cols} for r in rows); fh.flush(); os.fsync(fh.fileno())
    os.replace(part, path); return len(rows)


def geojson_to_wkb(geometry: Any) -> bytes | None:
    if isinstance(geometry, str): geometry = _json(geometry, None)
    if not geometry: return None
    typ = geometry.get("type"); order = "<"; out = bytearray([1])
    if typ == "Point":
        out += struct.pack(order + "I", 1); out += struct.pack(order + "dd", *geometry["coordinates"][:2]); return bytes(out)
    if typ == "Polygon":
        out += struct.pack(order + "I", 3); rings = geometry["coordinates"]; out += struct.pack(order + "I", len(rings))
        for ring in rings:
            out += struct.pack(order + "I", len(ring))
            for p in ring: out += struct.pack(order + "dd", float(p[0]), float(p[1]))
        return bytes(out)
    if typ == "MultiPolygon":
        out += struct.pack(order + "I", 6); polys = geometry["coordinates"]; out += struct.pack(order + "I", len(polys))
        for poly in polys:
            out += geojson_to_wkb({"type": "Polygon", "coordinates": poly})
        return bytes(out)
    return None


def write_geopackage(path: Path, features: Iterable[Mapping[str, Any]], table: str, srs_id: int = 4326) -> int:
    fs = list(features); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript("PRAGMA application_id=1196444487; PRAGMA user_version=10200; CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL,srs_id INTEGER PRIMARY KEY,organization TEXT NOT NULL,organization_coordsys_id INTEGER NOT NULL,definition TEXT NOT NULL,description TEXT); CREATE TABLE gpkg_contents (table_name TEXT PRIMARY KEY,data_type TEXT NOT NULL,identifier TEXT,description TEXT,last_change TEXT NOT NULL,min_x REAL,min_y REAL,max_x REAL,max_y REAL,srs_id INTEGER); CREATE TABLE gpkg_geometry_columns (table_name TEXT,column_name TEXT,geometry_type_name TEXT,srs_id INTEGER,z TINYINT,m TINYINT,PRIMARY KEY(table_name,column_name));")
        conn.execute("INSERT INTO gpkg_spatial_ref_sys VALUES(?,?,?,?,?,?)", ("WGS 84", srs_id, "EPSG", srs_id, "GEOGCS[\"WGS 84\"]", ""))
        ident = re.sub(r"[^A-Za-z0-9_]", "_", table)[:50] or "labels"
        cols = sorted({k for f in fs for k in f if k != "geometry"})
        col_sql = ",".join('"%s" TEXT' % c.replace('"','""') for c in cols)
        conn.execute(f'CREATE TABLE "{ident}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom BLOB{", " + col_sql if col_sql else ""})')
        bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
        for f in fs:
            g = f.get("geometry"); b = geometry_bbox(g); bounds[:] = [min(bounds[0], b[0]), min(bounds[1], b[1]), max(bounds[2], b[2]), max(bounds[3], b[3])]
            vals = [json.dumps(g, ensure_ascii=False) if isinstance(v, (dict, list)) else (None if v is None else str(v)) for k, v in [(c, f.get(c)) for c in cols]]
            if cols:
                quoted_cols = ",".join('"' + c.replace('"', '""') + '"' for c in cols)
                placeholders = ",".join("?" for _ in cols)
                conn.execute(f'INSERT INTO "{ident}"(geom,{quoted_cols}) VALUES(?,{placeholders})', [g and sqlite3.Binary(geojson_to_wkb(g) or b"")] + vals)
            else:
                conn.execute(f'INSERT INTO "{ident}"(geom) VALUES(?)', (g and sqlite3.Binary(geojson_to_wkb(g) or b""),))
        if not fs: bounds = [None, None, None, None]
        conn.execute("INSERT INTO gpkg_contents VALUES(?,?,?,?,?,?,?,?,?,?)", (ident, "features", ident, "URRS Lite human verified labels", now_iso(), *bounds, srs_id))
        conn.execute("INSERT INTO gpkg_geometry_columns VALUES(?,?,?,?,?,?)", (ident, "geom", "GEOMETRY", srs_id, 0, 0)); conn.commit()
    finally: conn.close()
    return len(fs)


def rebuild_review_decisions_from_jsonl(db: ReviewDB) -> dict[str, Any]:
    """Repair missing decision rows from the append-only log without deleting history."""
    if not EVENTS_PATH.exists(): return {"events": 0, "repaired": 0}
    repaired = events = 0
    for line in EVENTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        events += 1
        try: event = json.loads(line)
        except json.JSONDecodeError: continue
        sample_id = event.get("sample_id"); new_value = event.get("new_value") or {}
        if not sample_id: continue
        exists = db.conn.execute("SELECT 1 FROM review_decisions WHERE sample_id=?", (sample_id,)).fetchone()
        if not exists and new_value.get("decision"):
            s = db.conn.execute("SELECT project_id FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
            if s:
                db.conn.execute("INSERT OR IGNORE INTO review_decisions(sample_id,project_id,decision,error_type,negative_type,complete_background_confirmed,notes,reviewer,updated_at) VALUES(?,?,?,?,?,?,?,?,?)", (sample_id, s[0], new_value["decision"], new_value.get("error_type"), new_value.get("negative_type"), 0, None, event.get("reviewer"), event.get("timestamp", now_iso()))); repaired += 1
    db.conn.commit(); return {"events": events, "repaired": repaired}
