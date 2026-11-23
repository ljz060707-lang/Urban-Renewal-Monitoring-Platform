"""XIAN-R5 single-date delivery: frozen constants and small helpers.

This snapshot is the ONLY writer of new source code in this run.  The delivered
application (APPLICATION_V1_20261027_01) and every historical artefact are opened
read-only; the B0/E3/V2 runtimes are imported, never edited.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- environment
APP = Path(r'F:\xian\XIAN_R5_APPLICATION\APPLICATION_V1_20261027_01')
APP_SRC = APP / 'SOURCE_CODE' / 'src'
APP_ENTRY = APP / 'SOURCE_CODE' / 'tools' / 'run_xian_r5_integration.py'
ROUTING = APP / 'MODEL_ROUTING_MANIFEST.json'
PYTHON = Path(sys.executable)
# The RSBuilding code tree was relocated D:/E: -> F: during this session.  448/448
# frozen runtime bindings were re-hashed at the new root and matched exactly.
RSB_REPO = Path(r'F:\Data Set\RSBuilding\repo')
RUN_ROOT = Path(__file__).resolve().parents[2]
WORK = RUN_ROOT / 'work'
STATE = RUN_ROOT / 'state'
LOGS = RUN_ROOT / 'logs'
RUNS = RUN_ROOT / 'runs'
SINGLE = RUN_ROOT / 'single_date'
REF2019 = RUN_ROOT / 'reference_2019'
MONITOR = RUN_ROOT / 'monitor'

for _p in (APP_SRC, RSB_REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault('GDAL_PAM_ENABLED', 'NO')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

# ---------------------------------------------------------------- frozen grid
TEMPLATE = Path(r'F:\xian\10m底图\Xian_2018_TrueColor.tif')
FINE_SUBDIVISION = 10          # application V1 nested fine grid (10 m / 10 = 1 m)
METRIC_CRS = 'EPSG:32649'

# ---------------------------------------------------------------- source queue
# Every epoch is one registered GEID task whose mosaic is an actual city mosaic.
# epoch_label is the GEID task label, never an observation date.
EPOCHS = [
    dict(
        epoch_id='xa180429',
        epoch_label='xa180429',
        nominal_input_date='2018-04-29',
        actual_observation_date='2018-04-29',
        date_source='GEID_TILE_LOG',
        date_status='SINGLE_ACTUAL_DATE',
        dominant_share_pct=99.997905,
        mixing_severity='none_or_negligible',
        assumption='label equals recovered capture date',
        mosaic=r'F:\xian\亚米市区底图\xa180429_combined\xa180429_zoom_20.tif',
        mosaic_component=r'F:\xian\亚米市区底图\xa180429',
        date_table=r'F:\xian\亚米市区底图\date_audit\tables\tile_actual_date.parquet',
        date_audit=r'F:\xian\亚米市区底图\date_audit\tables\date_summary_by_epoch.csv',
        tile_origin=(420534, 311895),
        verified_tile_count=238970,
        missing_tile_count=0,
    ),
    dict(
        epoch_id='xa190101',
        epoch_label='xa190101',
        nominal_input_date='2019-01-01',
        actual_observation_date='2018-11-01',
        date_source='GEID_TILE_LOG',
        date_status='MIXED_2_ACTUAL_DATES_DOMINANT_REPORTED',
        dominant_share_pct=65.290048,
        mixing_severity='severe',
        assumption='dominant date by area; 34.71% of the mosaic is 2018-11-08 and is kept per object',
        mosaic=r'F:\xian\亚米市区底图\xa190101_combined\xa190101_zoom_20.tif',
        mosaic_component=r'F:\xian\亚米市区底图\xa190101',
        date_table=r'F:\xian\亚米市区底图\date_audit\tables\tile_actual_date.parquet',
        date_audit=r'F:\xian\亚米市区底图\date_audit\tables\date_summary_by_epoch.csv',
        tile_origin=(420534, 311895),
        verified_tile_count=238975,
        missing_tile_count=0,
    ),
    dict(
        epoch_id='xa201231',
        epoch_label='xa201231',
        nominal_input_date='2020-12-31',
        actual_observation_date='2019-12-10',
        date_source='GEID_TILE_LOG',
        date_status='MIXED_8_ACTUAL_DATES_DOMINANT_REPORTED',
        dominant_share_pct=31.301862,
        mixing_severity='severe',
        assumption='dominant date by area; 68.6% of the mosaic carries 7 other dates and 256 tiles are DATE_UNKNOWN',
        mosaic=r'F:\xian\亚米市区底图\xa201231_combined\xa201231_zoom_20.tif',
        mosaic_component=r'F:\xian\亚米市区底图\xa201231',
        date_table=r'F:\xian\亚米市区底图\date_audit\tables\tile_actual_date.parquet',
        date_audit=r'F:\xian\亚米市区底图\date_audit\tables\date_summary_by_epoch.csv',
        tile_origin=(420534, 311895),
        verified_tile_count=238719,
        missing_tile_count=256,
    ),
]

# ---------------------------------------------------------------- GT reference
GT_SHAPEFILE = Path(r'F:\xian\西安建筑轮廓数据\西安.shp')
GT_ROLE = 'DESIGNATED_GT_REFERENCE_DERIVED'

# ---------------------------------------------------------------- block policy
# Native mosaic grid: 154880 x 101120 px at 2.6822e-6 deg (0.247 m x 0.297 m).
# Core blocks are a multiple of the 192 px tile stride so the frozen 256/192 tile
# lattice is globally identical in every block.  A guard band keeps every object
# whose centroid is owned by this block fully inside the analysed window.
CORE = 16384
GUARD = 768
EDGE_MARGIN = 256            # objects closer than this to the read border are flagged as possibly truncated
TILE = 256
STRIDE = 192
BATCH = 8                    # documented batched scheduler; equivalence test in evidence/
B0_THRESHOLD = 0.5

# area / fraction
PIXEL_AREA_10M = 100.0       # EPSG:32649 10 m cell
AREA_TOL_M2 = 1e-3           # frozen tolerance for vector/fraction agreement


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(b):
    if isinstance(b, str):
        b = b.encode('utf-8')
    return hashlib.sha256(b).hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return json.load(f)


def atomic_write_csv(path, rows, columns=None):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        columns = columns or []
        tmp = path.with_suffix(path.suffix + '.tmp')
        with tmp.open('w', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow(columns)
        os.replace(tmp, path)
        return
    columns = columns or list(rows[0].keys())
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def log(message, stream=None):
    line = '%s %s' % (time.strftime('%Y-%m-%dT%H:%M:%S'), message)
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / 'production.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')


class StateFile:
    """Single writer for the per-work-unit state documents."""

    def __init__(self, name):
        self.path = STATE / name

    def update(self, **fields):
        try:
            cur = read_json(self.path)
        except Exception:
            cur = {}
        cur.update(fields)
        cur['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        atomic_write_json(self.path, cur)
        return cur
