import os
import json
import math
import requests
import rasterio
import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds

# =========================================================
# 1. 基本配置
# =========================================================

# 如果这个地址不通，你只需要改这里
# 常见是 /stac 或 /stac/
STAC_ROOT = "https://data.inpe.br/stac"
SEARCH_URL = STAC_ROOT.rstrip("/") + "/search"

COLLECTION = "CB4A-WPM-PCA-FUSED-1"

# 北京亦庄 AOI（可自行微调）
AOI_BBOX_WGS84 = [116.42, 39.72, 116.68, 39.90]  # minx, miny, maxx, maxy

YEARS = [2023, 2024, 2025, 2026]

# 优先同季节：4-6月
PREFERRED_MONTHS = [4, 5, 6]
TARGET_MONTH = 5
TARGET_DAY = 15

# 输出目录
OUT_ROOT = Path(r"D:\Data Set\Satellite\beijing_yizhuang")
RAW_CLIP_DIR = OUT_ROOT / "clipped_raw"
ALIGNED_DIR = OUT_ROOT / "aligned"
META_DIR = OUT_ROOT / "metadata"

RAW_CLIP_DIR.mkdir(parents=True, exist_ok=True)
ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. 工具函数
# =========================================================

def post_stac_search(payload):
    r = requests.post(SEARCH_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def parse_item_datetime(item):
    dt_str = item.get("properties", {}).get("datetime")
    if not dt_str:
        return None
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

def pick_asset_href(item):
    """
    尽量找到真正的 GeoTIFF / COG 资产
    """
    assets = item.get("assets", {})

    preferred_keys = [
        "data", "Data COG", "COG", "cog", "visual", "image", "analytic"
    ]

    # 优先按名字找
    for key in preferred_keys:
        if key in assets and "href" in assets[key]:
            href = assets[key]["href"]
            if href.lower().endswith((".tif", ".tiff")) or "http" in href:
                return href, key

    # 再遍历所有 assets
    for key, val in assets.items():
        href = val.get("href")
        if not href:
            continue
        if href.lower().endswith((".tif", ".tiff")):
            return href, key

    return None, None

def month_score(dt):
    if dt is None:
        return (999, 999, 999)
    month_penalty = min(abs(dt.month - m) for m in PREFERRED_MONTHS)
    center_penalty = abs(dt.month - TARGET_MONTH)
    day_penalty = abs(dt.day - TARGET_DAY)
    return (month_penalty, center_penalty, day_penalty)

def cloud_score(item):
    props = item.get("properties", {})
    # 有些数据不一定有云量
    for k in ["eo:cloud_cover", "cloud_cover", "cloud", "CLOUD_COVER"]:
        if k in props and props[k] is not None:
            try:
                return float(props[k])
            except:
                pass
    return 50.0  # 默认中等

def rank_items(items):
    ranked = []
    for item in items:
        dt = parse_item_datetime(item)
        ranked.append((month_score(dt), cloud_score(item), item))
    ranked.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in ranked]

def search_best_item_for_year(year):
    start = f"{year}-01-01T00:00:00Z"
    end = f"{year}-12-31T23:59:59Z"

    payload = {
        "collections": [COLLECTION],
        "bbox": AOI_BBOX_WGS84,
        "datetime": f"{start}/{end}",
        "limit": 200
    }

    data = post_stac_search(payload)
    items = data.get("features", [])

    if not items:
        return None

    ranked = rank_items(items)

    # 优先选能找到 tif 资产的
    for item in ranked:
        href, asset_key = pick_asset_href(item)
        if href:
            item["_selected_asset_key"] = asset_key
            item["_selected_href"] = href
            return item

    return None

def clip_remote_cog_to_aoi(href, out_path):
    """
    从远程 COG 直接裁剪 AOI
    """
    with rasterio.open(href) as src:
        src_crs = src.crs
        if src_crs is None:
            raise RuntimeError(f"Source CRS is missing: {href}")

        # 把 AOI 从 WGS84 转到影像 CRS
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src_crs, *AOI_BBOX_WGS84, densify_pts=21
        )

        # 防止超范围
        left = max(left, src.bounds.left)
        right = min(right, src.bounds.right)
        bottom = max(bottom, src.bounds.bottom)
        top = min(top, src.bounds.top)

        if not (left < right and bottom < top):
            raise RuntimeError("AOI does not intersect image bounds.")

        window = from_bounds(left, bottom, right, top, src.transform)
        window = window.round_offsets().round_lengths()

        data = src.read(window=window)
        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update({
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform,
            "compress": "lzw"
        })

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

def align_to_template(src_path, template_path, out_path):
    """
    把 src 重投影/重采样到 template 的同一网格
    """
    with rasterio.open(template_path) as tmp:
        template_profile = tmp.profile.copy()
        template_transform = tmp.transform
        template_crs = tmp.crs
        template_height = tmp.height
        template_width = tmp.width
        template_count = tmp.count
        template_dtype = tmp.dtypes[0]

    with rasterio.open(src_path) as src:
        out_profile = template_profile.copy()
        out_profile.update({
            "count": src.count,
            "dtype": src.dtypes[0],
            "compress": "lzw"
        })

        with rasterio.open(out_path, "w", **out_profile) as dst:
            for b in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=rasterio.band(dst, b),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=template_transform,
                    dst_crs=template_crs,
                    resampling=Resampling.bilinear
                )

def save_metadata(rows, out_csv, out_json):
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# =========================================================
# 3. 主流程
# =========================================================

def main():
    selected = []
    print("Searching STAC items...")

    for year in YEARS:
        print(f"\n[Year {year}] searching...")
        item = search_best_item_for_year(year)
        if item is None:
            print(f"  No item found for {year}")
            continue

        dt = parse_item_datetime(item)
        href = item["_selected_href"]
        asset_key = item["_selected_asset_key"]

        print(f"  Selected item: {item.get('id')}")
        print(f"  Datetime     : {dt}")
        print(f"  Asset key    : {asset_key}")
        print(f"  Href         : {href}")

        raw_out = RAW_CLIP_DIR / f"{year}_clip.tif"
        print(f"  Clipping to  : {raw_out}")
        clip_remote_cog_to_aoi(href, raw_out)

        selected.append({
            "year": year,
            "item_id": item.get("id"),
            "datetime": dt.isoformat() if dt else None,
            "asset_key": asset_key,
            "href": href,
            "raw_clip_path": str(raw_out),
            "cloud": cloud_score(item)
        })

    if len(selected) == 0:
        raise RuntimeError("No yearly images were selected. Check STAC root or AOI.")

    selected = sorted(selected, key=lambda x: x["year"])

    # 保存元数据
    save_metadata(
        selected,
        META_DIR / "selected_items.csv",
        META_DIR / "selected_items.json"
    )

    # 用第一期做模板
    template_path = Path(selected[0]["raw_clip_path"])
    print(f"\nTemplate image: {template_path}")

    t_names = ["T1.tif", "T2.tif", "T3.tif", "T4.tif"]

    for i, rec in enumerate(selected):
        src_path = Path(rec["raw_clip_path"])
        out_path = ALIGNED_DIR / t_names[i]

        if i == 0:
            # 第一张直接复制为模板
            print(f"Writing template as {out_path}")
            with rasterio.open(src_path) as src:
                profile = src.profile.copy()
                data = src.read()
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(data)
        else:
            print(f"Aligning {src_path.name} -> {out_path.name}")
            align_to_template(src_path, template_path, out_path)

    print("\nDone.")
    print(f"Aligned outputs saved to: {ALIGNED_DIR}")
    for p in sorted(ALIGNED_DIR.glob("T*.tif")):
        print(" -", p)

    print("\nMetadata saved to:")
    print(" -", META_DIR / "selected_items.csv")
    print(" -", META_DIR / "selected_items.json")


if __name__ == "__main__":
    main()