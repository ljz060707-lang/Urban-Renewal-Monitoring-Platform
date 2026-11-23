from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy import ndimage


CASE_DATES = [
    "2019-11-24",
    "2020-04-06",
    "2020-08-03",
    "2021-04-14",
    "2021-12-19",
    "2022-02-10",
]
STATES = [1, 1, 0, 0, 1, 1]
STATE_LABELS = ["旧建筑存在", "旧建筑存在", "拆除后空缺", "拆除后空缺", "新建筑出现", "新建筑出现"]
OLD_TRACK = "LF_007556"
NEW_TRACK = "LF_009322"
EPISODE = "redevelopment_LF_007556_LF_009322"
CENTROID = (116.3631155, 39.864891)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def percentile_rgb(rgb: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(rgb, dtype=np.float32)
    finite = np.isfinite(arr)
    vals = arr[finite]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32), 0.0, 0.0
    lo, hi = np.percentile(vals, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return out, float(lo), float(hi)


def read_rgb(source: Path, center_lon: float, center_lat: float, width: int, height: int):
    with rasterio.open(source) as ds:
        cx, cy = (~ds.transform) * (center_lon, center_lat)
        x0 = max(0, min(ds.width - width, int(round(cx - width / 2))))
        y0 = max(0, min(ds.height - height, int(round(cy - height / 2))))
        win = rasterio.windows.Window(x0, y0, width, height)
        bands = min(3, ds.count)
        raw = ds.read(list(range(1, bands + 1)), window=win)
        if bands == 1:
            raw = np.repeat(raw, 3, axis=0)
        elif bands == 2:
            raw = np.concatenate([raw, raw[1:2]], axis=0)
        rgb, lo, hi = percentile_rgb(np.moveaxis(raw, 0, -1))
        return rgb, (x0, y0, width, height), ds.transform, ds.crs, lo, hi


def component_in_window(mask_path: Path, center_lon: float, center_lat: float, window, require_present: bool):
    x0, y0, width, height = window
    with rasterio.open(mask_path) as ds:
        cx, cy = (~ds.transform) * (center_lon, center_lat)
        local_x = int(round(cx)) - x0
        local_y = int(round(cy)) - y0
        arr = ds.read(1, window=rasterio.windows.Window(x0, y0, width, height))
    binary = arr == 1
    if not require_present:
        return np.zeros_like(binary, dtype=bool), {"mask_present": False, "component_pixels": 0}
    labels, n = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    label_id = 0
    if 0 <= local_x < width and 0 <= local_y < height:
        label_id = int(labels[local_y, local_x])
    if label_id == 0:
        radius = 55
        xa, xb = max(0, local_x - radius), min(width, local_x + radius + 1)
        ya, yb = max(0, local_y - radius), min(height, local_y + radius + 1)
        sub = labels[ya:yb, xa:xb]
        candidates, counts = np.unique(sub[sub > 0], return_counts=True)
        if candidates.size:
            label_id = int(candidates[np.argmax(counts)])
    component = labels == label_id if label_id else np.zeros_like(binary, dtype=bool)
    yy, xx = np.where(component)
    stats = {
        "mask_present": bool(component.any()),
        "component_pixels": int(component.sum()),
        "component_bbox_px": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())] if xx.size else None,
        "component_label": int(label_id),
    }
    return component, stats


def add_component(ax, component: np.ndarray, color: str, label: str, alpha: float = 0.16):
    if not component.any():
        return
    ax.contourf(component.astype(np.float32), levels=[0.5, 1.5], colors=[color], alpha=alpha, origin="upper", extent=[0, component.shape[1], component.shape[0], 0])
    ax.contour(component.astype(np.float32), levels=[0.5], colors=[color], linewidths=2.1, origin="upper", extent=[0, component.shape[1], component.shape[0], 0])


def configure_fonts():
    try:
        fp = FontProperties(family="Microsoft YaHei")
        return fp
    except Exception:
        return FontProperties(family="DejaVu Sans")


def build_figure(records: list[dict], out_png: Path, out_pdf: Path, simple: bool = False):
    fp = configure_fonts()
    if simple:
        fig = plt.figure(figsize=(18, 5.8), dpi=240, facecolor="white")
        gs = GridSpec(2, 6, figure=fig, height_ratios=[5.2, 0.7], hspace=0.25, wspace=0.06)
        title_y = 0.985
    else:
        fig = plt.figure(figsize=(22, 10.8), dpi=240, facecolor="white")
        gs = GridSpec(3, 6, figure=fig, height_ratios=[7.7, 1.05, 1.25], hspace=0.28, wspace=0.08)
        title_y = 0.985
    fig.suptitle("北京南站重建案例（110011）", fontsize=24 if not simple else 19, fontweight="bold", y=title_y, fontproperties=fp)
    if not simple:
        fig.text(0.5, 0.948, "真实卫星底图 + 候选轨迹证据；candidate evidence, not Ground Truth", ha="center", va="center", fontsize=12.5, color="#4b5563", fontproperties=fp)
    for i, rec in enumerate(records):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(rec["rgb"], origin="upper")
        if rec["track_role"] == "old":
            add_component(ax, rec["component"], "#e4572e", "old")
        elif rec["track_role"] == "new":
            add_component(ax, rec["component"], "#007c91", "new")
        # Same target window across every date: this is the spatial anchor, not a derived GT box.
        x0, y0, w, h = rec["window"]
        cx, cy = rec["center_px"]
        box_w, box_h = 150, 105
        ax.add_patch(Rectangle((cx - box_w / 2, cy - box_h / 2), box_w, box_h, fill=False, edgecolor="#f2c94c", linewidth=1.25, alpha=0.85))
        ax.set_xlim(0, rec["rgb"].shape[1])
        ax.set_ylim(rec["rgb"].shape[0], 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(rec["date"], fontsize=11 if not simple else 9.2, pad=5, fontproperties=fp)
        if not simple:
            state_text = f"{rec['state']}  ·  {rec['state_label']}"
            ax.text(0.5, -0.08, state_text, transform=ax.transAxes, ha="center", va="top", fontsize=10.2, color="#1f2937", fontproperties=fp)
            ax.text(0.03, 0.04, "OLD" if rec["track_role"] == "old" else "NEW" if rec["track_role"] == "new" else "GAP", transform=ax.transAxes, ha="left", va="top", fontsize=9.2, fontweight="bold", color="#e4572e" if rec["track_role"] == "old" else "#007c91" if rec["track_role"] == "new" else "#6b7280", bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.86), fontproperties=fp)
    if simple:
        ax = fig.add_subplot(gs[1, :])
        ax.axis("off")
        xs = np.arange(6)
        for i, val in enumerate(STATES):
            ax.text(i, 0.5, str(val), ha="center", va="center", fontsize=16, fontweight="bold", color="#1f2937", fontproperties=fp)
            if i < 5:
                ax.plot([i + 0.12, i + 0.88], [0.5, 0.5], color="#9ca3af", linewidth=1.5)
        ax.set_xlim(-0.25, 5.25); ax.set_ylim(0, 1)
    else:
        ax_seq = fig.add_subplot(gs[1, :])
        ax_seq.axis("off")
        xs = np.arange(6)
        for i, val in enumerate(STATES):
            color = "#e4572e" if i < 2 else "#007c91" if i >= 4 else "#6b7280"
            ax_seq.text(i, 0.58, str(val), ha="center", va="center", fontsize=22, fontweight="bold", color=color, fontproperties=fp)
            if i < 5:
                ax_seq.plot([i + 0.12, i + 0.88], [0.58, 0.58], color="#9ca3af", linewidth=2.0)
        ax_seq.text(3.0, 0.09, "判定：Redevelopment candidate  ·  Episode type: redevelopment", ha="center", va="center", fontsize=12, color="#374151", fontproperties=fp)
        ax_seq.set_xlim(-0.25, 5.25); ax_seq.set_ylim(0, 1)
        ax_sum = fig.add_subplot(gs[2, :])
        ax_sum.axis("off")
        ax_sum.text(0.01, 0.68, f"Old track: {OLD_TRACK}    →    New track: {NEW_TRACK}    ·    Episode: {EPISODE}", fontsize=11.5, color="#111827", fontproperties=fp)
        ax_sum.text(0.01, 0.34, "Window: 2019-11-24 → 2022-02-10    ·    Confidence: 0.99786    ·    Status: candidate_not_ground_truth", fontsize=11.2, color="#374151", fontproperties=fp)
        ax_sum.text(0.01, 0.02, "基于北京南站 acceptance package 的真实底图与轨迹候选绘制；用于展示重建候选过程，不代表人工 Ground Truth。", fontsize=10.4, color="#6b7280", fontproperties=fp)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--acceptance-root", default=r"E:\URRS Output\beijing_nanzhan_tracker_acceptance_final_postfreeze_20260812_020404")
    ap.add_argument("--static-root", default=r"E:\URRS Output\beijing_nanzhan_tracker_acceptance_20260811_235827\06_static_building")
    ap.add_argument("--source-root", default=r"E:\Data Set\Satellite\beijing_nanzhan")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    acceptance = Path(args.acceptance_root)
    static_root = Path(args.static_root)
    source_root = Path(args.source_root)
    manifest = json.loads((acceptance / "01_data_registry" / "DATA_PROVENANCE.json").read_text(encoding="utf-8"))
    case_receipt = json.loads((acceptance / "10_case_mining" / "CASE_SELECTION_RECEIPT.json").read_text(encoding="utf-8"))
    selected = case_receipt["selected_cases"][0]
    if selected["old_track_id"] != OLD_TRACK or selected["new_track_id"] != NEW_TRACK or selected["sequence"] != "110011":
        raise RuntimeError("Existing case receipt does not match the requested real 110011 candidate")
    role_by_date = {CASE_DATES[0]: "old", CASE_DATES[1]: "old", CASE_DATES[2]: "gap", CASE_DATES[3]: "gap", CASE_DATES[4]: "new", CASE_DATES[5]: "new"}
    rgb_paths = {}
    records = []
    for date, state, label in zip(CASE_DATES, STATES, STATE_LABELS):
        scene = manifest["scenes"].get(date)
        if not scene or not scene.get("source_exists"):
            raise FileNotFoundError(f"Missing frozen source scene for {date}")
        source = Path(scene["selected_source"])
        if not source.exists():
            raise FileNotFoundError(str(source))
        rgb_paths[date] = source
    # A single common window is calculated from the frozen track centroid and reused verbatim.
    with rasterio.open(rgb_paths[CASE_DATES[0]]) as ds:
        cx, cy = (~ds.transform) * CENTROID
        width, height = 720, 500
        x0 = max(0, min(ds.width - width, int(round(cx - width / 2))))
        y0 = max(0, min(ds.height - height, int(round(cy - height / 2))))
        common_window = (x0, y0, width, height)
    mask_stats = {}
    for date, state, label in zip(CASE_DATES, STATES, STATE_LABELS):
        source = rgb_paths[date]
        rgb, window, transform, crs, lo, hi = read_rgb(source, *CENTROID, common_window[2], common_window[3])
        # read_rgb computes a centered window; force the frozen common window to avoid sub-pixel drift.
        with rasterio.open(source) as ds:
            raw = ds.read([1, 2, 3], window=rasterio.windows.Window(*common_window))
            rgb, lo, hi = percentile_rgb(np.moveaxis(raw, 0, -1))
            cx2, cy2 = (~ds.transform) * CENTROID
            center_local = (float(cx2 - common_window[0]), float(cy2 - common_window[1]))
        mask_path = static_root / date / "building_mask.tif"
        if not mask_path.exists():
            raise FileNotFoundError(str(mask_path))
        component, stats = component_in_window(mask_path, *CENTROID, common_window, role_by_date[date] != "gap")
        mask_stats[date] = stats
        records.append({"date": date, "state": state, "state_label": label, "track_role": role_by_date[date], "rgb": rgb, "window": common_window, "center_px": center_local, "component": component, "source": source, "mask_path": mask_path, "stretch_lo": lo, "stretch_hi": hi, "crs": str(crs)})
    main_png = out / "FIG_redevelopment_110011_main.png"
    main_pdf = out / "FIG_redevelopment_110011_main.pdf"
    simple_png = out / "FIG_redevelopment_110011_simple.png"
    build_figure(records, main_png, main_pdf, simple=False)
    build_figure(records, simple_png, out / "FIG_redevelopment_110011_simple.pdf", simple=True)
    source_entries = []
    for rec in records:
        source_entries.append({"date": rec["date"], "source_path": str(rec["source"]), "source_sha256": sha256(rec["source"]), "mask_path": str(rec["mask_path"]), "mask_sha256": sha256(rec["mask_path"]), "window_px": list(common_window), "percentile_stretch": [rec["stretch_lo"], rec["stretch_hi"]], "crs": rec["crs"], "track_role": rec["track_role"], "state": rec["state"], "component_stats": mask_stats[rec["date"]]})
    receipt = {"created_at": datetime.now(timezone.utc).isoformat(), "case": {"sequence": "110011", "old_track_id": OLD_TRACK, "new_track_id": NEW_TRACK, "episode_id": EPISODE, "dates": CASE_DATES, "confidence": selected["confidence"], "scientific_status": "candidate_not_ground_truth"}, "inputs": {"acceptance_root": str(acceptance), "case_selection_receipt": str(acceptance / "10_case_mining" / "CASE_SELECTION_RECEIPT.json"), "static_b0_cache_root": str(static_root), "source_root": str(source_root), "source_scenes": source_entries}, "processing": {"read_only_reuse": True, "local_case_crop_only": True, "common_window_px": list(common_window), "rgb_percentile_stretch": "2-98 per crop", "mask_interpretation": "selected frozen B0 component shown only on dates where the candidate track is present; gap dates intentionally have no candidate outline", "training_rerun": False, "tracker_rerun": False, "full_pipeline_rerun": False, "test_reads": {"WHU": 0, "SECOND": 0, "Test": 0}, "raw_imagery_modified": False, "protected_models_modified": False}, "outputs": {"main_png": str(main_png), "main_pdf": str(main_pdf), "simple_png": str(simple_png), "main_png_sha256": sha256(main_png), "main_pdf_sha256": sha256(main_pdf), "simple_png_sha256": sha256(simple_png)}}
    json_dump(out / "REDRAW_RECEIPT.json", receipt)
    report = f"""# 北京南站 110011 重建案例重绘\n\n- **案例**：`{OLD_TRACK}` → `{NEW_TRACK}`，episode `{EPISODE}`\n- **真实序列**：`1 → 1 → 0 → 0 → 1 → 1`\n- **日期**：{', '.join(CASE_DATES)}\n- **图层**：使用 `DATA_PROVENANCE.json` 指定的六期真实南站卫星影像；使用同一冻结 B0 静态 mask 缓存，仅在轨迹存在的四个日期叠加对应局部组件轮廓。\n- **裁剪**：统一窗口像素坐标 `{list(common_window)}`，每期保持同一范围与比例。RGB 使用每期局部有效像素 2–98 百分位拉伸，未改动源影像。\n- **科学状态**：`candidate_not_ground_truth`。该图展示模型轨迹候选与建筑轮廓证据，不是人工 Ground Truth，也不是已确认的城市更新事实。\n- **训练/推理**：没有训练、没有重跑 tracker 或南站全流程，只做单案例局部读取、裁剪和排版。\n- **安全**：WHU Test = 0，SECOND Test = 0；没有读取 Test/Holdout；B0、Full20、Registry 和历史 acceptance 结果未修改。\n\n输出：`FIG_redevelopment_110011_main.png`、`FIG_redevelopment_110011_main.pdf`，以及小尺寸备用图 `FIG_redevelopment_110011_simple.png`。\n"""
    (out / "REDRAW_REPORT.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
