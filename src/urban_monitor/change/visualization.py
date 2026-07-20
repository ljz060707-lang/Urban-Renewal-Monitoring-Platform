from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def _chart(path: Path, series: list[tuple[str, list[float], str]], title: str):
    width, height = 900, 420
    canvas = Image.new("RGB", (width, height), "#11151b"); draw = ImageDraw.Draw(canvas)
    draw.text((28, 18), title, fill="white")
    all_values = [value for _, values, _ in series for value in values if np.isfinite(value)]
    if not all_values:
        draw.text((28, 70), "No measured values available", fill="#9aa7b5")
    else:
        low, high = min(all_values), max(all_values)
        if high == low: high = low + 1
        for name, values, color in series:
            points = []
            for index, value in enumerate(values):
                x = 50 + index * (width - 90) / max(1, len(values) - 1)
                y = height - 45 - (value - low) * (height - 100) / (high - low)
                points.append((x, y))
            if len(points) > 1: draw.line(points, fill=color, width=3)
            elif points: draw.ellipse((points[0][0]-3, points[0][1]-3, points[0][0]+3, points[0][1]+3), fill=color)
            draw.text((width - 180, 20 + 20 * series.index((name, values, color))), name, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)


def build_training_visualizations(run: str | Path) -> Path:
    run = Path(run); root = run / "visualizations"; root.mkdir(parents=True, exist_ok=True)
    logs = []
    for path in sorted((run / "metrics").glob("train_log_*.csv")):
        frame = pd.read_csv(path); frame["stage"] = path.stem.removeprefix("train_log_"); logs.append(frame)
    frame = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    def values(column): return frame[column].astype(float).tolist() if column in frame else []
    charts = {
        "loss_curves.png": [("train", values("train_loss"), "#54c2ff"), ("validation", values("val_loss"), "#ffb454")],
        "learning_rate_curve.png": [("learning rate", values("learning_rate"), "#8ae66e")],
        "binary_metrics_curve.png": [("binary F1", values("val_binary_f1"), "#a78bfa")],
        "new_metrics_curve.png": [], "demolition_metrics_curve.png": [], "semantic_miou_curve.png": [],
        "gpu_memory_curve.png": [("GPU bytes", values("max_gpu_memory_bytes"), "#ff6b87")],
        "dataset_sampling_curve.png": [], "class_distribution.png": [], "threshold_sweep.png": [],
    }
    for filename, series in charts.items(): _chart(root / filename, series, filename.removesuffix(".png").replace("_", " ").title())
    combined_path = run / "metrics" / "combined.json"
    combined = json.loads(combined_path.read_text(encoding="utf-8")) if combined_path.exists() else {}
    best_index = int(frame["val_binary_f1"].astype(float).idxmax()) if not frame.empty and "val_binary_f1" in frame else None
    best = frame.loc[best_index].to_dict() if best_index is not None else {}
    rows = "".join(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>" for key, value in best.items())
    dataset_cards = []
    for path in sorted((run / "metrics").glob("*.json")):
        if path.name.startswith("train_log_"): continue
        payload = json.loads(path.read_text(encoding="utf-8")); binary = payload.get("binary", {})
        dataset_cards.append(f"<article><h2>{html.escape(path.stem)}</h2><p>F1 {binary.get('f1', 'N/A')}</p><p>IoU {binary.get('iou', 'N/A')}</p></article>")
    dashboard = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Change Teacher Training Dashboard</title><style>body{{font:15px system-ui;background:#0c1016;color:#edf2f7;margin:32px}}main{{max-width:1100px;margin:auto}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}article,table{{background:#151b24;border:1px solid #273140;border-radius:12px;padding:18px}}th,td{{padding:7px;text-align:left}}img{{max-width:100%;border-radius:10px}}a{{color:#63c5ff}}</style>
<main><h1>变化教师训练仪表板</h1><p>静态离线页面；所有数值来自本次运行文件。</p><div class='grid'>{''.join(dataset_cards)}</div><h2>Best epoch</h2><table>{rows}</table><h2>Curves</h2><img src='visualizations/loss_curves.png'><img src='visualizations/binary_metrics_curve.png'><img src='visualizations/gpu_memory_curve.png'></main></html>"""
    path = run / "training_dashboard.html"; path.write_text(dashboard, encoding="utf-8")
    return path
