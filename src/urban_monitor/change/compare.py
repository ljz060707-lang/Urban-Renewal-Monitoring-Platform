from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from PIL import Image, ImageDraw


def compare_runs(baseline: str | Path, candidate: str | Path, baseline_hash: str | None = None, candidate_hash: str | None = None) -> Path:
    baseline, candidate = Path(baseline), Path(candidate)
    def best(run: Path, checkpoint_hash: str | None):
        path = run / "metrics" / "threshold_scans" / checkpoint_hash / "threshold_scan.csv" if checkpoint_hash else run / "metrics" / "threshold_scan.csv"
        if not path.is_file(): raise FileNotFoundError(f"Run threshold scan first: {path}")
        frame = pd.read_csv(path); return frame.loc[frame.groupby("head")["f1"].idxmax()].set_index("head")
    left, right = best(baseline, baseline_hash), best(candidate, candidate_hash)
    rows = []
    for head in sorted(set(left.index) & set(right.index)):
        for metric in ("precision", "recall", "f1", "iou", "pr_auc", "false_negative_pixels", "false_positive_pixels", "threshold"):
            rows.append({"head": head, "metric": metric, "baseline": left.loc[head, metric], "candidate": right.loc[head, metric], "delta": right.loc[head, metric] - left.loc[head, metric]})
    output = pd.DataFrame(rows); root = candidate / "metrics"; root.mkdir(parents=True, exist_ok=True); output.to_csv(root / "model_comparison.csv", index=False, encoding="utf-8-sig")
    report = ["# Model comparison", "", f"- baseline: `{baseline}`", f"- baseline checkpoint SHA256: `{baseline_hash or 'legacy'}`", f"- candidate: `{candidate}`", f"- candidate checkpoint SHA256: `{candidate_hash or 'legacy'}`", "", "| head | metric | baseline | candidate | delta |", "|---|---|---:|---:|---:|"]
    report += [f"| {row.head} | {row.metric} | {row.baseline:.6f} | {row.candidate:.6f} | {row.delta:+.6f} |" for row in output.itertuples()]
    (candidate / "reports" / "MODEL_COMPARISON.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    visual = candidate / "visualizations"; visual.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (860, 270), "white"); draw = ImageDraw.Draw(image)
    draw.text((18, 18), "Baseline vs candidate: best validation F1", fill="black")
    for row_index, head in enumerate(sorted(set(left.index) & set(right.index))):
        baseline_f1, candidate_f1 = float(left.loc[head, "f1"]), float(right.loc[head, "f1"])
        y = 70 + row_index * 62; draw.text((18, y), head, fill="black")
        draw.rectangle((150, y, 150 + 550 * baseline_f1, y + 18), fill="#5b8fc4")
        draw.rectangle((150, y + 24, 150 + 550 * candidate_f1, y + 42), fill="#38a169")
        draw.text((715, y), f"base {baseline_f1:.3f}", fill="#285a8e")
        draw.text((715, y + 24), f"candidate {candidate_f1:.3f}", fill="#16713c")
    image.save(visual / "baseline_vs_candidate.png")
    comparison = candidate / "review" / "comparison"; comparison.mkdir(parents=True, exist_ok=True)
    rows_html = "".join(f"<tr><td>{row.head}</td><td>{row.metric}</td><td>{row.baseline:.6f}</td><td>{row.candidate:.6f}</td><td>{row.delta:+.6f}</td></tr>" for row in output.itertuples())
    (comparison / "index.html").write_text(f"<!doctype html><meta charset='utf-8'><title>Model comparison</title><h1>模型对比（validation）</h1><p>基线：{baseline.name}；候选：{candidate.name}。test 未参与。</p><table border='1' cellpadding='6'><tr><th>head</th><th>metric</th><th>baseline</th><th>candidate</th><th>delta</th></tr>{rows_html}</table>", encoding="utf-8")
    return root / "model_comparison.csv"
