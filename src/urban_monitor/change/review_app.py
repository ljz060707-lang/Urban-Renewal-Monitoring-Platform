from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .corrections import save_correction


def launch_review_app(run: str | Path, *, share: bool = False):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError('Interactive review requires: pip install -e ".[review]"') from exc
    run = Path(run).resolve(); samples_path = run / "review" / "review_samples.csv"
    if not samples_path.is_file(): raise FileNotFoundError(samples_path)
    frame = pd.read_csv(samples_path).fillna("")
    decisions_path = run / "review" / "review_decisions.csv"

    def show(index):
        index = max(0, min(int(index), len(frame)-1)); row = frame.iloc[index]
        return index, row["sample_id"], row["t1_path"], row["t2_path"], row.get("preview_path", ""), f"{index+1}/{len(frame)}"

    def save(index, decision, notes, reviewer, binary_file, new_file, demolition_file):
        row = frame.iloc[int(index)]; correction_paths = []
        for task, file in (("binary", binary_file), ("new", new_file), ("demolition", demolition_file)):
            if file:
                correction_paths.append(str(save_correction(file, run, row["sample_id"], task, reviewer=reviewer, notes=notes)))
        record = {"sample_id": row["sample_id"], "dataset_name": row["dataset_name"], "stage": row["stage"], "decision": decision,
                  "notes": notes, "reviewer": reviewer, "review_time": datetime.now(timezone.utc).isoformat(), "threshold": 0.5,
                  "correction_path": ";".join(correction_paths)}
        previous = pd.read_csv(decisions_path).to_dict("records") if decisions_path.exists() else []
        previous = [item for item in previous if item["sample_id"] != row["sample_id"]] + [record]
        pd.DataFrame(previous).to_csv(decisions_path, index=False, encoding="utf-8-sig")
        return "已保存，不会修改原始公共数据。"

    with gr.Blocks(title="变化教师人工审核") as app:
        index = gr.State(0); gr.Markdown("# 变化教师人工审核\n修正内容写入当前 run/review/corrections。")
        sample_id = gr.Textbox(label="sample_id", interactive=False); position = gr.Markdown()
        with gr.Row(): t1 = gr.Image(label="T1"); t2 = gr.Image(label="T2"); preview = gr.Image(label="GT / prediction")
        with gr.Row(): previous = gr.Button("上一张"); following = gr.Button("下一张")
        decision = gr.Dropdown(["correct","prediction_error","label_error","misregistration","uncertain","exclude","needs_mask_edit"], label="审核状态")
        notes = gr.Textbox(label="中文备注"); reviewer = gr.Textbox(label="审核人")
        binary_file = gr.File(label="修正 binary mask"); new_file = gr.File(label="修正 new mask"); demolition_file = gr.File(label="修正 demolition mask")
        save_button = gr.Button("保存", variant="primary"); status = gr.Markdown()
        app.load(show, [index], [index, sample_id, t1, t2, preview, position])
        previous.click(lambda value: max(0, int(value)-1), index, index).then(show, index, [index, sample_id, t1, t2, preview, position])
        following.click(lambda value: min(len(frame)-1, int(value)+1), index, index).then(show, index, [index, sample_id, t1, t2, preview, position])
        save_button.click(save, [index, decision, notes, reviewer, binary_file, new_file, demolition_file], status)
    return app.launch(inbrowser=True, share=share)
