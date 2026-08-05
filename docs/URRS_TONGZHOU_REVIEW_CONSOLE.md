# URRS Tongzhou Review Console

## 启动

在项目根目录执行：

    .\scripts\urrs_review.ps1

或：

    .\.venv-gpu\Scripts\python.exe tools\urrs_review.py serve --review-dir outputs\urrs\review\tongzhou_20260801 --port 8765

然后打开 http://127.0.0.1:8765。

服务只监听 localhost，只读取已有 Overlay、Geometry 和审计文件，不启动 GPU，不重新推理，不校准阈值。

## 清单与输出

固定 seed 为 20260801。当前清单包含四期 Geometry、三组已有 Full20 transition，以及 Surface State 样本。首次运行生成：

- REVIEW_INVENTORY.json / CSV
- previews/
- review_state.json
- review_events.jsonl
- review_decisions.csv
- review_decisions.jsonl

每次保存使用临时文件后原子替换，并记录 before/after 事件；撤销只回滚最近一次事件。

## 提交

完成必审样本后，在界面点击提交人工审查。系统要求确认：

- 全部必审样本；
- 2022 fragmentation；
- 三组 transition；
- Surface State；
- candidate_not_ground_truth；
- 通州未用于 checkpoint/阈值；
- reviewer 非空。

未满足条件时只保存草稿，不生成最终人工结论。

命令行状态和导出：

    python tools\urrs_review.py status --review-dir outputs\urrs\review\tongzhou_20260801
    python tools\urrs_review.py export --review-dir outputs\urrs\review\tongzhou_20260801
    python tools\urrs_review.py finalize --review-dir outputs\urrs\review\tongzhou_20260801 --confirmation-json HUMAN_CONFIRMATION.json

人工确认后再调用：

    python tools\urrs.py system finalize --review-file <path-to-HUMAN_VISUAL_REVIEW.json>

审核结果不会修改原始影像、GeoTIFF、GPKG、checkpoint 或模型阈值。
