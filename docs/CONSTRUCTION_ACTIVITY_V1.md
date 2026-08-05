# 建筑工地与施工状态识别服务

`construction_activity_v1` 是 URRS 的独立实验候选模块。它不替换现有
`URRS-Tongzhou-v1.0-candidate`，也不会修改 Full20、Surface Joint、
RSBuilding 或当前已选中的 no-MTI 时序路径。

## 科学边界

- 单期施工要素预测不是城市更新 Ground Truth。
- UCS 与 CWLD 保持各自原生四类标签空间；缺失类别不作为负样本。
- CWLD `engineering facility` 只形成工程设施证据，不能直接解释为施工工地。
- `pseudo_rule` 和 `model_prediction` 不进入正式 Validation 指标。
- 施工 MTI 当前为 `EXPERIMENTAL UNSELECTED`，正式路径仍是 no-MTI。
- 通州只用于推理与人工审核，不用于 checkpoint、阈值或类别权重选择。

## 数据位置与审计

默认只读发现 `D:\Dataset` 和 `D:\Data Set`，也可重复传入 `--data-root`：

```powershell
.\.venv-gpu\Scripts\python.exe tools\urrs.py construction audit `
  --data-root "D:\Data Set\URS" `
  --data-root "D:\Data Set\CWLD"
```

审计输出位于：

```text
outputs/urrs/audits/construction_activity_<timestamp>/
```

正式训练只在 pairing、标签、split 和许可证门禁全部通过后才允许。当前本地
URS 两个 package 目录均包含相同的 386 张 TIFF 和相同的 600 张 PNG；去重后
仍为 386 对，另有 214 个标签无影像，因此正式 UCS 训练被阻止。

## 开发级 smoke

```powershell
# 只检查 dataloader
.\.venv-gpu\Scripts\python.exe tools\urrs.py construction smoke --dataloader-only

# 固定 20 个 optimizer step；不会登记为正式模型
.\.venv-gpu\Scripts\python.exe tools\urrs.py construction smoke --device cpu
```

smoke checkpoint 带 `checkpoint_registerable=false`，不能进入正式 registry。

## 审核界面

```powershell
.\scripts\urrs_construction.ps1 -Stage review
```

打开 `http://127.0.0.1:8765`，切换到“施工状态与生命周期”。施工审核写入：

```text
outputs/urrs/review/tongzhou_20260801/construction_activity/
  construction_review_decisions.csv
  construction_review_decisions.jsonl
  construction_review_state.json
  REVIEW_CORRECTIONS.gpkg
```

这些文件与原有 `review_decisions.csv/jsonl` 完全独立。页面中的默认状态是
`unknown_state` / `uncertain`，明确显示 `PSEUDO LABEL — NOT GROUND TRUTH`；
用户确认后保存为 `human_verified`，修改后保存为 `human_corrected`。

## 当前状态

| 项目 | 状态 |
|---|---|
| 数据发现与标签审计 | PASS |
| 多 package 去重与 split 审计 | PASS（pairing 未通过） |
| dataloader smoke | PASS |
| 20-step CPU smoke | PASS |
| 正式 UCS / joint 训练 | BLOCKED |
| 地块分类器 | PENDING HUMAN LABELS |
| TFR 正式 Validation | NOT RUN |
| ConstructionMTI | EXPERIMENTAL UNSELECTED |
| 通州施工状态模型输出 | NOT RUN |
| 当前 URRS release | UNCHANGED |

