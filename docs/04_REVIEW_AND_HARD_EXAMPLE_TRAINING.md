# 首轮审核、阈值与困难样本训练

“漏检”是 GT 有建筑变化、模型未检出；“误检”相反。低阈值下整图偏蓝可能只是概率阈值过低，必须以 validation 扫描给出的默认阈值判断，不能凭颜色直接判定模型失效。

先生成或打开审核页：

```powershell
python run.py change-review --run 20260721_232528 --config configs/change_public_s2looking_windows.yaml
cd outputs/pretraining/change_s2looking/20260721_232528/review
python -m http.server 8000
```

右侧可切换真值、Binary、新建、拆除和差异图。阈值 1.00 或透明度 0 时应只显示 T2。用 1–7 标注决定，Enter 保存并下一条；导出 `review_decisions.csv` 或 `jsonl`，下次可重新导入。

阈值只能由 validation 选择：

```powershell
python run.py change-threshold-scan --run 20260721_232528 --config configs/change_public_s2looking_windows.yaml --split validation --device cuda:0
python run.py change-hard-examples --run 20260721_232528 --review-csv outputs/pretraining/change_s2looking/20260721_232528/review/review_decisions.csv --config configs/change_public_s2looking_hard_finetune.yaml
python run.py change-retrain --run 20260721_232528 --config configs/change_public_s2looking_hard_finetune.yaml --from-review --device cuda:0
```

困难样本索引明确按原始 split 门控：validation/test 中的漏检只用于诊断，绝不会写入训练索引。第二轮会新建 candidate run，首轮 checkpoint 不会被覆盖。

## 审核训练集困难样本

默认仍审核 validation。要为人工确认的困难样本准备训练集审核页，使用首轮 directional checkpoint 对**原始 train**推理：

```powershell
python run.py change-review --run 20260721_232528 --config configs/change_public_s2looking_windows.yaml --split train --selection hard --max-samples 150 --device cuda:0
```

页面写入 `review_train/`，不会覆盖 `review/`；导出文件名为 `review_decisions_train.csv` 和 `review_decisions_train.jsonl`。hard 排序依次优先真实变化近乎全空预测、低 recall、高 false-negative、新/拆除漏检和高不确定性。test 不会作为 Dataset 打开。
