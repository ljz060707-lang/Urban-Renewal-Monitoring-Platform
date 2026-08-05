# Building Foundation v2 B2 Boundary Sanity Failure Diagnosis

这是一条独立的诊断流水线，不是新的 B2 Pilot。它复用失败 B2 run 实际使用的 native-positive WHU Train 样本，关闭所有随机增强，并依次检查：

1. boundary target/metric 自洽性；
2. native decoder feature 的分辨率上限；
3. B0 单样本 Building 基线；
4. detached boundary head 是否可学习；
5. shared decoder 梯度余弦与 Building 输出漂移；
6. 修订后的 sanity gate 和 Boundary 的职责。

## 命令

```powershell
& .venv-gpu\Scripts\python.exe tools\init_building_v2_b2_boundary_diagnostic.py `
  --previous-run outputs\pretraining\building_foundation_v2_b2_boundary_ablation_20260805_212147

& .venv-gpu\Scripts\python.exe tools\run_building_v2_b2_boundary_diagnostic.py `
  --run-dir outputs\pretraining\building_foundation_v2_b2_boundary_diagnostic_<run_id> `
  --stage all
```

`--stage all` 只包含最多 500 step 的 detached head probe 和 100 step 的共享梯度探针，不启动完整 Pilot，也不运行 1036 张候选 Validation。WHU Test、SECOND Test、南京、通州、3D-GloBFP、Height、TFR、MTI 和 Construction 都被明确排除。

## 解释边界

本诊断的 sanity gate 只回答“训练链是否能学习边界”，不等同于候选模型晋升门槛。若 detached head 能达到 oracle 而 shared scope 使 Building logits 漂移，应优先考虑 `ROUTE_A_BOUNDARY_PROBE_FOR_OBJECTIZATION` 或带 B0-logit preservation 的低权重辅助损失，不应直接增加通用正则化。

最终状态写入 run 的 `11_acceptance/FINAL_B2_DIAGNOSTIC_RECEIPT.json` 和 `FINAL_REPORT.md`。
