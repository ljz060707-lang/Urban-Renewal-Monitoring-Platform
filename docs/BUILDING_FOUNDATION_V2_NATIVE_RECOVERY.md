# Building Foundation v2：原生 RSBuilding 恢复流程

本流程用于恢复并审计冻结的原生 RSBuilding `iter_2000.pth`，不把旧的 scratch U-Net 当作正式候选。每次运行写入新的
`outputs/pretraining/building_foundation_v2_native_recovery_<timestamp>/`，不会覆盖历史训练、模型注册表或原始数据。

## 安全边界

- WHU 只读取 `train` 和 `val`；适配器对 `test` 行和 `split=test` 直接拒绝。
- WHU Test/SECOND Test 像素读取必须保持 0。
- 原生 RSBuilding 使用独立环境 `C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe`；主项目 `.venv-gpu` 不安装或修改原生依赖。
- Full20、`iter_2000.pth`、registry 和原始数据均只读。
- 在标签合同、原生模型加载、完整 Validation 复现和输入管线等价性明确前，不启动长训练。

## CLI

```powershell
$PY = ".\.venv-gpu\Scripts\python.exe"
& $PY tools\building_foundation_v2_cli.py init-native-recovery --run-dir <RUN_DIR>
& $PY tools\building_foundation_v2_cli.py audit-labels --root "D:\Data Set\WHU_Building" --run-dir <RUN_DIR>
& $PY tools\building_foundation_v2_cli.py inspect-native --run-dir <RUN_DIR>
& $PY tools\building_foundation_v2_cli.py compare-pipeline --run-dir <RUN_DIR>
& $PY tools\building_foundation_v2_cli.py validate-native --run-dir <RUN_DIR>
& $PY tools\building_foundation_v2_cli.py finalize-native --run-dir <RUN_DIR>
```

原生 Validation 使用固定的 `whu_val_master.txt`（1,036 张），以原生配置、预处理和 checkpoint 直接推理一次。最终化会补齐旧版 Validation receipt 缺少的复现/分布字段，但不会重复推理。

## 当前恢复结果

当前 run 的 `02_native_recovery/NATIVE_RECOVERY_GATE.json` 为 `pass`，原生 `iter_2000` 已严格加载，完整 WHU Validation 复现通过，预测分布为 `healthy_prediction_distribution`。项目 adapter 与原生 6 通道、BGR→RGB、mean/std、增强仍记录为 `blocked_project_adapter_not_native_equivalent`，因此 `formal_training_allowed=false`；这不是候选模型改善结果。

正式 B1/B2/B3/H 训练、Nanjing/Tongzhou 诊断和 TFR/MTI/Construction 训练均未启动。下一步必须先完成人工标签预览签字和 native-equivalent input pipeline 对齐，再单样本/八样本 overfit 与短控实验。
