# Building Foundation v2 B2 边界辅助消融

这是一条隔离的 native RSBuilding 边界辅助消融流程。它保留 B0 的原生 building 输出，只在高分辨率 decoder feature 上增加轻量 boundary head；B2 必须从受保护的 `iter_2000.pth` 初始化，不能从 B1 初始化。

## 运行

```powershell
python tools/building_foundation_v2_cli.py init-b2-boundary
python tools/building_foundation_v2_cli.py run-b2-boundary --run-dir <B2_RUN> --stage smoke
python tools/building_foundation_v2_cli.py run-b2-boundary --run-dir <B2_RUN> --stage positive-overfit --one-iters 500
python tools/building_foundation_v2_cli.py finalize-b2-boundary --run-dir <B2_RUN>
```

`smoke` 和 `positive-overfit` 是 pilot 的硬门控。只有三种边界权重（0.05、0.10、0.20）都通过有限值、梯度和单样本 overfit 门控，才允许启动 pilot；否则会生成 blocked receipt，不会伪造候选。

单样本 overfit 的 `one_sample_positive_constantlr_*` 仅用于结构/梯度诊断，明确绕过源配置的 1000-step warmup；正式 smoke/pilot 保留源 scheduler。所有 manifest 只接收固定 B1 Train registry 和 WHU Validation，WHU Test、SECOND Test、南京、通州、3D-GloBFP、Construction、Height 和伪标签均不读取。

## 产物

每次运行写入独立的 `outputs/pretraining/building_foundation_v2_b2_boundary_ablation_<timestamp>/`，包含治理、边界 target 审计、架构/参数继承、smoke、pilot、Validation 和 acceptance receipts。`MODEL_SELECTION_RECEIPT.json` 是唯一模型选择来源；失败或未通过门控时不修改正式 registry。

## 当前运行边界

2026-08-05 的 B2 run 在 20-step smoke 后进入了真实 native-positive overfit 诊断；500-step constant-LR 诊断仍未达到 Building IoU ≥0.98 和 Boundary F1@2 ≥0.95，因此 pilot/full B2 Validation 被正确阻止。B0/B1 的 1036 张 WHU Validation 契约已独立复算；当前科学状态是 `blocked_b2_training_pipeline_failed`，不是 B2 candidate。
