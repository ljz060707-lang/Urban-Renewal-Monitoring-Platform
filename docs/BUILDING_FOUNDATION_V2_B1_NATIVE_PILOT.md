# Building Foundation v2：B1 原生受控 Pilot

这份说明记录 `B1_NATIVE_RESUME_CONTROL` 的可复现实验边界。它不是一轮新的 RSBuilding 体系，也不会改变已冻结的 `iter_2000.pth`、Full20、Construction 模型、数据集或正式 registry。

## 固定边界

- 初始化 checkpoint：`outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/iter_2000.pth`
- 初始化 SHA256：`f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7`
- 只允许 WHU `train` 和 `val`；WHU Test、SECOND Test、南京、通州和 3D-GloBFP 像素均不读取。
- 先通过人工 WHU label signoff，再通过 32 Train + 32 Validation 的 native/project 输入等价性门控。
- 先执行 20 iter 单样本和 40 iter 八样本 overfit sanity，成功后才允许 500 iter B1 pilot。

## 实际命令

```powershell
$PY = ".venv-gpu\Scripts\python.exe"
$NATIVE = "C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe"
$RUN = "outputs\pretraining\building_foundation_v2_b1_native_pilot_<timestamp>"

& $PY tools\building_foundation_v2_cli.py init-b1-pilot
& $PY tools\building_foundation_v2_cli.py run-b1-pilot --run-dir $RUN --stage sanity --native-python $NATIVE
& $PY tools\building_foundation_v2_cli.py run-b1-pilot --run-dir $RUN --stage b1 --max-iters 500 --native-python $NATIVE
& $PY tools\building_foundation_v2_cli.py finalize-b1-pilot --run-dir $RUN --native-python $NATIVE
```

`native_b1_worker.py` 保留原生 `MultiImageMixDataset` 的打包契约；单样本 sanity 只跳过需要三个不同样本的 Mosaic，仍执行原生载入、裁剪、标签和 `MultiImgPackSegInputs`。训练不会调用随机 scratch 初始化，而是显式加载 frozen `iter_2000.pth`。

## 选择规则

候选必须先有有限值、非坍缩、完整 Validation 和对象级指标。B0/B1 的比较只来自 WHU Validation：像素 IoU/F1、precision/recall、预测/实际面积比例，以及对象 IoU 0.3/0.5/0.7、split/merge、面积偏差和大小分层召回。未达到预注册实质改善门槛时，保留 B0，B1 checkpoint 只作为诊断产物。

## 输出

每次实验写入新的时间戳目录，包含：

- `00_governance/`：保护资产、runtime audit、GPU lock 和科学状态修正；
- `01_label_gate/`：只读导入的人工 WHU label signoff；
- `02_pipeline_equivalence/`：native/project 64 样本对照及差异；
- `03_training_sanity/`：单样本、八样本和有限值门控；
- `04_b1_training/`：native B1 日志、trainable parameter registry、checkpoint；
- `05_validation/`：Validation 像素和对象级指标；
- `11_acceptance/`：最终 receipt、B0/B1 对照、保护资产不变性和 `FINAL_REPORT.md`。

最终 scientific status 必须是实际结果之一：

- `native_baseline_retained_training_pipeline_validated`；
- `b1_native_resume_candidate_ready_for_boundary_ablation`；
- `awaiting_whu_label_human_signoff`；
- `blocked_input_pipeline_not_equivalent`；
- `blocked_training_pipeline_sanity_failed`。

`native_baseline_retained_training_pipeline_validated` 不等于生产可用，也不等于完整 WHU 泛化或人工 QA 完成。B2/B3、长训练、TFR/MTI/Construction/height training 均需另行审批，不能由本 pilot 自动启动。
