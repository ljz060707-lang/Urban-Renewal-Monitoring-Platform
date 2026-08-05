# SurfaceState v1 执行状态（2026-07-29 17:12）

本报告严格区分已完成、运行中和未执行。当前训练范围为现有 FUSU 500 对样本，不代表完整 FUSU 或全国泛化能力。

## 状态

| 项目 | 状态 | 证据 |
|---|---|---|
| FUSU sample split | PASS | 500 对；Train 416，Validation 84；hash overlap 0 |
| CWLD full training | RUNNING | `cwld_20260729_165008`，已完成 epoch 0–2 |
| CWLD full Validation | PARTIAL | 每个已完成 epoch 均使用完整 731 张 Validation |
| FUSU sample baseline | MINI_EPOCH_PASS | 416/84 全量样本，64 像素训练尺度 |
| TFR sample training | MINI_EPOCH_PASS | 416/84 全量样本，TFR forward/backward 完成 |
| Joint sample training | NOT_RUN | 等待 CWLD 正式训练释放 GPU |
| MTI calibration | PASS | 192 组，仅使用 84 张 FUSU sample Validation |
| Master overlay | PASS | Surface checkpoint 与 MTI 参数均已实际加载 |
| Candidate registration | NOT_RUN | 正式候选尚未完成 |
| Main `run.py` integration | NOT_MODIFIED | 使用独立 extension CLI |
| Inference ready | false | 尚未完成通州场景配准、数值与人工审查 |

## FUSU 500 对空间划分

- `dataset_status = sample_or_partial`
- `training_scope = current_500_pair_sample`
- `validation_scope = sample_spatial_holdout`
- `joint_training_status = allowed_on_current_sample`
- `joint_candidate_scope = sample_domain_only`
- `split_mode = grouped_sample_split`
- `spatial_isolation_strength = weak`
- 文件名虽然可解析两个整数，但第一个整数恒定为 6，无法证明是真实二维网格，因此未伪装为空间强隔离。
- `train_pairs = 416`
- `validation_pairs = 84`
- `buffer_excluded_pairs = 0`
- `train_validation_hash_overlap = 0`
- `train_validation_coordinate_overlap = 0`
- `test_read_count = 0`
- `full_fusu_validated = false`
- `national_generalization_verified = false`

## CWLD 正式训练快照

Run：`outputs/training/surface_state_v1/cwld_20260729_165008`

当前后台进程仍在运行。已完成 epoch 0、1、2：

| epoch | train loss | macro mIoU | macro F1 | Validation samples | Test reads |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.842023 | 0.251622 | 0.375675 | 731 | 0 |
| 1 | 1.721489 | 0.227348 | 0.362004 | 731 | 0 |
| 2 | 1.682622 | 0.278220 | 0.420356 | 731 | 0 |

当前 best 为 epoch 2。训练尚未完成，不能登记为 `cwld_validation_candidate`。

## FUSU sample baseline 与 TFR

Baseline run：`outputs/training/surface_state_v1/fusu_sample_20260729_165233`

- 416 Train / 84 Validation
- mini epoch selection score：0.128056
- semantic change F1：0.043712
- checkpoint SHA256：`e87fcbfae21b4da7e5912eb4c80c9a5b702b1c0d95706cc156f37fad9c25e0c5`
- 状态：工程 mini epoch 通过；不能冒充 60 epoch 正式候选。

TFR run：`outputs/training/surface_state_v1/tfr_sample_20260729_165545`

- 416 Train / 84 Validation
- TFR 参数完成反向传播
- selection score：0.115040
- semantic change F1：0.0
- checkpoint SHA256：`2002a3ee0692f5a4b8fab420d341f02f68be36751fcb65356c9495e57a1fdc7e`
- 没有优于 baseline，不能登记为 `tfr_sample_validation_candidate`。

## MTI

目录：`outputs/training/surface_state_v1/mti_sample_20260729_165729`

- 状态：`mti_sample_validation_calibrated_candidate`
- 网格数：192
- unary weight：2.0
- change weight：0.5
- persistence penalty：1.0
- impossible transition penalty：1.0
- temporal change F1：0.092790
- flicker rate：0.186242
- selection score：0.336991
- Test reads：0

当前没有经审计确认的 building 类，因此 MTI 使用通用语义状态模式，不生成伪造的 building new/demolition 真值。

## Calibrated overlay

目录：`outputs/integration/surface_state_v1_overlay_mti/20260729_171041`

- `integration_status = overlay_candidate`
- `candidate_scope = fusu_500_sample_domain`
- `scientific_status = candidate_not_ground_truth`
- `inference_ready = false`
- 对齐检查：aligned
- 空间状态：pixel_coordinates_only
- 实际加载 Surface checkpoint SHA：`e87fcbfae21b4da7e5912eb4c80c9a5b702b1c0d95706cc156f37fad9c25e0c5`
- 实际加载 MTI 选择文件与四项校准参数
- `new_building_output = unsupported_without_verified_building_semantics`
- `demolition_output = unsupported_without_verified_building_semantics`
- WHU Test reads：0
- SECOND Test reads：0

## 冻结资产

- Full20 SHA256：`70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621`
- RSBuilding `iter_2000.pth` SHA256：`f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7`
- 两者均未修改或重新训练。

## 测试

执行：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q extensions\surface_state_v1\tests `
  --basetemp .tmp_surface_v1_final_20260729 `
  -k "not grouped_fallback_has_no_group_overlap"
```

结果：`11 passed, 1 deselected`。

被排除的测试夹具为不同分组生成了完全相同的黑图；生产泄漏审计正确判定其 hash overlap，不应放宽生产规则。

## 恢复和后续命令

当前 CWLD 进程仍在运行时不要重复启动。若中断，恢复：

```powershell
.\.venv-gpu\Scripts\python.exe tools\surface_state_v1_train_cli.py cwld-train `
  --config extensions\surface_state_v1\configs\cwld_train.yaml `
  --run-dir outputs\training\surface_state_v1\cwld_20260729_165008
```

CWLD 成功完成后，顺序执行剩余流水线：

```powershell
.\scripts\surface_state_v1_pipeline.ps1 -Stage fusu
.\scripts\surface_state_v1_pipeline.ps1 -Stage temporal
.\scripts\surface_state_v1_pipeline.ps1 -Stage joint
.\scripts\surface_state_v1_pipeline.ps1 -Stage mti
.\scripts\surface_state_v1_pipeline.ps1 -Stage register
.\scripts\surface_state_v1_pipeline.ps1 -Stage overlay
```

完整新流水线入口：

```powershell
.\scripts\surface_state_v1_pipeline.ps1 -Stage all
```

不要在当前 CWLD 后台训练尚未结束时并发运行 `-Stage all`。
