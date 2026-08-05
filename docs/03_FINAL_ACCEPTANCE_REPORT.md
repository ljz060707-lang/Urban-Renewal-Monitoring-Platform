# 最终训练就绪验收报告

实施日期：2026-07-21（Asia/Shanghai）。本报告只记录已实际执行的检查；未拥有的数据和未执行的训练不会标记为通过。

## 本轮工程修复

- 公开变化教师与个人 AOI 建筑实例流程使用独立命令、数据根目录和输出目录。
- 新增 `change-preflight`：检查 CUDA、公开数据目录、配对、分割隔离、标签监督能力、SECOND 映射和磁盘空间；只读原始数据。
- validation 与 test 彻底隔离。只有显式 `use_test_as_validation: true` 才会借用 test，并写入报告。
- 不具备某一阶段监督的公开数据集会写为 `skipped_no_supervision`，不会伪造 checkpoint 或指标。
- `--quick` 只抽取确定性的 train/validation 子集，绝不抽取 test。
- 个人流程支持 `--data-root "D:/Data Set/Personal"`、`personal-preflight`，并确定性选择 GPKG 的 `buildings` 图层（或单图层文件）。

## 实际外部数据检查

| 数据集 | 目录 | train | validation | test | 实际结论 |
|---|---|---:|---:|---:|---|
| S2Looking | `D:/Data Set/S2Looking` | 3500 | 500 | 1000 | 5000 个成对样本已通过读取、尺寸、监督和方向一致性检查；可作为二分类和方向阶段训练数据。 |
| SECOND | `D:/Data Set/SECOND` | 0 | 0 | 0 | 目录不存在；且正式配置没有已验证的类别映射，阻断全量三教师训练。 |
| DSIFN | `D:/Data Set/DSIFN` | 0 | 0 | 0 | 目录不存在，阻断全量三教师训练。 |
| Personal | `D:/Data Set/Personal` | — | — | — | 根目录不存在；尚无可验收的个人 AOI 影像和标注。 |

S2Looking 的 RGB `label1`/`label2` 采用其已验证的黑底、单一前景色编码读取；通用二值标签读取仍拒绝任意 RGB 标签，避免错误接受未知编码。

## 实际 GPU quick run

已执行：

```powershell
$env:PYTHONPATH='src'
.venv-gpu\Scripts\python.exe -u run.py change-all --config configs/change_public_s2looking_windows.yaml --device cuda:0 --quick
```

结果：退出码 0，耗时约 327 秒，实际运行目录：

`outputs/pretraining/change_s2looking/20260721_165658/`

- 设备：NVIDIA GeForce RTX 4060 Laptop GPU，8,188 MiB；Torch `2.12.1+cu130`；CUDA runtime `13.0`。
- quick index：32 个 train、8 个 validation、0 个 test；验证来源为 validation。
- 二分类和方向阶段成功；语义阶段因 S2Looking 没有语义监督而明确写为 `skipped_no_supervision`。
- 最终 checkpoint：`checkpoints/best_directional_change.pth`；评估、训练仪表盘和静态 review 均已生成。

已重新用 CPU 加载该 checkpoint。载入结果的 `stage` 为 `directional`、`epoch` 为 `0`，且其 `RUN_MANIFEST.json` 状态为 `success`。训练后的验证函数也实际以该最终 checkpoint 运行，不是只检查文件存在。

## 实际命令与测试结果

| 命令 | 结果 |
|---|---|
| `.venv-gpu\Scripts\python.exe -m pytest -q --basetemp ..\..\pytest-v051-full-rerun` | `122 passed, 3 deselected`。 |
| `.venv-gpu\Scripts\python.exe -m pytest -q -m train_smoke --basetemp ..\..\pytest-v051-train-smoke` | `3 passed, 122 deselected`；包含 tiny CPU 训练、best checkpoint 重载、推理和 resume。 |
| `run.py change-demo --device cpu --force` | 成功；输出 `outputs/pretraining/change_demo/20260721_171230/`。 |
| `run.py change-doctor --config configs/change_public_windows.yaml` | 按预期失败：GPU 和 GIS 包正常，但 SECOND 目录不存在。 |
| `run.py change-preflight --config configs/change_public_windows.yaml` | 按预期失败：S2Looking 通过；SECOND/DSIFN 路径缺失。报告为 `outputs/pretraining/change/preflight/PRETRAIN_CHECKLIST.{md,json}`。 |
| `run.py personal-preflight --aoi nanjing_pilot --data-root "D:/Data Set/Personal"` | 按预期失败：`images/` 目录不存在。报告为 `outputs/nanjing_pilot/preflight/PERSONAL_PREPARE_CHECKLIST.md`。 |

未执行的真实训练：DSIFN-only quick、SECOND-only quick、三数据集联合 quick 和真实个人 AOI 训练；原因都是对应原始数据不存在，不能以 synthetic 数据替代或声称通过。

## 修改范围

- `src/urban_monitor/change/{splits,training,evaluation,preflight,cli,preparation,registry}.py`：公开数据预检、严格 split 隔离、确定性 quick、按监督跳过阶段、最后成功 checkpoint 下游联动。
- `src/urban_monitor/change/adapters/{base,s2looking,second}.py`：S2Looking 已知 RGB 标签安全读取；SECOND 显式 `mapping_verified: false` 阻断正式流程，RGB 语义颜色绝不猜测。
- `src/urban_monitor/{cli,preflight,ground_truth,preparation,config}.py` 和 `change/local_labels.py`：外部个人数据根、个人预检、geometry-only/单图层 GPKG 兼容和多图层歧义阻断。
- `configs/change_public_windows.yaml`、`configs/change_public_s2looking_windows.yaml`、`scripts/windows/`：Windows 空格路径的正式配置与一键脚本。
- `README.md`、`docs/01_PUBLIC_DATASET_TRAINING_WINDOWS.md`、`docs/02_PERSONAL_QGIS_GEOOSAM_WORKFLOW.md`、本报告：面向实际操作的说明与边界。
- `tests/test_training_readiness_v051.py`：覆盖 split、quick、标签读取、阶段跳过、GPKG 图层和外部数据根。

## 当前结论

- `READY_FOR_PUBLIC_FULL_TRAINING`: **否**。S2Looking 单数据集训练已验证，但 SECOND、DSIFN 不在给定路径，SECOND 类别映射也未被确认；因此不能声称完整公开三教师训练已就绪。
- `READY_FOR_PERSONAL_ANNOTATION_AND_TRAINING`: **否**。程序和 QGIS/GeoOSAM 接入流程已准备好，但 `D:/Data Set/Personal` 尚不存在，无法对真实个人 AOI 做数据预检或训练验收。

## 下一步

```powershell
python run.py change-preflight --config configs/change_public_windows.yaml
python run.py personal-preflight --aoi nanjing_pilot --data-root "D:/Data Set/Personal"
python run.py change-labels --aoi nanjing_pilot --data-root "D:/Data Set/Personal"
```

补齐 SECOND、DSIFN 及其已验证 `class_mapping_file` 后，再执行完整公开训练；个人 AOI 放入对应 `images/` 与 `labels/` 后，先运行个人预检，再运行 `python run.py all --aoi <aoi_id> --data-root "D:/Data Set/Personal"`。
