# v0.4.0 实施前审计

审计时间：2026-07-19（Asia/Shanghai）

## 仓库与分支

- 当前交付目录不含 `.git`，无法识别本地分支、远端或提交，也无法执行可信的 `git pull main`。
- 本轮以用户提供的 v0.3.0 源码快照为基线继续实施。恢复 GitHub 同步需要仓库 URL 或包含 `.git` 的克隆目录。

## 实际基线验证

执行：

```powershell
python -m pytest -q
python scripts/run_phase1_smoke.py --config configs/example.yaml --force
```

由于当前 Windows 会话的系统 Python 不可靠，实际使用 Codex bundled Python，并通过 `PYTHONPATH` 加载工作区依赖；pytest 另指定工作区 `--basetemp`。

- pytest：`51 passed, 9 warnings in 3.10s`。
- warning：均为 Rasterio 1.5 / NumPy 2.5 的 shape 弃用提示。
- smoke：成功，run 目录为 `outputs/demo_aoi/runs/smoke_20260719T025402Z/`。
- smoke 生成 14 个 COCO image、20 个 annotation、6 条轨迹和 1 期 student label。
- smoke 的 segmentation 指标来自 synthetic GT 与 mock prediction 的比较；tracking/event/grid 因无独立 GT 明确为 `not computable`。
- smoke 没有训练或加载 Mask R-CNN。当前开发环境检测结果为 `torch=False`、`torchvision=False`。

## 当前训练和推理入口

- 训练入口：`scripts/train_teacher.py` → `urban_monitor.training.train_teacher()`。
- 当前训练只写 `epoch_XXX.pth` 与 `latest.pth`，没有验证循环、`best.pth`、`last.pth`、完整 resume、AMP、early stopping 或训练报告。
- 推理入口：`scripts/infer_timeseries.py`，默认读取 `checkpoints/latest.pth`，不是 `best.pth`。
- 旧 smoke 调用 `scripts/make_mock_predictions.py`，所以 smoke 成功不能证明训练 checkpoint 被推理使用。

## 普通用户尚不能直接放入数据训练的原因

1. 所有主要脚本都要求 `--config`，且配置中必须写 `data_root`、`manifest` 与 `output_root`。
2. `manifest.csv` 必须人工准备；没有文件发现、日期解析、冲突检查或自动生成。
3. 数据须预先处于相同投影 CRS、范围、分辨率与栅格网格；没有自动生成对齐工作副本。
4. annotation validator 强制要求 `object_id`、`class_name`、`geometry_role`、`review_status`、`source_date`、`notes`、`tracking_geometry_wkt`；geometry-only 标签不能直接使用。
5. COCO 只导出含对象切片，只有单一 `annotations.json`，没有空间 train/validation 划分、空样本控制或 preview。
6. 缺少根目录统一入口、数据 preflight、缓存和正式 run 生命周期。

## 风险与断点

- **manifest**：当前必须人工填写。
- **Ground Truth**：字段要求对普通 QGIS 用户过严；点、线、空几何和修复失败对象会使验证直接失败，未形成 reject/filtered 报告。
- **空间泄漏**：当前 split 来自 manifest 的日期行，切片没有空间 block 分组；重叠切片和同一建筑跨日期可能进入不同集合。
- **best checkpoint**：不能保存；只保存 epoch 和 latest。
- **训练结果推理**：真实推理可加载 checkpoint，但默认 latest；现有 smoke 完全没有使用训练结果。
- **review 闭环**：`apply_reviews` 写出 `tracks_reviewed.gpkg` 和 `state_final`，但 `export_student_labels.py` 固定读取 `tracks_summary.gpkg`，`downscale.py` 固定使用 `state`；review 不会影响 student export。
- **--force**：`run_phase1_smoke.py` 和 synthetic 数据生成路径实际检查或使用它；多数脚本只解析参数，没有改变覆盖/缓存行为。
- **tracking / lineage 规模**：`_match`、forced merge 和 lineage adjacency 都有前后对象全量嵌套比较，属于无界 O(N²) 候选生成；ambiguity 仅按 current 方向计算 margin。
- **observation matrix**：只有 `usable/present`，没有完整 `observation_status`，未检测容易被当作可用 absence。
- **confidence**：主要按时间间隔与 mean score 分 high/medium/low，没有结构化 `confidence_score`。
- **README**：263 行，英文和内部技术内容较多，要求用户手写 manifest 并执行多个脚本，不适合第一次使用的普通用户。

## v0.4.0 收口原则

- 复用现有 GIS、几何、tracking、review 与 provenance 能力。
- 只增加一条面向用户的 `run.py` 路径；旧脚本作为兼容入口保留。
- 每一阶段先补最小可运行测试并通过，再进入下一阶段。
- synthetic/demo 指标只用于验证流水线，不声明真实中国城市泛化能力。

## 实施后闭环（2026-07-19）

- 根入口、自动发现、geometry-only GT、自动 UTM、空间 split、COCO、真实训练、验证、整图推理、tracking、review、缓存和 run 生命周期均已实现。
- 基础测试最终为 `90 passed, 1 deselected`；单独真实训练 smoke 为 `1 passed`。
- 最终网络隔离 demo 成功生成 `best.pth`、`last.pth`、验证指标、完整影像预测、review HTML 与 RUN_MANIFEST。
- 不带 `--force` 重跑时，prepare、train、evaluate、predict、track 均在 RUN_MANIFEST 中记录为 `reused`。
- 本文件前半部分保留实施前事实，不能把旧 smoke 的 mock 指标误写成真实模型能力。
