# v0.4.0 Plug-and-Train 实施报告

实施日期：2026-07-19（Asia/Shanghai）

## 1. 本轮解决了什么

v0.4 将 v0.3 的多个独立脚本收口为一个普通用户入口：

```bash
python run.py all --aoi <aoi_id>
```

该入口按失败即停止原则执行 doctor、prepare、train、evaluate、predict、track 和 report。正式运行写入独立 run 目录；只有全流程成功才更新 `latest.json`。

## 2. 用户现在如何放数据

```text
data/<aoi_id>/
├── images/
│   ├── 2022-04-01.tif
│   ├── 2023-04-01.tif
│   └── 2024-04-01.tif
└── labels/
    ├── 2022-04-01.gpkg
    ├── 2023-04-01.gpkg
    └── 2024-04-01.gpkg
```

`quality/`、`aoi.gpkg` 与 `project.yaml` 都是可选项。影像和标签必须可读取并带真实 CRS。

## 3. 用户现在如何运行

```bash
pip install -e ".[train]"
python run.py doctor
python run.py all --aoi beijing_demo
```

CPU 可直接安装；CUDA 用户先按 PyTorch 官方命令安装匹配版本，再用 `pip install -e ".[train]" --no-deps` 避免覆盖。

## 4. 自动发现的数据格式

- 影像：`.tif`、`.tiff`；日期支持 `YYYY-MM-DD`、`YYYY_MM_DD`、`YYYYMMDD`。
- 标签：`.gpkg`、`.geojson`、`.shp`，推荐 GPKG。
- 同日存在多个影像或多个标签时列出全部冲突文件并停止。
- 影像和标签日期不一一对应时停止，不静默跳过。
- Windows 空格、中文路径均由 `pathlib` 和 UTF-8 输出处理；已完成中文 AOI 全流程验证。

## 5. Ground Truth 最低要求

标签最低可只有 Polygon/MultiPolygon geometry 和 CRS。系统自动补充 `class_name=building`、`geometry_role=visible_roof`、`review_status=reviewed`、文件名日期、稳定 object ID、schema 0.4.0 与坐标编码。

中英文字段别名会写入 `field_mapping.json`。点、线、空几何、修复失败或完全在影像外的对象进入 rejected；小对象进入 filtered；跨范围对象裁剪。make-valid、MultiPolygon、courtyard/interior rings 均保留。

## 6. 自动 manifest 规则

优先级为：

```text
--manifest > project.yaml manifest > manifest.csv > manifest.generated.csv > 自动发现
```

自动发现写 `data/<aoi_id>/manifest.generated.csv`，不会覆盖手写 `manifest.csv`。date、image、label、CRS 与 split 必须有值；无法自动确定的传感器、太阳高度、离轴角等字段可为空。

## 7. 空间 split 规则

分析 UTM 下生成固定米制 block，固定 seed 决定 train/val；同一 block 在所有日期保持同一 split。tile 只有完全处于一个 block 时才进入相应集合，跨 block 边界的 tile 排除，因此 overlap tile 不会在 train/val 共享像素。验证建筑少于 20 时警告但不伪造验证结果。

最终 demo 报告：train 9 个正切片、val 3 个正切片、边界排除泄漏为 0。

## 8. COCO 导出规则

GeoTIFF 使用 window read。对象按 tile 裁剪并计算 visibility；RLE 支持 Polygon、MultiPolygon 与 holes。正/空切片按配置确定性采样；image ID 和 annotation ID 来自稳定 hash。分别写 `annotations_train.json` 与 `annotations_val.json`，并生成 20 张训练前 preview。

## 9. 训练流程

模型为 torchvision Mask R-CNN ResNet50-FPN，box 与 mask predictor 均替换为 background/building 两类 head。支持 CPU、CUDA、CUDA AMP、gradient clipping、固定 seed、Windows workers=0、early stopping、每 epoch checkpoint 和 resume。

默认 `pretrained=false`，不会隐式下载权重。用户可提供本地 `weights_path`，或显式允许预训练下载。

## 10. best checkpoint 选择规则

每个 epoch 保存 `epoch_XXX.pth`，同步保存 `last.pth`。`best.pth` 按验证 F1 选择；checkpoint 包含 model、optimizer、scheduler、epoch 和 best metric。resume 从 `last.pth` 恢复完整状态。

最终真实 demo：CPU、1 epoch、9 train images、3 val images、训练 3.81 秒，best 和 last 都生成。由于随机初始化、极小数据且只有 1 epoch，best F1 为 0；这证明流水线和 checkpoint 可用，不代表模型精度。

## 11. 验证与推理流程

验证和完整影像推理都只加载 `best.pth`。验证输出 precision、recall、F1、matched IoU、Dice、对象数误差、面积误差、size-stratified CSV、预测/FP/FN/Error GPKG 与 20 张 preview。无验证 GT 时停止数值指标计算。

整图推理复用训练 RGB bands，执行 overlap tiles、polygon NMS 和规则化，写 raw、regularized、WGS84 exchange 与 checkpoint hash。最终 demo 的随机模型在阈值 0.99 下没有检测对象，因此 tracking 合法为空；未把空结果包装成精度成功。

## 12. review 下游闭环修复

student export 优先读取 `tracks_reviewed.gpkg`；状态优先级为 `state_final > reviewed_state > state`。`rejected` 或 `review_status=rejected` 永久排除。人工日期质量不会作为 absence 证据；重复 apply review 幂等并保留 reviewer/lineage provenance。

集成测试已验证：把 auto `new_building` 改为 rejected 后重新导出，该对象不进入 student label。

## 13. Tracking、observation 与 confidence

tracking 和 lineage 先用 STRtree/GeoPandas spatial index 查询 expanded bbox 候选，再计算 IoU、buffer/bbox IoU、centroid、area、shape；assignment 仅在候选连通分量内运行。gate 同时约束距离、重叠和面积比；过大面积变化进入 lineage/construction candidate。ambiguity 使用前后对象双向 margin。

observation status 完整支持 present、usable_absent、not_detected、unusable、occluded、registration_issue、not_covered、ambiguous；只有前两项进入状态证据。结构化 confidence 及可配置权重见 `docs/CONFIDENCE_MODEL.md`，它不是校准概率。

## 14. 缓存规则

输入影像、标签、project YAML、resolved config 和模型配置分别 hash。标签变化使 prepare/split/COCO 失效；训练配置变化重新训练；仅推理阈值变化不重新训练。日志和 RUN_MANIFEST 写 `reused`、`recomputed` 或 `invalidated`。

最终 demo 不带 force 重跑为 9 秒，prepare、train、evaluate、predict、track 五阶段全部实际记录为 `reused`。`--force` 创建新 run 并强制重算；legacy 写文件脚本也拒绝无 force 的静默覆盖。

## 15. 实际测试命令与结果

当前目录不含 `.git`，所以无法拉取或确认 main；本轮以用户 ZIP/source snapshot 实施。基线实际结果为 51 passed；基线 old smoke 成功但使用 mock prediction。

最终实际执行：

```text
python -m pytest -q
=> 90 passed, 1 deselected, 65 warnings in 10.72s

python -m pytest -q -m train_smoke -o addopts=
=> 1 passed, 90 deselected, 5 warnings in 13.73s

python scripts/run_phase1_smoke.py --config configs/example.yaml --force
=> success, outputs/demo_aoi/runs/smoke_20260719T041735Z

python run.py doctor
=> 环境正常；Torch 2.13.0+cpu，Torchvision 0.28.0+cpu，CUDA False

python run.py demo --force
=> success, 22.9s, run 20260719_121832

python run.py demo
=> success, 9.0s, run 20260719_121913；五个计算阶段 cache=reused

python -m pip wheel . --no-deps --wheel-dir <workspace>/wheel-v040
=> urban_renewal_phase1-0.4.0-py3-none-any.whl built successfully
```

警告来自 NumPy 2.5 / Rasterio、pycocotools 的弃用接口，不是测试失败。

## 16. train smoke 结果

`tests/test_training_v040.py` 在 CPU 上真实完成 1 epoch，生成 best/last，重新加载 best 执行 forward，执行 validation inference、完整影像 inference，并从 last 恢复到下一 epoch。不是通过 mock 或仅检查文件存在。

## 17. demo 结果

最终强制重算 run：`outputs/demo_aoi/runs/20260719_121832/`。

- best：`training/checkpoints/best.pth`
- last：`training/checkpoints/last.pth`
- evaluation：`evaluation/evaluation_summary.md`
- predictions：`predictions/`
- tracking：`tracking/`
- review：`review/index.html`
- run manifest：`RUN_MANIFEST.json`

另实际完成中文路径：`python run.py demo --aoi 中文演示 --force`，成功 run 为 `outputs/中文演示/runs/20260719_120304/`。

## 18. 输出目录

正式目录包含 `RUN_MANIFEST.json`、`resolved_config.yaml`、`logs/`、`prepared/`、`data_report/`、`training/`、`evaluation/`、`predictions/`、`tracking/`、`review/` 和 `reports/`。失败 run 保留 failure stage、error、traceback；只在完整成功后写 `outputs/<aoi_id>/latest.json`。

## 19. 已知限制

1. 最终目录无 `.git`，未执行 GitHub pull/push，也不能证明与某远端 main 相同。
2. 只构建 wheel 并在当前依赖环境验证；未在另一台全新机器重装全部 GDAL/Torch 依赖。
3. synthetic 1-epoch 随机模型没有产生预测或轨迹，不代表真实数据精度；真实训练通常需要更多对象和 epoch。
4. 浏览器插件安全策略阻止访问 localhost review 页面；HTML 生成、文件结构和相关单元测试通过，但本轮未获得插件内视觉截图。
5. GCJ-02/BD-09 不内置转换，必须先转换成有真实定义的 CRS 并记录 provenance。
6. off-nadir 缺失时参考影像无法使用该项排序；系统会在 preflight 警告。
7. lineage 是待审核候选，不是自动确认的物理重建事实。

## 20. 下一步接入第一批真实数据

1. 在 `data/<aoi_id>/images` 放入 3—8 期 0.3—2 m GeoTIFF。
2. 在 `labels` 放同日期 Polygon/MultiPolygon GPKG；只有 geometry 即可。
3. 运行 `python run.py doctor`。
4. 先运行 `python run.py prepare --aoi <aoi_id>`，检查 data report、split map 和 COCO previews。
5. 确认无 CRS/日期/范围问题后运行 `python run.py all --aoi <aoi_id>`。
6. 训练后以真实 validation 指标、错误图层和 review 页面决定是否增加 GT、epoch 或调整阈值，不以 synthetic 指标作为验收。
