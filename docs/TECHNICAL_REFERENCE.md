# v0.4 技术参考

## 1. 项目发现与配置

正式数据根目录为 `data/<aoi_id>`。`images` 与 `labels` 必须存在；`quality`、`aoi.gpkg`、`project.yaml` 可选。

日期文件名支持 `YYYY-MM-DD`、`YYYY_MM_DD`、`YYYYMMDD`。同一天多个同类文件会列出全部冲突路径并停止。影像与标签日期必须一一对应。

manifest 优先级：

```text
--manifest > project.yaml manifest > manifest.csv > manifest.generated.csv > 自动发现
```

自动生成只写 `manifest.generated.csv`，不会覆盖 `manifest.csv`。所有默认配置集中在 `urban_monitor.config.DEFAULT_CONFIG`；`project.yaml` 递归覆盖默认值。

## 2. Prepare

Prepare 读取所有影像和标签 CRS。影像 footprint 转到 WGS84 后计算 AOI，按中心点选择北半球 WGS84 UTM。超过 Phase-1 宽度限制或没有共同覆盖时停止。

参考影像按以下顺序选择：覆盖面积最大、nodata 最少、原始分辨率最高、日期最早。对齐工作副本采用参考影像转到分析 UTM 后的网格：

- RGB/连续影像：bilinear；
- quality/category mask：nearest；
- vector label：GeoPandas `to_crs`。

原始文件不修改。`reference_grid.json`、`crs_provenance.json` 和 `alignment_report.md` 记录选择与转换。

## 3. Ground Truth

geometry-only 矢量可直接进入 prepare。字段默认值、别名、make-valid、reject、裁剪、小对象过滤、MultiPolygon 与 holes 规则见根目录 `GROUND_TRUTH_SPEC.md`。

## 4. 空间划分

分析 CRS 下建立固定米制 block，固定 seed 选择 validation block。相同 block 在所有日期保持相同 split。tile 只有完全位于一个 block 内才保留；跨边界 tile 被排除，避免 overlap 产生共享像素。

输出 `spatial_split.gpkg`、`split_summary.csv`、`split_by_date.csv` 和 `split_map.png`。验证建筑少于 20 只 warning，不伪造验证结果。

## 5. Tile 与 COCO

GeoTIFF 使用 Rasterio window read。先枚举 window、空间 split 与 annotation 元数据，再只读取最终选中的 tile。建筑先按 tile 裁剪，`clipped_area/original_area` 为 visibility；低于阈值的碎片不导出。

Polygon、MultiPolygon 与 holes 栅格化为 COCO RLE。image/annotation ID 来自日期、窗口、object ID 和几何 hash，相同输入可复现。空 tile 按配置与 positive tile 比例确定性抽样。

输出：

```text
prepared/coco/
├── train/
├── val/
├── annotations_train.json
├── annotations_val.json
├── dataset_summary.json
└── previews/
```

## 6. 训练

模型是 torchvision Mask R-CNN ResNet50-FPN，box predictor 与 mask predictor 都替换为两类 head：background、building。

默认 `pretrained=false`，避免隐式网络下载。设置 `pretrained=true` 时，只有权重已缓存、提供本地 `weights_path`，或显式允许下载才继续。demo 固定不下载。

训练支持 CPU/CUDA、CUDA AMP、gradient clipping、固定 seed、Windows workers=0、epoch checkpoint、`last.pth`、按验证 F1 选择的 `best.pth`、optimizer/scheduler/epoch/best metric resume 和 early stopping。CUDA OOM 会给出减小 batch/tile 与 resume 建议。

## 7. Preflight 与验证

训练前生成 `reports/PREFLIGHT_REPORT.md`。缺 CRS、无图像、无标签、全部标签为空、train/val 为空、RGB band 越界、图像不可读或 COCO 无效会停止。

validation inference 只加载 `best.pth`。指标包括 precision、recall、F1、mean matched IoU、Dice、object count error、building area error；按 GT 面积三分位输出 small/medium/large。无 validation GT 时不报告数值。

输出 metrics JSON/CSV、object/size metrics、prediction/error/FP/FN GPKG，以及 10 good + 10 hard preview。

## 8. 完整影像推理

每期 prepared GeoTIFF 使用与训练相同 RGB bands，执行 overlap tiled inference、polygon NMS 和规则化。输出 raw、regularized 与 EPSG:4326 exchange GPKG。

每条预测包含 `prediction_id`、`score`、`source_date`、`model_version`、`checkpoint_hash`、`schema_version`。人工标签不会被覆盖。

## 9. Tracking、lineage 与 observation

Tracking 和 lineage 使用 STRtree/GeoPandas spatial index 先查询 expanded bounding-box 候选，再计算 IoU、buffer IoU、bbox IoU、centroid、area 和 shape。候选必须同时满足 centroid、空间 overlap 与正常面积比 gate。

assignment 在空间候选连通分量内运行 Hungarian，不建立全 AOI 密集比较。最优/次优 margin 同时从 previous 和 current 两个方向计算；低 margin 为 `ambiguous_match`。

observation status 与证据规则见 `GROUND_TRUTH_SPEC.md`。结构化置信度公式见 `CONFIDENCE_MODEL.md`。

## 10. Review 闭环

`apply_reviews` 保留 auto `state`，写 `state_final`、reviewer、review_time、review_version 和 lineage action。重复应用使用原 auto summary + review overlay，保持幂等。

student export 优先 `tracks_reviewed.gpkg`；状态优先级为 `state_final > reviewed_state > state`。rejected 或 review_status rejected 永久排除。

## 11. Cache 与 run 生命周期

stage hash 分为 data、training、evaluation、inference、tracking。未变化结果通过同卷 hardlink 复用，失败时回退为文件复制；不会使用 Windows 符号链接。

正式 run 含 `RUN_MANIFEST.json`、resolved config、日志、prepared、training、evaluation、predictions、tracking、review 和 reports。失败 run 记录 failure_stage、error_message、traceback_path。只有完整成功更新 `outputs/<aoi_id>/latest.json`。

`--force` 创建新 run 并强制重算；不带 force 时日志明确写 `reused`、`recomputed` 或 `invalidated`。

## 12. 已知限制

- 真实精度取决于当地影像、季节、建筑形态和 GT；synthetic demo 只验证 plumbing。
- 随机初始化 1 epoch demo 可能没有达到推理阈值的建筑，因此 tracking 可为空。
- RGB quality heuristic 不能替代供应商 QA 和人工审核。
- GCJ-02/BD-09 转换未内置；需外部可信转换与 provenance。
- lineage 是待审核候选，不是自动确认的物理重建事实。
