# v0.3.0 实施前审计

审计日期：2026-07-18  
输入包：`urban-renewal-phase1-starter-v2 (1).zip`  
输入包 SHA-256：`438EF1ACF73BE1E6FEEFCCD597F1AC1120090EA3A0B37B1F88E2AC1B08741AB7`

## 1. 已实际实现

- 本地 `manifest.csv` 驱动的 4 期 synthetic 数据读取。
- 同 CRS、同 transform、同尺寸栅格的严格一致性检查。
- GeoPackage 建筑标注切片并导出 COCO。
- Mask R-CNN 训练/推理接口、mock prediction、保守多边形规则化。
- IoU、buffer IoU、bbox IoU、质心距离和面积相似度联合匹配。
- 短缺测重连、基本状态区间、merge/split 候选 lineage。
- RGB 启发式双波段质量掩膜。
- 静态 HTML 审核页、CSV 回写、10 m fraction 标签。

## 2. 文档声明但未完整实现

- 原始代码要求所有影像预先处于完全相同网格，不能诊断后重投影到 reference grid。
- 没有统一 schema、CRS、坐标编码和 provenance 模块。
- `manifest` 缺少 provider/manual mask、sensor、GSD、CRS、坐标编码、配准状态等真实数据字段。
- 质量掩膜未按 manual > provider > RGB > default 合成。
- 匹配没有最佳/次佳差距和 `ambiguous` 标记，shape 相似度未进入得分。
- lineage 为逐对象启发式扫描，不是显式二部图连通分量实现。
- 没有独立 observation matrix 文件、事件数据库汇总和空间 holdout 评价。
- 审核结果会合并写出，但缺少 reviewer、review_time、review_version 和严格幂等保护。
- student exporter 没有三个排除清单、reference raster 严格网格对齐和完整 GeoTIFF provenance tags。
- CLI 普遍没有 `--force`，也没有统一 run 目录和 `RUN_MANIFEST.json`。

## 3. 基线验证（修改前）

### pytest

- bundled Python 首次失败：`No module named pytest`。
- 系统 `python.exe` 失败：`指定的登录会话不存在`。
- 安装最小 CPU/GIS 依赖到工作区后，默认 pytest 临时目录报 `PermissionError: ... Temp\\pytest-of-ljz52`，结果为 `5 passed, 1 error`。
- 改用工作区 `--basetemp` 后：`6 passed, 2 warnings in 1.33s`。

### synthetic smoke

原始流程已真实完成：

- synthetic manifest；
- 4/4 影像对齐检查；
- COCO：14 images / 20 annotations；
- 4 期 mock predictions。

随后在 `regularize_predictions.py` 失败：

```text
ModuleNotFoundError: No module named 'torch'
```

根因是 `src/urban_monitor/inference.py` 顶层导入 Torch；仅使用几何规则化函数也被迫依赖深度学习栈。压缩包自带的后续 tracking/review/student 输出不能作为本次 smoke 成功证据，因为它们在本轮命令开始前已经存在。

## 4. 中国真实影像接入风险

- 输入 CRS 缺失时当前代码只会在部分路径中失败，没有一致、可审计的拒绝策略。
- 米制 buffer/面积逻辑依赖调用者先提供投影数据，未阻止 EPSG:4326 或 EPSG:3857。
- 高德/腾讯 GCJ-02、百度 BD-09 和 unknown 标注没有准入阻断。
- 高层屋顶视差与整体地面配准尚无成对诊断报告。
- provider QA、人工遮挡和 registration issue 不能合成为统一质量证据。
- CSV 未统一使用 UTF-8 BOM，中文 Windows/Excel 流程不稳定。

## 5. 遗留配置与硬编码

- 未发现 Cape Town、Johannesburg、South Africa、南半球 EPSG:327xx 或个人绝对路径。
- synthetic 数据写死 `EPSG:32650`，作为北京演示合理，但核心代码尚不能根据 AOI 自动选本地 UTM。
- 测试使用 `EPSG:3857` 做正式面积/距离，和 v0.3 约束冲突，需替换为本地 UTM 并增加拒绝测试。
- 脚本路径均由配置或参数提供；输出结构仍是旧的 `outputs/<aoi_id>/...`。

## 6. 中文路径现状

- 代码普遍使用 `pathlib.Path`，是良好基础。
- 未有中文路径、空格路径、UTF-8 BOM CSV、中文审核备注的自动测试。
- 多处 `read_csv/to_csv` 未显式指定 `utf-8-sig`。

## 7. 本轮修改范围

- 扩展现有 config/manifest/alignment/quality/geometry/tracking/review/downscale，保留旧调用接口。
- 新增最小职责模块：`schema.py`、`crs.py`、`coordinate_encoding.py`、`alignment_diagnostics.py`、`evaluation.py`、`provenance.py`、`providers/local_raster.py`。
- 新增统一 schema/CRS/alignment/evaluation/smoke CLI 和不少于 30 项 CPU 测试。
- 延迟导入 Torch，使 synthetic smoke 不需 CUDA、Torch 或权重。
- 更新 v0.3 配置、全部交付文档和 changelog。

## 8. 必须兼容的接口

- `load_config()`、`resolve_path()`、`load_manifest()` 和 `DateRecord` 现有字段。
- `validate_alignment()` 的严格检查模式。
- `regularize_building_polygon()`、`regularize_prediction_frame()`。
- `assign_tracks()`、`summarize_tracks()`、`detect_lineage_events()` 的现有参数与返回值。
- `build_review_queue()`、`apply_reviews()`。
- `export_student_change_labels()` 的现有参数；仅增加可选参数。
- 现有逐脚本 workflow 和 `configs/example.yaml`。

## 9. 实施门槛

每一层新增逻辑必须先有对应 CPU 测试。最终 smoke 必须使用全新的 run 目录，逐项核对要求的输出，避免把压缩包预置结果误判为本轮生成。
