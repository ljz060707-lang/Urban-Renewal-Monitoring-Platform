# v0.3.0 China Real-data Hardening 实施报告

## 1. 修改摘要

在 v0.2 教师模型、规则化、追踪、审核和下采样接口上增量加固；没有重建平行项目。新增输入准入、CRS 安全、对齐诊断、质量证据融合、确定性追踪、复杂 lineage、审核 provenance、MMU 审计、评价和统一 smoke run。

## 2. 文件级修改

- 新增：`schema.py`、`crs.py`、`coordinate_encoding.py`、`alignment_diagnostics.py`、`evaluation.py`、`provenance.py`、`providers/local_raster.py`。
- 修改：config、manifest、geometry、inference、tracking、quality、review、downscale。
- 新增 schema/CRS/alignment/evaluation/smoke CLI；全部原有 CLI 接受 `--force`。
- 新增 10 个测试文件，总测试数由 6 增至 51。
- 更新 README、实施规范、Ground Truth、robustness、sources、config、changelog。

## 3. 中国场景适配

兼容合法取得的 GF、商业高分、航空/UAV、地方正射和本地历史 GeoTIFF/COG；核心只依赖本地 manifest。商业数据和模型权重不被视为免费资源，也不会自动下载。

## 4. 坐标系方案

原始层保留文件 CRS；分析层按 AOI 中心选择 WGS84 UTM 326xx；交换层 EPSG:4326；显示层 EPSG:3857。北京/上海/成都通过 32650/32651/32648 自动选择测试。超过 100 km 的 Phase-1 AOI被拒绝并要求拆分。

## 5. 网络底图风险

`gcj02`、`bd09`、`unknown` 不能直接进入训练或米制分析；不能用 `set_crs("EPSG:4326")` 伪装。只记录经验证的转换 provenance，不内置来源不明的转换公式。

## 6. Manifest schema

支持 date/image/quality/provider/manual mask、sensor/source/GSD/off-nadir/sun/season/CRS/coordinate encoding/acquisition time/registration status/notes。旧 manifest 可读取并产生兼容 warning。

## 7. Annotation schema

区分 visible roof、ground footprint、canonical tracking anchor；包含 object/class/review/date/notes/tracking WKT/source/model/schema 字段，拒绝缺失 CRS 与坏 WKT。

## 8. Prediction schema

包含 prediction ID、`[0,1]` score、model version、schema version 和有 CRS 的有效几何。

## 9. Tracking schema

长表保存 present/usable、raw/buffer/bbox IoU、centroid、area、shape、ambiguity margin、匹配来源、registration status、schema version。

## 10. Lineage schema

事件包含稳定 hash event ID、日期、父/子 IDs、数量、双向 coverage、review status、schema version 和几何；支持 merge、split、complex redevelopment。

## 11. 匹配公式

候选门控使用 raw IoU、buffer IoU、bbox IoU 或最大质心距离；相似度为配置权重下的 IoU、buffer IoU、bbox IoU、质心、面积和 shape 相似度加权平均。最优与次优差值低于阈值标记 ambiguous。

## 12. Lineage 算法

相邻日期建立旧/新多边形二部重叠图并遍历连通分量：N→1 为 merge，1→N 为 split，N→M 为 complex。merge child 使用新 track ID，父轨迹终止，关系保存在事件图中。

## 13. 状态机规则

仅 usable 日期支持 absence。短 Present-Absent-Present 且几何相似视为 seasonal occlusion；明显面积/几何改变视为 construction candidate；长或未修复非单调序列为 uncertain。新建/拆除只报告日期区间。

## 14. 质量掩膜优先级

manual > provider QA/UDM > RGB heuristic > default。输出 band 1 usable、band 2 reason code；registration issue/occluded 不支持 absence 推断。

## 15. 人工审核

HTML 展示各日期、检测/代表轮廓、off-nadir、season、date quality 和 lineage。审核人使用页面内输入框；CSV 包含 reviewer/time/version/中文 notes。`apply_reviews` 写 `state_final`，不改 auto state 或原始 annotations，重复应用不增加记录。

## 16. MMU 与学生标签

默认 200 m²、10 m、4 倍 supersampling；输出 new/demolition/construction/any-change fraction 且保证 `[0,1]`。199/201 m² 已测试。缺日期、低置信度、低于 MMU 分别进入三个 CSV；事件仍留在高分数据库。GeoTIFF 包含 model/schema/MMU/resolution/AOI/CRS/time/run tags。

## 17. 实际测试命令

```powershell
$env:PYTHONPATH="<workspace>\\work\\python-packages;src"
python -m pytest -q --basetemp <workspace>\\work\\pytest-temp-v030
python scripts/run_phase1_smoke.py --config configs/example.yaml --output-root outputs --run-id v030_smoke_01 --force
```

## 18. 实际测试结果

- 修改前：默认 Temp 下 5 passed / 1 environment error；工作区 basetemp 下原始 6 passed。
- 修改后：51 passed，无失败；9 条 warning 均来自 Rasterio 1.5 与 NumPy 2.5 的已知 shape 弃用提示。
- 第一次 v0.3 smoke 在隐式 `tabulate` 依赖处失败并记录 failed manifest；改用内置 Markdown 格式化后同一 run 恢复成功。
- Browser 渲染验证发现原生 `prompt()` 在应用内浏览器不受支持；改为页面内审核人输入后，CSV 导出包含状态、日期质量、审核人和中文备注，console 无新增错误。

## 19. Smoke 输出

`outputs/demo_aoi/runs/v030_smoke_01/`：全部验收文件、14 COCO images、20 annotations、6 tracks、1 merge lineage、1 student-label date 和成功 `RUN_MANIFEST.json`。

## 20. 已知限制

- 未用真实中国影像训练或评估；synthetic 1.0 只验证 plumbing。
- 真实 Mask R-CNN 未安装 Torch/CUDA 或下载权重。
- RGB QA 不能替代供应方 QA 和人工判断。
- GCJ-02/BD-09 转换未实现，须由可验证外部流程完成。
- tracking/event/grid 专用 GT 未提供，因此明确写 `not computable`。

## 21. 首批真实影像接入

按 README 清单建立 ASCII AOI ID，放入 3–8 期合法 GeoTIFF，核对 CRS，编写 v0.3 manifest，验证 schema/CRS/alignment，统一 reference grid，融合 QA，QGIS 标注，再执行 COCO/训练/推理/规则化/追踪/lineage/state/review 回写/MMU/evaluation。原始数据和 annotation 不变，人工决定作为 overlay。

## 22. RSBuilding 接口

继续以 COCO/GPKG/manifest 为文件契约；后续 adapter 只负责第二教师输入输出，不把 OpenMMLab 或 provider 逻辑写入 tracking/state。

## 23. ContUrbanCD 接口

使用 observation matrix、reviewed event inventory 和 reference-aligned fractional grids 作为 Phase 2 输入；本轮不训练完整 ContUrbanCD。

## 24. 未完成事项

真实 AOI 数据授权、人工 Ground Truth、CUDA 教师训练、独立空间 holdout、真实 event/tracking GT 和 Phase-2 学生模型需要用户数据与计算环境，未被伪造为已完成。
