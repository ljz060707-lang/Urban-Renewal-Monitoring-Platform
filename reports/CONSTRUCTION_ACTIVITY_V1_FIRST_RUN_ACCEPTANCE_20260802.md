# Construction Activity V1 — 首次实施与验收报告

生成日期：2026-08-02  
模块：`construction_activity_v1`  
发布范围：`experimental_candidate`  
科学状态：`candidate_not_ground_truth`

## 1. 数据审计

### UCS / URS

- resolved root：`D:\Data Set\URS`
- package roots：
  - `600 manually delineated UCS labels`
  - `600 manually delineated UCS labels part2`
- 数据记录：官方主记录 `10.5281/zenodo.17404901`；本地文件与较新的记录 `18997896` 元数据校验一致。
- 两个 package 目录各有 386 个 TIFF 和 600 个 PNG。
- 去重后：386 个唯一影像、600 个唯一标签、386 对、214 个无影像标签。
- 重复：386 个影像 stem、600 个标签 stem；重复文件无 hash 冲突。
- 类别：background、exposed areas、DPN-covered areas、Building。
- 标签编码：indexed/palette PNG，值域 0–3。
- 影像样本：512×512、4 bands、int16、EPSG:32650、约 1 m。
- RGB band semantics：未在本地包中明确。
- dataset license：Zenodo 审计元数据中未声明；不把论文 CC BY 4.0 自动外推为数据许可证。
- pairing gate：FAIL。
- Test：项目空间 split 产生 46 个 locked test 记录；pixel reads = 0。

### Split

- 策略：GeoTIFF centroid 形成空间 group，seed `20260729`。
- train / validation / locked test：296 / 44 / 46。
- image hash overlap：0。
- group overlap：0。
- 隔离强度：`moderate_unbuffered_block_split`。

### CWLD

- resolved root：`D:\Data Set\CWLD`
- 复用既有 RGBA distance codec 与审计 manifest。
- 原生类别独立保留：background、vacant landfillable、engineering facility、waste dumping。
- 禁止解释：engineering facility ≠ active construction。
- Test reads：0。

最新完整审计：
`outputs/urrs/audits/construction_activity_20260802_111842/`

## 2. 模型与科学标签

已实现：

- shared RGB + deterministic grayscale texture encoder；
- UCS native head；
- CWLD native head；
- unified evidence projector + class-presence mask；
- boundary head；
- uncertainty head；
- dataset-specific CE/Dice/Boundary/Uncertainty loss；
- parcel feature schema 与保守场景规则；
- metric CRS area audit、vector repair、raster/vector discrepancy；
- Construction TFR（T=2/T=4、缺失时相、非等间隔）；
- ConstructionNoMTI；
- ConstructionMTI 接口（默认 disabled，`experimental_unselected`）；
- state graph、pseudo label provenance、Hungarian candidate matching、split/merge；
- 生命周期与 Tongzhou / national-screening 管线门禁接口。

正式指标只允许 `official_ground_truth`、`human_verified`、`human_corrected`。
`pseudo_rule` 与 `model_prediction` 不能进入正式 Validation。

## 3. 已运行验收

### 定向测试

```text
python -m pytest -q tests/unit/construction --basetemp .pytest-construction-v1-final
14 passed in 3.47s
```

### Dataloader smoke

- 状态：PASS。
- 读取 train 样本：4。
- image/target：3×128×128 / 128×128。
- image range：[0,1]。
- Test pixel reads：0。

### 20-step training smoke

- 设备：CPU（检测到 GPU 已有占用，未争用）。
- optimizer steps：20。
- loss first / last：2.3030059 / 1.8954855。
- probability range：[0.0291290, 0.7332211]。
- NaN / Inf：无。
- checkpoint：`outputs/training/construction_activity_v1/smoke_20260802_105901/checkpoints/smoke_checkpoint.pth`
- SHA256：`628d04ba2471954a5327cf7ee20ad6a6276469f9ad2d2607666f53a682b032d7`
- registerable：false。
- formal Validation：NOT RUN。

## 4. 审核闭环

现有 URRS Tongzhou Review Console 已新增独立“施工状态与生命周期”标签页：

- 固定 16 条跨四期 review track；
- 默认状态 `unknown_state`、转换 `uncertain`；
- 明示 `PSEUDO LABEL — NOT GROUND TRUTH`；
- 支持两种对象、四期状态、三段转换、更新/扩张/修缮候选、边界审核、置信度与备注；
- 人工确认 → `human_verified`；人工修改 → `human_corrected`；
- 独立 CSV / JSONL / state，不写原 Geometry 审核文件。
- QGIS 导出：`construction_activity/REVIEW_CORRECTIONS.gpkg`，6,741 个 RSBuilding building candidates，非施工 Ground Truth。

HTTP/API 验收：根页面、JS、construction inventory 与首张预览均返回 200；inventory 16 条。
应用内浏览器连接因 Windows sandbox helper 故障未完成视觉自动化，因此状态为：
`automatic_http_qa=pass`、`browser_visual_qa=blocked_by_tooling`、`human_visual_review=pending`。

## 5. 安全复核

- current URRS release：UNCHANGED。
- selected temporal integration：no-MTI。
- ConstructionMTI：EXPERIMENTAL UNSELECTED。
- Full20 SHA256：`70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621`。
- RSBuilding iter_2000 SHA256：`f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7`。
- UCS Test reads：0。
- CWLD Test reads：0。
- WHU Test reads：0。
- SECOND Test reads：0。
- DSIFN/S2 Test：NOT RUN。
- 原始数据修改：0。
- 删除文件：0。

## 6. Gate 与未完成项

| Gate / 阶段 | 真实状态 |
|---|---|
| CONSTRUCTION_DATA_READY | true（部分 paired 数据） |
| CONSTRUCTION_LABELS_READY | true |
| pairing_pass | false |
| license_verified | false |
| split_leakage_pass | true |
| development_smoke_allowed | true |
| formal_ucs_training_allowed | false |
| CONSTRUCTION_PIXEL_MODEL_READY | false |
| CONSTRUCTION_BOUNDARY_READY | false |
| CONSTRUCTION_PARCEL_READY | PENDING HUMAN LABELS |
| CONSTRUCTION_TEMPORAL_LABELS_READY | PENDING HUMAN LABELS |
| CONSTRUCTION_TFR_READY | NOT RUN |
| CONSTRUCTION_MTI_EVALUATED | EXPERIMENTAL UNSELECTED |
| CONSTRUCTION_TRACKING_READY | NOT RUN |
| CONSTRUCTION_TONGZHOU_READY | NOT RUN |
| CONSTRUCTION_NATIONAL_SCREENING_READY | PENDING DATA |

正式 UCS / joint 训练没有启动。要解除阻断，需提供 214 张可与现有标签 stem 匹配的真实影像、明确的数据许可证，以及可审计的 RGB band 语义；不能用重复目录替代缺失影像。

## 7. 可复制命令

```powershell
cd "C:\Users\ljz52\Documents\Codex\2026-07-18\urban-renewal-phase1-starter-v0-2\outputs\urban-renewal-phase1-starter"

.\.venv-gpu\Scripts\python.exe tools\urrs.py construction audit `
  --data-root "D:\Data Set\URS" `
  --data-root "D:\Data Set\CWLD"

.\.venv-gpu\Scripts\python.exe tools\urrs.py construction status

.\scripts\urrs_construction.ps1 -Stage review
```

