# Urban Renewal Change V2 架构

V2 把“一个模型等于一个城市更新结论”改成教师能力组合。每个教师只提供自己真实具备的证据，缺失教师对应输出保持 `PENDING`，不会填零或伪造。

## 教师职责

| 教师 | 角色 | 可用输出 |
|---|---|---|
| DSIFN | 任意地表变化 | any-change probability / mask |
| S2Looking | 建筑方向变化 | building-change、new-building、demolition probability |
| SECOND | 语义变化 | semantic T1/T2、transition；映射确认后才允许建筑方向 |
| RSBuilding | 建筑几何 | T1/T2 building probability、mask、boundary |

SECOND 的 `mapping_verified: false` 不是程序错误。此时 any-change 与两期 semantic 可训练，new/demolition 必须标记 `skipped_no_supervision`。

## 监督矩阵

| Head | DSIFN | S2Looking | SECOND |
|---|---:|---:|---:|
| any_change | 是 | 是 | 是 |
| new_building | 否 | 是 | 条件式 |
| demolition | 否 | 是 | 条件式 |
| semantic_t1 | 否 | 否 | 是 |
| semantic_t2 | 否 | 否 | 是 |

条件式仅表示 `mapping_verified=true` 且 `building_class_id` 已明确。不存在监督时不构造全零 GT。

## Checkpoint 合并

V2 bootstrap 以 DSIFN checkpoint 提供共享编码器、解码器、binary 与 uncertainty 参数；以 S2Looking checkpoint 提供 new/demolition head；semantic head 随机初始化。每个 tensor 都检查名称、shape 与 dtype，任何不匹配都会停止。

报告：`outputs/pretraining/change_v2/bootstrap/CHECKPOINT_MERGE_REPORT.json`

## GeoTIFF 推理

```powershell
python run.py urban-renewal-preflight

python run.py urban-renewal-infer `
  --t1 "D:/Data Set/Satellite/pilot_01/T1.tif" `
  --t2 "D:/Data Set/Satellite/pilot_01/T2.tif" `
  --output "D:/Data Set/Satellite/pilot_01/results" `
  --device cuda:0
```

推理按 window 读取，使用重叠 tile 与 Hann 权重融合。T2 网格不同会通过显式 WarpedVRT 对齐到 T1；输出保持 T1 的 CRS、transform、extent、分辨率和 nodata 语义。结果包含概率栅格、阈值 mask、候选 GPKG、`preview.png`、`index.html`、`MUDI_BASE_FEATURES.csv` 与 `INFERENCE_MANIFEST.json`。

分辨率解释：

- ≤1 m：可作为建筑对象 refinement；
- 1–5 m：建筑候选；
- >5 m：建筑群、地块或街区级候选，不宣称单体建筑精确边界。

## Test 隔离

`configs/V2_HOLDOUT_PROTOCOL.json` 将 SECOND official test 预留为 V2 holdout。开发阶段配置只允许读取 train/validation；DSIFN old test、S2Looking test 和 V2 holdout 的读数必须为 0。困难样本命令固定 `--split train`：

```powershell
python run.py change-mine-v2-hard-examples `
  --index path/to/public_change_index.csv `
  --split train
```

旧 DSIFN final test 不得再次运行。

