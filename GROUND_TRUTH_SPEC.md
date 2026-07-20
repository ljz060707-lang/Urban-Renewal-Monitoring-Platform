# Ground Truth 规范（v0.4.0）

## 1. 最低要求

每个日期准备一个带真实 CRS 的 GPKG、GeoJSON 或 Shapefile。最低可只有 geometry：

```text
geometry
```

geometry 必须是 Polygon 或 MultiPolygon。推荐标注影像中实际可见的屋顶边界，不要把阴影、树木、脚手架、吊车或停车场并入完成建筑。

## 2. 自动补齐字段

geometry-only 文件在 prepare 时自动补齐：

| 字段 | 默认值 |
|---|---|
| `class_name` | `building` |
| `geometry_role` | `visible_roof` |
| `review_status` | `reviewed` |
| `source_date` | 从文件名解析 |
| `object_id` | 日期与规范化几何的稳定 SHA-1 ID |
| `schema_version` | `0.4.0` |
| `coordinate_encoding` | 根据分析 CRS 为 `projected` 或 `wgs84` |
| `notes` | 空字符串 |
| `tracking_geometry_wkt` | 空字符串 |

自动 ID 在相同日期、相同标准化几何下可复现。

## 3. 字段别名

- `class`、`label`、`category`、`类别`、`类型` → `class_name`；
- `date`、`time`、`日期`、`影像日期` → `source_date`；
- `id`、`fid`、`ID`、`编号` → `object_id`。

每个文件实际采用的映射写入 `data_report/field_mapping.json`，不会静默转换。属性中的日期若与文件名日期不一致会停止。

## 4. 类别

默认只训练 `building`。可在 `project.yaml` 中覆盖：

```yaml
data:
  classes: [building]
```

未选类别进入 rejected 报告，不进入 COCO。

## 5. 几何处理

处理顺序：

1. 拒绝缺失 CRS；
2. 重投影工作副本到分析 CRS；
3. 空几何、点、线进入 `rejected_labels.gpkg`；
4. 无效 Polygon/MultiPolygon 先 `make_valid`；
5. 修复后仍无 polygonal geometry 的对象进入 rejected；
6. 完全在影像外的对象进入 rejected；
7. 部分超范围对象裁剪到影像范围并记录 `clipped_to_image`；
8. 小于 `data.min_building_area_m2` 的对象进入 `filtered_small_objects.gpkg`；
9. 其余对象进入 `accepted_labels.gpkg` 和 prepared 日期标签。

MultiPolygon 的所有部分都会保留；courtyard 和 interior rings 不会被静默移除。COCO 使用 RLE mask 表达 holes。

## 6. 屋顶与地面 footprint

- `visible_roof`：默认分割监督目标；
- `ground_footprint`：有可信地籍或人工依据时使用；
- `canonical_tracking_anchor`：仅用于跨日期身份匹配。

可在 `tracking_geometry_wkt` 放入可信 canonical geometry。不要只因影像倾斜就猜测地面 footprint。

## 7. 审核与缺失证据

日期级审核支持：

- `usable`；
- `unusable`；
- `occluded`；
- `registration_issue`。

只有 `present` 和满足覆盖、可用比例、配准与无歧义条件的 `usable_absent` 能进入状态推断。其他未检测不能作为拆除或新建证据。

轨迹最终状态优先级：

```text
state_final > reviewed_state > state
```

`rejected` 轨迹绝不进入学生标签。

## 8. 坐标安全

- 原始文件 CRS 保留不变；
- 分析使用自动选择的本地 WGS84 UTM；
- 交换副本使用 EPSG:4326；
- 网页显示可使用 EPSG:3857。

GCJ-02、BD-09 或 unknown 坐标只能作为 review 参考，完成可验证转换并记录 provenance 前不得进入训练或正式面积/距离计算。

## 9. 输出报告

```text
outputs/<aoi_id>/runs/<run_id>/data_report/
├── accepted_labels.gpkg
├── rejected_labels.gpkg
├── filtered_small_objects.gpkg
├── field_mapping.json
└── label_validation_report.md
```

## 10. v0.5 双时相变化监督

公共变化索引使用 `has_binary`、`has_directional`、`has_semantic`。缺失任务路径为空且 flag 为 false，不得写入全零替代标签。

- S2Looking：`GT_NEW=label1`，`GT_DEM=label2`，`GT_CD=GT_NEW OR GT_DEM`。
- SECOND：两期语义 ID 必须来自经确认的类别映射；按明确 building class 生成 new/demolition。
- DSIFN：仅有 `GT_CD`，不能根据未知时间方向推断 new/demolition。

本地逐期 GPKG 差分结果只能标记 `requires_human_review`。人工确认并进入 reviewed index 后，才可作为微调监督。人工修正必须保留原标签 hash、修正 hash、审核人、时间、备注和版本。`exclude`、`misregistration`、未修正 `label_error` 不进入重训；`rejected` 不进入最终学生标签。
