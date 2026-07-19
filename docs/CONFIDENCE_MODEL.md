# v0.4 结构化置信度模型

`confidence_score` 是用于排序和审核优先级的规则分数，范围为 `[0,1]`。它不是经过真实样本 calibration 的概率，不能解释为“事件有 80% 概率为真”。

## 公式

对每个分量先截断到 `[0,1]`：

```text
confidence_score = sum(weight_i * component_i) / sum(weight_i)
```

分量包括：

| 分量 | 含义 |
|---|---|
| `detection_score` | 轨迹内平均检测分数 |
| `valid_observations` | `present` 与 `usable_absent` 的有效观测数量，3 次达到满分 |
| `interval_width` | 状态变化区间越窄，分数越高；730 天及以上为 0 |
| `usable_fraction` | 各日期位置的平均可用比例 |
| `match_score` | 已匹配观测的平均空间匹配分数 |
| `ambiguity_margin` | 双向最优/次优 margin；0.20 及以上达到满分 |
| `registration` | 未发生 registration failed 的日期比例 |
| `off_nadir_consistency` | 离轴角差越小，分数越高；30° 差异为 0 |
| `canonical_geometry` | 使用可信 canonical tracking geometry 的观测比例 |
| `lineage_clarity` | 存在未审核 lineage 时降低该分量 |
| `manual_review` | 自动结果为 0.5，人工审核后为 1；rejected 最终分数为 0 |

默认权重和 high/medium/low 阈值集中在 `urban_monitor.config.DEFAULT_CONFIG["confidence"]`，可在 `project.yaml` 的 `confidence` 节覆盖。

## 等级

- `high`：分数不低于 0.75；
- `medium`：分数不低于 0.50；
- `low`：其余情况。

存在 `ambiguous_match` 或状态为 `uncertain` 时，自动分数会被限制在 medium 阈值以下。真实数据到位后，应使用独立审核集检查各分量和权重，再决定是否进行概率校准。
