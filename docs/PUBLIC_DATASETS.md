# 公共变化数据集

本项目不复制、分发或自动下载任何第三方数据。使用前请从官方来源核对最新许可；论文的开放许可不自动等同于数据集许可。

## S2Looking

- 用途：侧视、高分辨率建筑新建/拆除变化。
- 官方字段：`Image1`、`Image2`、`label`、`label1`、`label2`。
- 本项目映射：`GT_NEW = label1`，`GT_DEM = label2`，统一 `GT_CD = GT_NEW OR GT_DEM`。
- 一致性：原始 `label` 与方向标签不一致的像素比例写入 `s2looking_inconsistent.csv`。
- 来源：[S2Looking 论文与下载说明](https://www.mdpi.com/2072-4292/13/24/5094)。
- 许可：本轮未在数据下载页确认独立数据许可。论文为开放访问并不自动授权数据商业使用；商业场景必须另行确认。

## SECOND

- 用途：中国城市语义变化检测。
- 规模说明与类别定义以 [SECOND 官方页面](https://captain-whu.github.io/SCD/) 为准。
- 本项目读取两期语义标签并生成：
  - `GT_CD = GT_T1 != GT_T2`
  - `GT_NEW = T1 != building AND T2 == building`
  - `GT_DEM = T1 == building AND T2 != building`
- 单通道类别 ID 可直接读取。RGB 标签必须由 `class_mapping_file` 显式转换；没有经确认的颜色表时立即停止。
- 输出类别分布、转移矩阵、映射记录和预览，避免错误映射进入训练。
- 许可：官方页面未提供可由本项目直接确认的宽泛商业授权，使用者必须自行核对下载条款。

## DSIFN

- 用途：双时相二值变化预训练。
- 本项目只使用二值 `GT_CD`。数据没有可靠方向监督，因此 `GT_NEW`、`GT_DEM` 保持缺失，绝不填成零。
- 来源：[DSIFN 官方代码库](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images)。
- 许可：官方说明限制为研究/非商业用途；商业使用前必须取得单独授权。

## ChangeMamba 预处理数据

[ChangeMamba 官方仓库](https://github.com/ChenHongruixuan/ChangeMamba)说明了 SECOND 的单通道转换与二值标签生成。ChangeMamba 代码使用 Apache-2.0，但这不改变原数据集许可。项目仅提供外部适配入口，不复制其代码、权重或数据。

## 配置示例

```yaml
datasets:
  s2looking:
    enabled: true
    root: D:/Datasets/S2Looking
  second:
    enabled: true
    root: D:/Datasets/SECOND
    building_class_id: 1
    class_mapping_file: D:/Datasets/SECOND/verified_colors.json
    mapping_verified: true
  dsifn:
    enabled: true
    root: D:/Datasets/DSIFN
```

路径只写在配置中，不写死在源码。`change-doctor` 和 `change-prepare` 会验证目录、文件一一对应、尺寸、标签可读性和 SECOND 映射条件。
