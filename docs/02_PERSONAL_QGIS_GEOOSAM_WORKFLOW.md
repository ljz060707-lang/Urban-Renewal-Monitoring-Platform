# 个人小片区训练集：QGIS 与 GeoOSAM 零基础标注指南

GeoOSAM 只用于快速提出候选边界；每个结果都必须人工检查。每期提交一个完整建筑图层，不要单独导出 new、demolition、tracking 或 quality 文件。

## 1. 目录与命名

```text
D:\Data Set\Personal\nanjing_pilot\
├── images\2022-04-01.tif
├── images\2023-04-01.tif
├── labels\2022-04-01.gpkg
└── labels\2023-04-01.gpkg
```

影像和标签都使用 `YYYY-MM-DD`。每个 GPKG 最低只有一个 `buildings` 图层，几何为 Polygon/MultiPolygon；字段可为空，仅 geometry 即可。

## 2. 第一期完整标注

1. 安装 QGIS，打开“插件 → 管理并安装插件”，搜索 GeoOSAM，并打开其面板。
2. 添加 `2022-04-01.tif`，右键查看影像真实 CRS。
3. 创建 GeoPackage 图层，文件为 `labels/2022-04-01.gpkg`，图层名必须是 `buildings`，类型 Polygon，CRS 与影像一致。
4. 用点提示或框选提示生成候选，接受后逐一修正并保存。

标注可见屋顶外轮廓：不含阴影、树冠、道路、停车场；可明确分开的建筑分别画。第一期标注研究区内全部可见建筑，不只画变化建筑。

## 3. 后续日期增量标注

复制上一期 GPKG，改名为当期日期；加载新影像；保留稳定建筑、删除拆除建筑、用 GeoOSAM 添加新建筑、修正轮廓后保存。最终 GPKG 仍必须包含该日期全部可见建筑。

在“处理工具箱 → 修复几何”检查空几何、自相交、极小碎面和明显重叠。

## 4. 检查与运行

```powershell
python run.py personal-preflight `
  --aoi nanjing_pilot `
  --data-root "D:/Data Set/Personal"

python run.py prepare --aoi nanjing_pilot --data-root "D:/Data Set/Personal"
python run.py train --aoi nanjing_pilot --data-root "D:/Data Set/Personal"
python run.py all --aoi nanjing_pilot --data-root "D:/Data Set/Personal"
```

检查清单：

- [ ] 同日期 TIFF/GPKG 成对存在
- [ ] 每期一个 GPKG、建筑图层为 `buildings`
- [ ] Polygon/MultiPolygon 且 CRS 真实存在
- [ ] 每期包含全部可见建筑
- [ ] 阴影、树木、车辆没有标入建筑
- [ ] 几何已保存并位于正确 labels 目录
