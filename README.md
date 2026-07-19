# 城市更新遥感监测平台

把多期高分辨率影像和建筑 Ground Truth 放入指定文件夹，运行一条命令，即可训练建筑识别模型并生成跨期城市更新结果。

最短流程：

```text
数据：data/<aoi_id>/images/ 和 data/<aoi_id>/labels/
安装：pip install -e ".[train]"
运行：python run.py all --aoi beijing_demo
结果：outputs/beijing_demo/latest.json 指向最新成功 run
```

不需要手写 manifest，也不需要逐个执行内部脚本。

## 1. 安装

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[train]"
```

Linux / WSL：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[train]"
```

CPU 环境可直接使用上面的命令。需要 CUDA 时，先按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 安装与本机 CUDA 匹配的 Torch/torchvision，再执行：

```bash
pip install -e .
pip install tqdm tensorboard
pip install -e ".[train]" --no-deps
```

这样不会用通用 wheel 覆盖已经安装的 CUDA Torch。本项目不要求 Conda，也不把 Docker 作为唯一运行方式。

## 2. 放入数据

只要求 `images` 和 `labels` 两个目录：

```text
data/beijing_demo/
├── images/
│   ├── 2022-04-01.tif
│   ├── 2023-04-01.tif
│   └── 2024-04-01.tif
└── labels/
    ├── 2022-04-01.gpkg
    ├── 2023-04-01.gpkg
    └── 2024-04-01.gpkg
```

- 影像和标签必须使用相同日期命名。
- 日期支持 `YYYY-MM-DD`、`YYYY_MM_DD`、`YYYYMMDD`。
- 影像支持 GeoTIFF；标签支持 GPKG、GeoJSON、Shapefile，推荐 GPKG。
- 标签可以只有 geometry，系统自动补齐训练字段。
- 影像和标签必须带真实 CRS。
- 不要把高德 GCJ-02 或百度 BD-09 图层直接当作 WGS84 Ground Truth。
- 同一天有多个影像或多个标签时，系统会列出冲突文件并停止。

可选的 `data/<aoi_id>/project.yaml` 只用于覆盖默认值。例如：

```yaml
project:
  name: 北京示例片区

data:
  rgb_bands: [1, 2, 3]

training:
  epochs: 30
```

不写 `project.yaml` 也能运行。默认不下载预训练权重，使用随机初始化；如需预训练权重，请在项目配置中显式设置。

## 3. 检查环境

```bash
python run.py doctor
```

命令会检查 Python、Torch、Torchvision、CUDA、Rasterio、GeoPandas、Shapely、OpenCV、中文路径、写权限和磁盘空间。异常时会给出缺失项和安装命令。

## 4. 一键训练

```bash
python run.py all --aoi beijing_demo
```

系统依次完成环境检查、文件发现、manifest、CRS/对齐、GT 校验、空间划分、切片、COCO、训练、验证、完整影像推理、tracking、lineage、review 页面和报告。

流程失败会立即停止并返回非零退出码。失败 run 会保留错误阶段和 traceback，但不会更新 `latest.json`。

## 5. 查看结果

先打开：

```text
outputs/beijing_demo/latest.json
```

其中 `run_directory` 指向本次完整结果。常用文件：

```text
training/checkpoints/best.pth
training/checkpoints/last.pth
evaluation/evaluation_summary.md
evaluation/metrics.json
predictions/
tracking/
review/index.html
reports/PREFLIGHT_REPORT.md
RUN_MANIFEST.json
```

验证与完整影像推理默认都使用 `best.pth`，不会自动改用 `last.pth`。

## 6. 常用命令

```bash
python run.py doctor
python run.py prepare --aoi beijing_demo
python run.py train --aoi beijing_demo
python run.py train --aoi beijing_demo --resume
python run.py predict --aoi beijing_demo
python run.py all --aoi beijing_demo --force
python run.py demo
```

- 输入未变化时会显示 `reused`。
- 标签变化会重做 GT、split、tiling 和 COCO。
- 训练配置变化会重训。
- 只改推理阈值不会重训。
- `--force` 强制重算对应完整流程。

## 7. 常见错误

- **CRS 缺失**：在 QGIS 中确认真实 CRS 后重新导出，不要随意“指定为 WGS84”。
- **日期不对应**：检查影像与标签的文件名；日期必须一一对应。
- **标签不在影像范围**：检查 CRS、坐标单位和标注位置；完全在外的对象会被拒收。
- **CUDA 内存不足**：减小 `training.batch_size` 或 `tiling.tile_size`，再使用 `--resume`。
- **没有安装 Torch**：执行 `pip install -e ".[train]"`；CUDA 用户按上面的官方步骤安装。
- **GCJ-02 / BD-09**：高德、百度坐标不能直接作为正式训练坐标。
- **Windows 中文路径**：项目已测试 UTF-8 中文路径；若第三方 GDAL 环境仍失败，先缩短路径并运行 `doctor`。
- **验证集对象过少**：系统会 warning；不能把训练集结果冒充验证指标，应扩大 AOI 或增加标注。

## 8. 数据要求

- 建议 0.3—2 m GeoTIFF；
- Polygon / MultiPolygon 建筑 Ground Truth；
- 建议 3—8 期影像；
- 默认训练 `building`；
- 建议按影像中可见屋顶边界标注；
- courtyard、interior rings 和 MultiPolygon 会保留；
- 点、线、空几何、无法修复对象和影像外对象会进入 rejected 报告。

技术细节、字段、空间划分、置信度、追踪与 provenance 见 [docs/TECHNICAL_REFERENCE.md](docs/TECHNICAL_REFERENCE.md)、[GROUND_TRUTH_SPEC.md](GROUND_TRUTH_SPEC.md) 和 [docs/CONFIDENCE_MODEL.md](docs/CONFIDENCE_MODEL.md)。
