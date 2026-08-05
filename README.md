# 城市更新遥感监测平台

本仓库有两套独立教师：公共双时相变化教师，以及个人小片区建筑实例教师。原始数据不会被修改，也不要提交到 Git。

## 项目总览与权威清单

项目现已使用统一注册表管理模型、数据集、历史实验和可视化资产。先运行：

```powershell
python run.py project-status
python run.py model-list
python run.py dataset-list
python run.py build-gallery
```

本地状态页位于 `outputs/catalog/project_dashboard.html`，可视化库位于
`outputs/catalog/visual_gallery.html`。Full20 当前是 `validation_candidate`，
不是经过独立 Test 证明的最终模型；北京通州输出是 candidate
building-change objects，不是确认建筑统计。

架构与安全边界见 [项目概览](docs/PROJECT_OVERVIEW.md)、
[Test 安全策略](docs/TEST_SAFETY_POLICY.md) 和
[数据集接入指南](docs/DATASET_ONBOARDING.md)。本轮真实核验结果见
[注册与架构重构验收](docs/PROJECT_REGISTRY_ACCEPTANCE.md)。

## 路线 A：公共变化教师

数据位置：

```text
D:\Data Set\S2Looking
D:\Data Set\SECOND
D:\Data Set\DSIFN
```

先使用 Windows 配置和体检：

```powershell
python run.py change-preflight --config configs/change_public_windows.yaml
python run.py change-all --config configs/change_public_windows.yaml --device cuda:0 --quick
python run.py change-all --config configs/change_public_windows.yaml --device cuda:0
```

正式训练强制 CUDA；CPU 只允许 `python run.py change-demo --device cpu --force`。SECOND 的建筑类别和 RGB 映射绝不自动猜测。详细步骤见 [公共训练指南](docs/01_PUBLIC_DATASET_TRAINING_WINDOWS.md)。

## 路线 B：个人小片区建筑教师

默认仓库内目录为 `data/beijing_demo/`；也可将 AOI 数据保留在外部磁盘并通过 `--data-root` 指定。

数据位置：

```text
D:\Data Set\Personal\nanjing_pilot\
├── images\2022-04-01.tif
├── labels\2022-04-01.gpkg
└── project.yaml                 # 可选
```

每期只需一个同日期 GPKG，优先图层名 `buildings`，且可以只含 `geometry`。运行：

```powershell
python run.py personal-preflight --aoi nanjing_pilot --data-root "D:/Data Set/Personal"
python run.py all --aoi nanjing_pilot --data-root "D:/Data Set/Personal"
```

默认仍是仓库内 `data/<aoi_id>`，因此旧命令 `python run.py all --aoi beijing_demo` 保持有效。详细标注流程见 [QGIS + GeoOSAM 指南](docs/02_PERSONAL_QGIS_GEOOSAM_WORKFLOW.md)。

## 安装

先在 PowerShell 执行 `nvidia-smi`。CUDA 用户按照 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与驱动匹配的 Torch/torchvision，确认 CUDA 后再安装其余项目依赖，避免 pip 覆盖 CUDA Torch：

```powershell
py -3.11 -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 按 PyTorch 官方选择器安装 torch 与 torchvision
pip install -e . --no-deps
pip install PyYAML numpy pandas rasterio geopandas shapely pyproj scipy Pillow opencv-python-headless scikit-image pycocotools tqdm tensorboard
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

CPU 或 demo 环境可使用 `pip install -e ".[train]"`。

## 输出

- 公共教师：`outputs/pretraining/change/latest.json`、`checkpoints/`、`metrics/`、`training_dashboard.html`、`review/index.html`。
- 个人 AOI：`outputs/<aoi_id>/latest.json`、`training/checkpoints/best.pth`、`evaluation/`、`predictions/`、`tracking/`、`review/`。

## Windows 脚本

`scripts/windows/` 提供环境检查、公共 preflight/quick/full 与个人 preflight/all。所有脚本在仓库根目录运行，路径含空格已引用，且不会自动删除输出或启动 full 训练。

## 常见 blocker

- **SECOND 映射未确认**：填写真实 `building_class_id`、RGB `class_mapping_file`，再将 `mapping_verified: true`。
- **没有 validation**：默认不使用 test；仅临时设置 `use_test_as_validation: true` 才可继续，并会写入报告。
- **GPKG 多图层**：重命名建筑图层为 `buildings`，或在 `project.yaml` 设置 `data.building_layer`。
- **CUDA OOM**：auto 只尝试 16、8、4、2、1；仍失败请降低 `patch_size`。
## V2 多教师城市更新推理

V2 将 DSIFN 的任意变化、S2Looking 的建筑方向变化、可选 SECOND 语义变化和可选 RSBuilding 建筑几何组合起来。缺失教师不会阻断已有能力，缺失结果会显示 `PENDING`。

```powershell
python run.py urban-renewal-preflight
python run.py urban-renewal-infer --t1 "D:/Data Set/Satellite/pilot_01/T1.tif" --t2 "D:/Data Set/Satellite/pilot_01/T2.tif" --output "D:/Data Set/Satellite/pilot_01/results" --device cuda:0
```

架构、监督矩阵、GeoTIFF 输出和 test 隔离规则见 [V2 架构说明](docs/V2_ARCHITECTURE.md)。
