# 公共数据集训练：Windows 与 RTX 显卡零基础指南

支持 S2Looking（二值与新建/拆除）、SECOND（二值、语义与方向）和 DSIFN（仅二值）。三者都是公共变化教师，不能替代个人 AOI 的逐期建筑实例模型。

## 1. 安装与 CUDA

安装 Python 3.11、创建 `.venv-gpu`，在 PowerShell 激活后先运行 `nvidia-smi`。到 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/) 选择与当前驱动匹配的 Windows/Pip/CUDA 命令，安装后验证：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
pip install -e . --no-deps
pip install PyYAML numpy pandas rasterio geopandas shapely pyproj scipy Pillow opencv-python-headless scikit-image pycocotools tqdm tensorboard
```

不要在已经安装 CUDA Torch 后直接运行会重新解析 `torch` 的通用安装命令。

## 2. 数据位置与路径

```text
D:\Data Set\S2Looking
D:\Data Set\SECOND
D:\Data Set\DSIFN
```

YAML 使用 `"D:/Data Set/S2Looking"`；PowerShell 使用 `"D:\Data Set\S2Looking"`。原始数据保持原目录，不能复制进 Git。

## 3. 先体检

```powershell
python run.py change-doctor --config configs/change_public_windows.yaml
python run.py change-preflight --config configs/change_public_windows.yaml
```

报告写在 `outputs/pretraining/change/preflight/PRETRAIN_CHECKLIST.md`。其中 `READY_TO_TRAIN=false` 时不能训练；先处理 blocker。

SECOND 必须确认真实建筑类别。单通道标签须让 `building_class_id` 出现在实际类别中；RGB 标签还须给出覆盖全部颜色的 JSON 映射，最后显式设 `mapping_verified: true`。不要猜测颜色或类别。

## 4. 推荐训练顺序

当前仓库已提供实际验证过的 S2Looking-only 配置：

```powershell
python -u run.py change-all --config configs/change_public_s2looking_windows.yaml --device cuda:0 --quick
```

补齐 DSIFN 或 SECOND 后，复制 `configs/change_public_windows.yaml`，仅保留对应数据集的 `enabled: true` 再运行同一条命令。SECOND 必须先补齐真实 `building_class_id`、`class_mapping_file` 和 `mapping_verified: true`；在此之前不提供会误导用户的可运行单数据集配置。

三套数据齐全且 SECOND 映射通过预检后，执行：

```powershell
python -u run.py change-all --config configs/change_public_windows.yaml --device cuda:0 --quick
```

quick 实际限制每个启用数据集最多 32 个 train 和 8 个 validation 样本，执行 forward、backward、validation、checkpoint reload、dashboard 与 review。它从不使用 test 做选择。

通过 quick 后，再运行：

```powershell
python -u run.py change-all --config configs/change_public_windows.yaml --device cuda:0
```

阶段关系：binary 使用全部可用监督；semantic 仅 SECOND；directional 仅 S2Looking + SECOND。无监督阶段会记录 `skipped_no_supervision`，最终 checkpoint 采用最后一个真实成功阶段。中断后加入 `--resume`。

## 5. 查看结果

打开 run 中的 `training_dashboard.html` 与 `review/index.html`；`latest.json` 只在完整成功后更新。验证与独立 test 指标分别写入 `metrics/` 与 `metrics/test/`。
