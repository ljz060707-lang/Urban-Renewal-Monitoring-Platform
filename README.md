# 城市更新遥感监测平台

把多期高分辨率影像和建筑 Ground Truth 放入指定文件夹，运行一条命令，即可训练建筑识别模型并生成跨期城市更新结果。

最常用流程：

```text
数据：data/<aoi_id>/images/ 和 data/<aoi_id>/labels/
安装：pip install -e ".[train]"
运行：python run.py all --aoi beijing_demo
结果：outputs/beijing_demo/latest.json
```

v0.5 另提供独立的双时相变化教师，用 S2Looking、SECOND、DSIFN 做 GPU 预训练。它不会替换 v0.4 的建筑实例教师。

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

CPU 用户可直接安装。CUDA 用户应先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装与显卡驱动匹配的 Torch/torchvision，再执行：

```bash
pip install -e ".[train]" --no-deps
```

可选交互审核工具：

```bash
pip install -e ".[review]"
```

## 2. 放入本地 AOI 数据

只要求 `images` 和 `labels`：

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

- 影像和标签使用相同日期命名。
- 标签优先使用 GPKG，可以只有 geometry。
- 影像和标签必须带真实 CRS。
- 不要把高德 GCJ-02 或百度 BD-09 图层直接当作 WGS84 Ground Truth。
- 不需要手写 manifest；可选 `project.yaml` 只用于覆盖默认值。

## 3. 一键运行建筑实例教师

```bash
python run.py doctor
python run.py all --aoi beijing_demo
```

系统自动完成发现、manifest、CRS 对齐、GT 校验、空间划分、切片、COCO、Mask R-CNN 训练、验证、全图预测、tracking、lineage、review 和报告。

结果由 `outputs/beijing_demo/latest.json` 指向，常用文件为：

```text
training/checkpoints/best.pth
evaluation/evaluation_summary.md
predictions/
tracking/
review/index.html
RUN_MANIFEST.json
```

## 4. 公共变化模型 GPU 预训练

1. 把数据集真实路径写入 `configs/change_public.yaml`，并设置 `enabled: true`。
2. SECOND 必须明确 `building_class_id`；RGB 标签还必须提供经过确认的 `class_mapping_file`。
3. 运行：

```bash
python run.py change-doctor --config configs/change_public.yaml
python run.py change-all --config configs/change_public.yaml --device cuda:0
```

正式 `change-train` / `change-all` 强制 CUDA，不会静默回退 CPU。仅以下网络无关 demo 允许 CPU：

```bash
python run.py change-demo --device cpu
```

公共训练输出：

```text
outputs/pretraining/change/latest.json
checkpoints/best_binary_change.pth
checkpoints/best_semantic_change.pth
checkpoints/best_directional_change.pth
training_dashboard.html
review/index.html
```

## 5. 审核、修正与本地微调

```bash
python run.py change-review --run latest --config configs/change_public.yaml
python run.py change-review --run latest --config configs/change_public.yaml --interactive
python run.py change-retrain --run latest --from-review --device cuda:0
python run.py change-labels --aoi beijing_demo
python run.py change-finetune --aoi beijing_demo --checkpoint latest-directional --device cuda:0
```

静态审核页无需服务即可打开。修正 mask 写入 `review/corrections/`，不会修改原始公共数据。`exclude`、`misregistration` 和未修正的 `label_error` 不进入重训；联合审核中的 `rejected` 不进入学生标签。

本地 `change-labels` 只生成候选。人工确认前，它们不是 Ground Truth，也不能用于 fine-tune。

## 6. 常用命令

```bash
python run.py prepare --aoi beijing_demo
python run.py train --aoi beijing_demo --resume
python run.py predict --aoi beijing_demo
python run.py all --aoi beijing_demo --force
python run.py change-prepare --config configs/change_public.yaml
python run.py change-train --config configs/change_public.yaml --stage directional --device cuda:0
```

## 7. 常见错误

- **CRS 缺失**：在 QGIS 中确认真实 CRS 后重新导出，不要随意指定为 WGS84。
- **日期不对应**：检查影像与标签文件名；同日冲突会明确报错。
- **CUDA 不可用**：运行 `change-doctor`，检查 Torch 是否为 CUDA 构建以及 `nvidia-smi` 是否可见。
- **CUDA OOM**：减小 patch 或 batch；`batch_size: auto` 只有限尝试 16、8、4、2、1。
- **SECOND RGB 映射缺失**：系统拒绝猜测类别颜色，必须提供经确认的映射文件。
- **验证监督缺失**：系统输出 `available: false`，不会报告伪造的 0 分。
- **公共数据许可**：先阅读 [docs/PUBLIC_DATASETS.md](docs/PUBLIC_DATASETS.md)，不要默认允许商业用途。

## 8. 技术文档

- [本地 AOI 与 v0.4 技术参考](docs/TECHNICAL_REFERENCE.md)
- [变化教师结构与训练](docs/CHANGE_TEACHER_TECHNICAL.md)
- [公共数据集与许可](docs/PUBLIC_DATASETS.md)
- [人工审核和修正回流](docs/HUMAN_REVIEW_WORKFLOW.md)
- [Ground Truth 规范](GROUND_TRUTH_SPEC.md)
