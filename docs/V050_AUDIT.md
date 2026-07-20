# v0.5.0 实施前审计

审计日期：2026-07-19（Asia/Shanghai）

## 仓库状态

- 本地目录：`C:/Users/ljz52/Documents/Codex/2026-07-18/urban-renewal-phase1-starter-v0-2/outputs/urban-renewal-phase1-starter`。
- 当前分支：`main`；初始提交：`5adb6a2`。
- 远端：`https://github.com/ljz060707-lang/Urban-Renewal-Monitoring-Platform.git`。
- 修改前工作树干净；`git pull --ff-only ... main` 返回 `Already up to date`。

## 实际基线验证

```text
python -m pytest -q
=> 90 passed, 1 deselected, 65 warnings in 18.37s

python run.py demo
=> success
=> outputs/demo_aoi/runs/20260719_230810
```

demo 在 CPU 上生成 `best.pth`、验证输出、完整影像推理、tracking 阶段和静态 review 页面。随机初始化 1 epoch 没有达到阈值的预测，因此输出 `No tracks produced`；这不是精度证明。

## 当前 CLI

已有 `doctor`、`prepare`、`train`、`evaluate`、`predict`、`track`、`all`、`demo`。没有 `change-doctor`、`change-prepare`、`change-train`、`change-all`、`change-demo`、`change-review`、`change-retrain`、`change-labels` 或 `change-finetune`。

## 当前建筑教师训练流程

- 单期教师为 torchvision Mask R-CNN ResNet50-FPN。
- 输入为单期 GeoTIFF 和建筑 Polygon/MultiPolygon GT。
- 支持 CPU/CUDA device 自动选择、CUDA AMP、gradient clipping、best/last/epoch checkpoint、resume 和 early stopping。
- validation 与完整影像推理读取 `best.pth`。
- 当前接口和 `python run.py all --aoi <aoi_id>` 必须保持兼容，不由变化教师替代。

## 当前 GPU 支持与实际环境

- 硬件：`NVIDIA GeForce RTX 4060 Laptop GPU`，驱动 `610.74`，显存 `8188 MiB`。
- 当前 Torch：`2.13.0+cpu`；`torch.cuda.is_available() == False`；CUDA runtime、cuDNN 均为空。
- 因此 v0.4 的代码具有 CUDA 路径，但当前环境尚不能真实执行 GPU 训练。
- v0.5 正式 `change-train/change-all` 必须强制 CUDA；只有 `change-demo --device cpu` 允许 CPU。

## 当前 review、tracking 和 fusion

- 静态 review 显示逐期建筑、轨迹状态、日期质量和 lineage，使用 localStorage 并导出 CSV。
- tracking 已使用空间索引、双向 ambiguity margin、完整 observation status 和结构化 confidence。
- review 可写 `state_final`，rejected 不进入 student export。
- 当前没有变化概率栅格、双教师 fusion、模型冲突字段、变化 mask 编辑或 correction 回流。

## 当前变化教师与公共数据能力

- `src/urban_monitor/change/` 不存在。
- 没有双时相模型、masked multi-task loss、平衡采样器或分阶段变化训练器。
- 没有 S2Looking、SECOND 或 DSIFN adapter，不能读取其原始目录或生成统一 index。
- 当前配置的候选路径 `D:/Datasets/S2Looking`、`D:/Datasets/SECOND`、`D:/Datasets/DSIFN` 以及 `data/public_change/*` 均不存在。
- 因缺少真实数据，本轮不能报告三个数据集的真实样本数、指标或 quick training 结果。

## 必须保持兼容的接口

1. 根目录 `run.py` 与现有 v0.4 子命令。
2. `data/<aoi_id>/images` + `labels` 最简目录。
3. Mask R-CNN checkpoint、evaluation、predictions、tracking、review 与 `latest.json` 目录合同。
4. `state_final > reviewed_state > state` 和 rejected 排除规则。
5. 现有 90 项基础测试及独立 `train_smoke` 标记。
6. 公共变化教师必须是独立模块；默认后端不能要求自定义 CUDA kernel。

## 审计结论

应在 v0.4 旁新增独立 `urban_monitor.change` 分支，通过 adapter 统一公共数据，以 masked supervision 训练多头双时相模型，再以概率栅格与建筑对象做融合。当前 GPU 硬件存在，但 CUDA Torch 与公共数据缺失；最终报告必须把 `implemented`、`CPU synthetic smoke`、`GPU smoke`、`public-data quick training` 和 `full training` 分开陈述。
