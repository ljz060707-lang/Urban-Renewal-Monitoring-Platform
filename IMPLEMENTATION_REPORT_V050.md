# v0.5.0 Multi-Teacher GPU Training Edition 实施报告

实施日期：2026-07-19（Asia/Shanghai）

## 1. 交付状态摘要

| 项目 | 状态 | 真实证据 |
|---|---|---|
| v0.4 建筑实例教师兼容 | 已实现并回归通过 | 基础套件包含全部旧测试 |
| 三个公共数据 adapter | 已实现 | adapter/preparation 测试通过 |
| 多任务变化教师 | 已实现 | CPU forward、训练和 checkpoint 重载通过 |
| binary/semantic/directional | 已实现并完成 CPU synthetic demo | 每阶段 1 epoch |
| GPU 训练路径 | 已实现 | CUDA 门禁、AMP/DDP/auto-batch 代码与测试入口 |
| 真实 GPU smoke | **未通过** | 活动 Torch 为 CPU build；临时 CUDA wheel 安装 10 分钟超时 |
| 公共数据 quick training | **未执行** | 6 个候选数据路径均不存在 |
| 公共数据 full training | **未执行** | 同上 |
| 静态 review | 已实现并浏览器实测 | 3 卡片、筛选、无横向溢出 |
| 交互 review | 已实现，未启动实测 | 当前环境未安装 Gradio extra |
| correction 回流 | 已实现并测试 | 修正优先、排除和困难样本测试通过 |
| local fine-tune | 调度与入口已实现，未用真实 AOI 训练 | 缺人工确认本地变化 index |
| 双教师 fusion | 已实现并测试 | disagreement 和 rejected 闭环测试通过 |

## 2. 新增模块

- `change/schema.py`：统一样本契约。
- `change/adapters/`：S2Looking、SECOND、DSIFN 原目录 adapter。
- `change/preparation.py`：统一 index、标准化标签、统计和预览。
- `change/datasets.py`、`augmentations.py`、`samplers.py`：dataset、同步增强和平衡采样。
- `change/models/`：MultiTask Siamese U-Net 与可选 ChangeMamba adapter。
- `change/losses.py`：masked multi-task loss。
- `change/training.py`：CUDA/AMP/DDP、阶段训练、resume、best/last 和微调调度。
- `change/metrics.py`、`evaluation.py`、`visualization.py`：分数据集指标、预测、错误集和 dashboard。
- `change/review.py`、`review_app.py`、`corrections.py`：静态/交互审核与修正回流。
- `change/local_labels.py`：本地相邻日期候选变化标签。
- `change/fusion.py`、`joint_review.py`：双教师 fusion 与联合审核。
- `change/cli.py`：全部 `change-*` 命令。

## 3. 公共数据读取与标签映射

adapter 直接读取各数据集原目录，不要求用户复制重排。

- S2Looking：读取 `Image1/Image2/label/label1/label2`；new=`label1`，demolition=`label2`，统一 binary 为二者 OR；原 `label` 不一致写审计。
- SECOND：读取 `T1/T2/GT_T1/GT_T2`，可选原 `GT_CD`；单通道类别 ID 可直接使用，RGB 必须提供显式颜色表。building class 必须明确。输出类别分布、转移矩阵、mapping JSON 和预览。
- DSIFN：读取 T1/T2/binary。方向与语义监督保持缺失，不填零。

统一索引由三个 supervision flag 控制 loss。没有 GT 的指标输出 `available:false` 和 `null`。

## 4. 模型与损失

默认模型为共享 ResNet34 encoder 的 MultiTask Siamese U-Net。各尺度拼接 T1、T2 与绝对差特征；输出 binary、new、demolition、T1 semantic、T2 semantic 和 uncertainty 六个 head。

binary/new/demolition 使用 BCEWithLogits + Dice，semantic 使用 Cross Entropy + multiclass Dice。任务只在相应 flag 为 true 的样本上计算。支持类别权重和可选 focal gamma。

默认后端不依赖自定义 CUDA kernel。ChangeMamba 仅作为 Linux/WSL 外部 adapter，不复制其源码或权重。

## 5. 采样和增强

dataset-balanced sampler 支持 weighted、round-robin 和 DDP rank view。默认建议 S2Looking:SECOND:DSIFN=`2:1:1`，固定 seed，并记录实际采样。

空间增强同步作用于 T1/T2/所有标签；两期只允许独立的轻光度扰动。不会人为制造不同几何配准。

## 6. GPU 配置与 checkpoint

正式 `change-train/change-all` 强制 `cuda:N`，CPU 仅允许 `change-demo`。训练器支持 AMP、GradScaler、gradient clipping、pin memory、non-blocking transfer、persistent workers、CUDA memory、OOM 提示、early stopping、resume 和 DDP。

`batch_size:auto` 只尝试 16、8、4、2、1。checkpoint 保存 model、optimizer、scheduler、scaler、epoch、stage、best metric、配置、映射、数据注册、版本、git commit 和 seed。best/last/epoch 同状态优先用硬链接去重。

## 7. 三阶段训练

1. binary：三套数据，输出 `best_binary_change.pth`。
2. semantic：SECOND，继承 binary best，输出 `best_semantic_change.pth`。
3. directional：S2Looking+SECOND，继承 semantic best，输出 `best_directional_change.pth`。

本地 fine-tune 继承 directional best，按 head-only → partial encoder → full model 调度，输出 `best_china_urban_change.pth`。

## 8. 评估和可视化

每个数据集及 combined 分别输出 JSON。支持 binary Precision/Recall/F1/IoU/Dice/Specificity/Balanced Accuracy/PR-AUC，directional new/demolition 指标，以及 semantic per-class IoU/mIoU/building IoU/change accuracy。

生成 predictions、10-panel previews、错误集合、训练曲线、GPU memory 图和静态 `training_dashboard.html`。没有实测序列的图明确显示无数据，不填造数值。

## 9. 审核、修正和融合

静态 review 支持筛选、排序、阈值/透明度、状态、备注、localStorage 和 CSV。浏览器实测修复了 CSV 转义、横向溢出及隐藏 mask 显示问题。

Gradio 可选工具把修正写入独立 corrections 目录。回流时 `human_corrected_label > original_dataset_label`；exclude/misregistration/未修正 label_error 排除，prediction_error 加入困难样本。

本地 GPKG 差分只生成 candidate。双教师 fusion 对 track 几何统计变化概率并生成 agreement/confidence/priority。联合审核写 `state_final`，rejected 不进入学生融合标签。

## 10. 实际执行命令和结果

基线：

```text
git pull --ff-only Urban-Renewal-Monitoring-Platform main
python -m pytest -q
python run.py demo
```

基线结果：`90 passed, 1 deselected`；v0.4 demo 成功。

本轮最终验证：

```text
python -m pytest -q
python -m pytest -q -m train_smoke
python run.py change-demo --device cpu --force
python run.py change-doctor --config data/change_demo/change_demo.yaml
```

最终基础测试实际结果：`115 passed, 1 skipped, 3 deselected`。

train smoke：`3 passed, 116 deselected`。包含 v0.4 实例教师训练、变化模型 forward、变化教师 1 epoch/checkpoint 重载推理。

GPU 标记测试：`1 skipped, 118 deselected`；跳过原因是活动 Torch 为 CPU build，不是 GPU 测试通过。

CPU change demo：成功。每套 synthetic 数据 2 个样本（train 1 + val 1），三阶段各 1 epoch。

## 11. Synthetic demo 实际指标

这些数值只证明 pipeline 产出真实指标，不代表模型精度：

| 数据集 | Binary F1 | Binary IoU | PR-AUC | Directional GT | Semantic mIoU |
|---|---:|---:|---:|---|---:|
| S2Looking | 0.246575 | 0.140625 | 0.811368 | 有 | 不可计算 |
| SECOND | 0.246575 | 0.140625 | 0.811368 | 有 | 0.035156 |
| DSIFN | 0.246575 | 0.140625 | 0.811368 | 不可计算 | 不可计算 |

## 12. 实际输出路径

成功 run：`outputs/pretraining/change_demo/20260719_235625/`

```text
checkpoints/best_binary_change.pth
checkpoints/best_semantic_change.pth
checkpoints/best_directional_change.pth
training_dashboard.html
review/index.html
metrics/s2looking.json
metrics/second.json
metrics/dsifn.json
metrics/combined.json
```

validation 已从磁盘重新加载 `best_directional_change.pth`。`best_directional_change.pth`、`last_directional.pth` 与当期 epoch 文件已用 `fsutil hardlink list` 确认去重。

## 13. GPU smoke 与公共数据训练

硬件：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，驱动 610.74。

活动环境：Torch `2.13.0+cpu`，torchvision `0.28.0+cpu`，`torch.cuda.is_available()==False`，CUDA runtime/cuDNN 为 null。

官方索引存在 `torch 2.13.0+cu130` 和 `torchvision 0.28.0+cu130`。本轮尝试向临时 `.cuda_runtime` 安装，命令在 604 秒超时且未留下可导入目录。因此：

- GPU smoke passed：**否**。
- quick training passed：**否，真实公共数据缺失**。
- full training completed：**否**。

缺失路径：

```text
D:/Datasets/S2Looking
D:/Datasets/SECOND
D:/Datasets/DSIFN
data/public_change/s2looking
data/public_change/second
data/public_change/dsifn
```

## 14. 已知限制

- 尚未在 CUDA Torch 上实测 AMP、OOM auto-batch 或 DDP；测试入口会在无 CUDA 时明确 skip。
- 未对真实公共数据执行 quick/full training，不能报告真实样本数和精度。
- Gradio UI、ChangeMamba、真实 AOI local fine-tune 和真实双教师融合尚未端到端运行。
- new/demolition/semantic 训练曲线在 trainer 尚未逐 epoch 记录时为明确的无数据占位图；最终 evaluation JSON 是可用的。
- fusion confidence 是规则得分，不是校准概率。

## 15. 正式训练命令

安装 CUDA Torch 并放入三套数据后：

```bash
python run.py change-doctor --config configs/change_public.yaml
python run.py change-prepare --config configs/change_public.yaml
python run.py change-all --config configs/change_public.yaml --device cuda:0 --quick
python run.py change-all --config configs/change_public.yaml --device cuda:0
```

只有 quick 成功后才建议启动完整训练。
