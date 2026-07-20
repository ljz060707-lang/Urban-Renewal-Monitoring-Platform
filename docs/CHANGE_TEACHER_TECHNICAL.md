# 变化教师技术参考

## 定位

变化教师是 v0.5 新增的独立双时相像素模型。v0.4 Mask R-CNN 仍负责逐期建筑实例；二者在真实 AOI 推理后通过 fusion 合并证据。

## 统一数据契约

`public_change_index.csv` 包含 T1、T2、binary、new、demolition、T1/T2 semantic 路径以及 `has_binary`、`has_directional`、`has_semantic`。缺失监督为 `None/空路径 + false`，不允许创建全零替代标签。

adapter 只读原数据，标准化 mask 写入 run 的 `prepared/normalized_labels/`。

## 默认模型

`MultiTaskSiameseUNet` 使用共享 ResNet34 encoder。T1/T2 经过相同权重，各尺度融合：

```text
[T1 feature, T2 feature, abs(T1 - T2)]
```

U-Net decoder 输出：

- `binary_change`
- `new_building`
- `demolition`
- `semantic_t1`
- `semantic_t2`
- `uncertainty`

预训练 encoder 默认不静默下载。`pretrained: true` 时必须提供本地 `weights_path`，或显式设置 `allow_download: true`。

ChangeMamba 是 Linux/WSL 可选后端，要求外部 checkout、`selective_scan` 和本地权重。默认 Windows/Linux 路径不依赖自定义 CUDA kernel。

## Masked multi-task loss

默认总损失：

```text
0.5 * L_binary + 1.0 * L_new + 1.0 * L_demolition + 0.5 * L_semantic
```

二值任务使用 BCEWithLogits + Dice，语义任务使用 Cross Entropy + multiclass Dice。每个任务只在相应监督 flag 为 true 的样本上计算；返回任务级和数据集级损失。可选 focal gamma 与语义类别权重。

## 增强和采样

裁剪、翻转、90°旋转和有界平移同步作用于 T1、T2 与所有标签。亮度、对比度和轻噪声可分别作用于 T1/T2。不会对两期执行不同的几何变换。

默认按数据集权重采样，而不是按原始样本量直接混合：

```yaml
dataset_sampling:
  s2looking: 2.0
  second: 1.0
  dsifn: 1.0
```

支持 weighted、round-robin 和 DDP rank 切片，随机种子与 epoch 决定采样序列。

## 三阶段训练

1. binary：S2Looking + SECOND + DSIFN，输出 `best_binary_change.pth`。
2. semantic：SECOND，继承 binary best，输出 `best_semantic_change.pth`。
3. directional：S2Looking + SECOND，继承 semantic best，输出 `best_directional_change.pth`。

每阶段保存 best、last 和有限数量 epoch 快照。相同状态优先使用硬链接去重；失败时才复制。checkpoint 含 model、optimizer、scheduler、GradScaler、epoch、stage、best metric、resolved config、数据注册、类别映射、版本、git commit 和随机种子。

## GPU

正式训练强制 `cuda:N`。实现 AMP、GradScaler、gradient clipping、pin memory、non-blocking transfer、persistent workers、CUDA memory 日志、OOM 提示、resume、early stopping 与可选 DDP。

`batch_size: auto` 只按 16、8、4、2、1 尝试一次，最多 5 次。不会在正式训练循环中无限猜测。

本轮开发机的 RTX 4060 可被 `nvidia-smi` 识别，但活动 Torch 为 CPU build；临时 CUDA wheel 安装在 10 分钟限制内超时，因此不能声称 GPU smoke 通过。

## 评估

每个数据集和 combined 分别写 JSON。没有该任务 GT 时输出 `available: false` 和 `null`，不输出伪零分。评估包括 binary、directional、semantic 指标、预测 mask、错误集合与审核索引。

## 本地微调

本地候选经人工确认并形成 reviewed index 后，继承 directional best：

1. encoder 冻结；
2. 解冻 layer3/layer4；
3. 全模型低学习率微调。

输出 `best_china_urban_change.pth`，公共预训练权重不被覆盖。

## 双教师融合

fusion 在建筑 track 几何范围统计 any/new/demolition/uncertainty raster，结合 `state_final > reviewed_state > state` 生成一致性、融合分数和 review priority。该分数是规则分数，不是校准概率。冲突、歧义、lineage、施工候选与配准问题进入联合审核。
