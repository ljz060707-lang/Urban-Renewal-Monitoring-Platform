# Change Detection 下一轮训练优化报告

## 验收范围

- Baseline run：`20260721_232528`
- Demolition-head-only smoke：`20260724_013154`
- Train / validation / test：3500 / 500 / 1000
- 所有开发审计和训练均未读取 test 图像，`test_used=false`
- 本轮只执行 1 epoch smoke，未启动长训练

## 1. 原训练最浪费时间的位置

100 个真实 batch（batch=16、workers=0）的 profile：

- total batch：2.9716 秒
- data load：2.5032 秒
- PNG decode：2.1575 秒
- augmentation：0.3261 秒
- forward：0.2061 秒
- backward：0.2284 秒
- optimizer：0.0186 秒

数据等待占单 batch 约 84%，其中 PNG decode 是主因。GPU forward/backward 不是主瓶颈。

## 2. 原训练最浪费监督信号的位置

原逻辑只提高困难图片的 image-level sampling weight，随后仍在 1024×1024 图像上均匀随机裁 256×256 patch。对照采样中，hard-FN crop 命中率只有 **35.94%**。

现已改为：

1. 选择 sample；
2. 选择 binary/new/demolition 目标；
3. 优先读取对应 FN mask；
4. 围绕正像素或连通区域定向裁剪；
5. 保留 70% target、20% boundary、10% random 的可配置随机性。

同一 500-patch 审计中，change-aware hard hit rate 为 **93.75%**；真实 smoke 完整 epoch 为 **92.00%**。

## 3. 3500 张 train 的真实监督信号

| Head | 正类图片 | 正类图片比例 | 组件数 | 面积 P10/P25/Median/P75/P90（像素） |
|---|---:|---:|---:|---|
| Binary | 3495 | 99.86% | 28075 | 195 / 383 / 805 / 1612 / 3378 |
| New | 2954 | 84.40% | 18584 | 201 / 403 / 852 / 1697 / 3586.7 |
| Demolition | 2585 | 73.86% | 11278 | 184 / 346 / 693 / 1374 / 2803.6 |

按组件面积 P25/P75 分层：

- Binary：small 7018 / medium 14040 / large 7017
- New：small 4631 / medium 9308 / large 4645
- Demolition：small 2818 / medium 5641 / large 2819

## 4. 每 epoch 实际看到的 patch

Run `20260724_013154` epoch 0：

- total：3496
- background：337（9.64%）
- binary positive：3159（90.36%）
- new positive：1771（50.66%）
- demolition positive：2432（69.57%）
- both positive：1044
- hard-FN：525（15.02%）
- hard-FN hit：483（92.00%）

训练仍使用全部 train 数据；没有缩减 train 或 validation。

## 5. Binary / New / Demolition 梯度冲突

50 个真实 train batch、共享参数、分别独立 backward：

| 梯度对 | 均值 | 中位数 | cosine < 0 |
|---|---:|---:|---:|
| Binary / New | 0.7737 | 0.8475 | 0% |
| Binary / Demolition | 0.5894 | 0.6585 | 4% |
| New / Demolition | 0.1115 | 0.0886 | 32% |

结论：并非三个任务全面冲突。最明显冲突位于 **New 与 Demolition 的共享参数更新**，解释了此前 Demolition 改善时 New 经常退化。当前证据支持 head-only 隔离，不支持立即引入 PCGrad/GradNorm。

## 6. Batch size / DataLoader

对 batch 4/8/12/16 与 workers 0/2/4 做真实训练步 benchmark，排除 2 个 warm-up batch：

- 最优：batch=8、workers=4，15.36 samples/s，峰值约 1.85 GiB
- 次优：batch=16、workers=4，14.44 samples/s，峰值约 3.29 GiB
- 原始 profile：batch=16、workers=0，5.38 samples/s

因此选择 batch=8、workers=4、`persistent_workers=true`、`pin_memory=true`、`prefetch_factor=2`。没有因为显存尚余就盲目增大 batch。

已实现可选 `cache_mode=none|ram` LRU 解码缓存，但当前 smoke 保持 `none`：workers=4 已显著缓解解码瓶颈，避免四个 Windows worker 各自占用大 RAM cache。

## 7. Source validation cache 与 threshold scan

缓存 key 包含：

- checkpoint SHA256
- validation identity/index SHA256
- model config hash
- preprocessing hash
- split

缓存保存三个 head 的 raw probability 和 target，不只保存 F1。首次完整 scan 约 134 秒；缓存命中后约 45 秒，重复运行节约约 **89 秒（约 66%）**。

Threshold scan 现在对 500 validation 只 forward 一遍，然后在 CPU 上向量化计算 19×3=57 行阈值结果。test 不允许用于 threshold selection。

## 8. Demolition-head-only 1 epoch

启动时只有：

- `demolition_head.weight`
- `demolition_head.bias`

可训练，共 33 / 35,428,376 参数（0.000093%）。

第一次 smoke 暴露出真实冻结漏洞：仅设置 `requires_grad=false` 仍会让共享 BatchNorm buffers 在 `model.train()` 下更新，并启用 Dropout，导致 Binary/New logits 大幅变化。现已改为全模型 eval，仅 demolition head 处于 train mode，并对参数和 buffers 一并做 SHA256 回归检查。

成功 run `20260724_013154`：

| 指标 | Source | Candidate | Delta |
|---|---:|---:|---:|
| Binary F1 | 0.421373 | 0.421373 | 0 |
| Binary PR-AUC | 0.366963 | 0.366963 | 0 |
| New F1 | 0.446676 | 0.446676 | 0 |
| New PR-AUC | 0.393635 | 0.393635 | 0 |
| Demolition F1 | 0.318343 | 0.318934 | +0.000590 |
| Demolition PR-AUC | 0.217482 | 0.218448 | +0.000967 |

- 固定 20 张 validation：Binary/New logits max absolute difference 均为 0
- encoder、binary head、new head 的 state SHA256 训练前后完全一致
- 训练 147.36 秒，完整 validation 39.00 秒
- improvement per GPU hour（按正式 threshold scan PR-AUC）：约 **0.0187**
- `model_selection_status=improved`

## 9. 最终判断

下一步值得：**继续一轮 demolition-head-only**。

暂不值得：

- 解冻 decoder：New/Demolition 冲突证据明确；
- 改 loss：尚无证据表明 loss 是主瓶颈；
- 换模型：数据解码与 crop 信号问题已解释主要效率损失；
- 直接长训练：当前提升为正但幅度小，应继续逐 epoch gate。

## 唯一推荐命令

```powershell
.\.venv-gpu\Scripts\python.exe -u run.py change-retrain `
  --run 20260721_232528 `
  --config configs/change_public_s2looking_demolition_head_epoch2.yaml `
  --from-review `
  --device cuda:0
```

该命令从已接受的 `20260724_013154` checkpoint 再执行唯一一个受控 epoch；仍使用完整 train、完整最终 validation，并保持 test 隔离。
