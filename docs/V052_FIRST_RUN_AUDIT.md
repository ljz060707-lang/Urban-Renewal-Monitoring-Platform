# v0.5.2 首轮训练诊断审计

审计对象：`outputs/pretraining/change_s2looking/20260721_232528`；数据集为 S2Looking。本文记录的是实际读取到的首轮产物，而不是历史报告的转述。

## 已确认的首轮事实

- directional 最佳 checkpoint：`checkpoints/best_directional_change.pth`，best epoch 22。
- validation 为 500 张，train 为 3500 张，test 为 1000 张；本轮 threshold scan 的 `test_used_for_validation=false`。
- directional checkpoint 在 validation、阈值 0.50 的 binary F1=0.421373、IoU=0.266924。
- binary-only 阶段只运行 8 个 epoch，全部 `val_binary_f1=0` 后早停；其训练/验证 loss 下降但在固定 0.50 下没有有效正类预测。
- 方向训练阶段能够恢复 binary head，因此最终方向 checkpoint 的 binary 概率保存和指标通路是可工作的。

## 概率与页面根因

`evaluation.py` 对 logits 调用 `sigmoid` 后保存 PNG，映射为 0--255。默认 F1 阈值为 0.50。原 review 页面先绘制 T2，随后在同一个主 canvas 使用 `putImageData` 写整幅透明层；该 API 不做 source-over alpha 合成，透明/黑色像素覆盖了底图，故出现“蓝色和黑色”。

已改为：先画 T2，再在独立 offscreen canvas 上创建 mask，最后用 `drawImage(offscreen)` 与 `source-over` 叠加。因此阈值 1.00、透明度 0 都保留 T2。

## 阈值扫描（仅 validation）

| head | best threshold | Precision | Recall | F1 | IoU | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| binary | 0.50 | 0.410462 | 0.432880 | 0.421373 | 0.266924 | 0.366963 |
| new | 0.60 | 0.438239 | 0.455445 | 0.446676 | 0.287562 | 0.393635 |
| demolition | 0.30 | 0.332341 | 0.305477 | 0.318343 | 0.189303 | 0.217482 |

结论：首轮 binary-only F1=0 属于 **model_collapse**（早停后的 binary-only 背景解），不是最终 directional checkpoint 的阈值问题、序列化问题或 fake metric。最终模型本身仍有大量漏检（binary FN=298153 像素），所以适合进入人工确认的困难样本闭环；不能直接从 validation 漏检回灌训练。

## 本轮真实执行

- validation threshold scan（GPU，500 validation；test 未读取）
- binary stage diagnostic
- 静态 review 页面重建并以 prepared index 回填 GT 资产
- GLM dry-run 10 张联系图（没有 API key、零远程调用）
- 新增单元测试 3 项，全部通过

尚未执行：真实 GLM API 调用、100 张人工审核、困难样本微调、候选模型对比。它们依赖用户提供的人工审核结果（以及可选 API key），因此不能诚实地声称完成。
