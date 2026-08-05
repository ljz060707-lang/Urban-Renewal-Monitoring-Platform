# 城市更新多源遥感项目概览

这个项目正在从“变化检测模型集合”升级为“可审计的城市更新识别系统”。

## 当前真正可用

- Any Change：DSIFN teacher 与 Full20 V2。
- New Building / Demolition：S2Looking teacher 与 Full20 V2。
- Semantic T1 / T2：Full20 基于 SECOND 的语义监督。
- 北京通州真实场景推理：结果是候选变化对象，不是 Ground Truth。
- 模型、数据集、实验和可视化注册表。
- 统一输出协议、Test 安全守卫、时序状态规则接口。

## 正在完成

- WHU / RSBuilding building geometry teacher。当前被 CUDA NVRTC 运行库阻断。
- Surface State：vacant land / construction site / building。接口、指标和 dataset onboarding 已完成；尚无经核验的正式训练数据。
- Temporal State Engine：规则引擎已工作，等待 surface-state 与 geometry 概率输入后进入真实场景验收。

## 科研路径

多时相影像 → Change / Direction / Surface State / Geometry → 对象提取 → 跨期 tracking → 状态序列 → Urban Renewal Event Database。

“Construction”不能仅凭单张裸地图像判断。可靠结论需要视觉状态、连续变化、建筑几何出现、稳定观察次数和实际时间跨度共同支持。

## 日常入口

```powershell
python run.py project-status
python run.py model-list
python run.py dataset-list
python run.py build-gallery
```

权威机器清单位于 `configs/registry/` 和 `outputs/catalog/`。
