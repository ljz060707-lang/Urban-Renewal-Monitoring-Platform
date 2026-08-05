# Repository Audit — 2026-07-28

本审计只读取项目内代码、manifest、指标、checkpoint 元数据和可视化资产路径，没有读取任何锁定 Test 图像像素，也没有启动训练。

## 已核验模型

- DSIFN Binary Teacher：checkpoint 存在，SHA256 与登记值一致；DSIFN Test 已 spent。Validation F1 0.830425，已执行的一次 Test F1 0.628140，存在明显泛化差距。
- S2Looking Direction Teacher：归档 checkpoint 存在；S2Looking Test 已 spent。它只作为历史 teacher / initialization source。
- Full20 V2：run `20260727_001029`、20 epoch、Test reads=0。状态必须是 `validation_candidate`，不能称 final 或 test-proven。
- RSBuilding Swin-T：外部 pretrained geometry teacher。必须使用独立 `rsbuilding-win` 环境，当前不是 inference-ready。
- WHU candidate：Train/Validation 资产和 manifest 已准备；Test locked，pixel reads=0。Smoke 被缺失 `nvrtc-builtins64_117.dll` 阻断。

## 已核验数据角色

- DSIFN：binary change，3600/340/48，Test spent。
- S2Looking：binary/new/demolition，3500/500/1000，Test spent。
- SECOND：semantic/directional，2668/300 development split 均来自官方 Train；Test locked。
- WHU：building geometry，4736/1036/2416；Test locked。
- Tongzhou：外部真实场景，不是 benchmark split，也没有 Ground Truth。

## 与历史描述的差异

- 项目已存在大量 V2、final-test、review、hard-example 与 inference 工程代码；本轮不重写这些模块。
- `outputs/models` 中只有三个已归档模型目录；RSBuilding 与 WHU candidate 此前没有统一 model card，本轮只新增元数据目录，不复制外部权重。
- Tongzhou 的现有数字只能称为 candidate building-change objects，不能称 confirmed buildings 或真实城市更新数量。

## 保留事项

- 所有旧 run、checkpoint、Test receipt 与可视化原样保留。
- `.venv-gpu` 未安装 RSBuilding 依赖。
- 没有重新运行 Full20、DSIFN Test、S2Looking Test，也没有读取 SECOND/WHU Test。
