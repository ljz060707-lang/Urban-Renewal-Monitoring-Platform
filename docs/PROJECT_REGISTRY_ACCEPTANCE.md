# 项目注册与架构重构验收

验收日期：2026-07-28

## 结论

本轮没有重新训练模型、没有修改原始数据、没有覆盖历史 run/checkpoint，
也没有读取 DSIFN、S2Looking、SECOND 或 WHU 的 Test 图像。

项目现在具有一套权威、机器可读的模型、数据集、实验和产物目录。现有
变化检测成果已接入统一清单；Surface State、Temporal State、对象追踪与
统一推理目录已建立明确接口，但没有把尚未训练的能力伪装成可用模型。

## 实际登记结果

- 模型：5 个。
- 数据集：7 个（含一个明确标记为 planned 的未来北京数据模板）。
- 历史实验：48 个 run。
- 产物：4,207 个。
- 可视化入口：3,206 个。
- gallery 构建耗时：本机约 5.1 秒。

可视化扫描会跳过 checkpoint、prepared tiles、cache、tensorboard 等大目录；
同一目录最多索引 20 张图片，避免把数万张重复预览塞入单个 HTML。原文件
不会被删除，完整路径仍可从产物清单追溯。

## 当前模型状态

| Model | Status | Test policy | Inference |
|---|---|---|---|
| `dsifn_binary_v1` | `archived` | `spent_test` | ready, binary only |
| `s2looking_directional_v1` | `archived` | `spent_test` | ready, historical teacher |
| `full20_multisource_v2` | `validation_candidate` | `locked_holdout` | ready for candidate screening |
| `rsbuilding_swin_t_pretrained` | `external_pretrained` | `locked_holdout` | blocked |
| `rsbuilding_swin_t_whu_candidate` | `blocked` | `locked_holdout` | unavailable |

Full20 只能称 validation-only candidate。北京通州输出只能称
candidate building-change objects。

## Checkpoint 核验

| Model | SHA256 |
|---|---|
| DSIFN binary | `d458e9b6bed969fcf79a38603608834b5bcafd8c055ba6b61ffcba8387819210` |
| S2Looking directional archive | `4fd4c10cc8ceb0ce66c10c7e1f6bde4902404ff465a01423bdf4585653a6978c` |
| Full20 V2 | `70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621` |

RSBuilding 的外部权重未复制进项目，也未在错误环境中尝试加载。

## 真正可工作

- `model-list`、`dataset-list`、`experiment-list`、`artifact-list`。
- 自动构建 CSV/JSON 目录、model cards、gallery 和项目状态页。
- canonical output 中明确区分 available、pending、unsupported；不支持的
  head 返回 `None + status`，不返回假 0。
- metadata-first dataset audit；未经核验的 label mapping 会阻断 prepare。
- Test 访问守卫与 spent/locked 策略。
- 可配置 Temporal State Engine。
- 基于 STRtree 的一对一、一对多、多对一对象候选关联。
- 统一推理目录和 capability manifest 初始化。

## 仅为 scaffold / interface

- Surface State 的网络定义和指标可用，但没有经核验数据适配器 tensor
  reader，因此 `train-surface-state` 会明确阻断，不会伪造 smoke success。
- `infer` 与 `vectorize` 当前建立统一输出目录和 capability manifest，
  不是新的模型推理实现。
- Temporal State Engine 已通过合成序列测试，尚未在北京真实 surface-state
  概率上验收。
- 北京 waste / TFR / MTI 仅有 dataset adapter 和 mapping 模板；native label
  含义未核验，不能训练。

## WHU / RSBuilding blocker

独立环境缺少 `nvrtc-builtins64_117.dll`，错误发生在 CUDA
`trunc_normal_ → erfinv_` 初始化路径。当前状态是 assets/dataset/manifest
ready、smoke pending CUDA NVRTC fix、inference ready=false。WHU Test
仍锁定且 pixel reads=0。

## 实际验证

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q tests/test_project_registry_architecture.py
.\.venv-gpu\Scripts\python.exe run.py model-list
.\.venv-gpu\Scripts\python.exe run.py dataset-list
.\.venv-gpu\Scripts\python.exe run.py build-gallery
```

架构专项测试覆盖注册表、unsupported 输出、Test 守卫、时序规则、lineage、
surface-state 指标、统一 manifest 和未配置数据集零像素读取。

结果：`7 passed`。另有一个既有 `.pytest_cache` 目录无写权限警告，不影响
测试执行或产物。

内置浏览器实际验收：

- 项目状态页正确显示 5 个模型、7 个数据源、48 个实验；
- gallery 正确载入 3,206 个可视化条目；
- Full20 模型筛选得到 1,405 个匹配项；
- Full20 + external inference 筛选得到 42 个匹配项；
- 当前视口已加载图片中 broken image 数量为 0；
- 首次验收发现的模板语法和下拉刷新问题已经修复后复测。
