# 亚米遥感系统代码基线

本系统面向多城市的亚米级建筑制图与变化监测。西安是已实施的应用与验证案例，各模块的通用职责、城市适配范围和 F 盘归档方式见[模型与城市应用范围](MODEL_AND_CITY_SCOPE.md)。

从单期建筑识别开始，经外立面、分栋和位置治理，最后进行跨期身份关联、状态识别与面积交付。仓库保留既有城市更新平台的公共变化教师、建筑教师和审阅工具。西安模块可通过下表找到入口。

| 模块 | 应用代码 | 训练与研究 |
|---|---|---|
| 建筑识别 | [单期 B0](../applications/single_date/xian_single_date/b0.py)、[应用模型适配](../src/xian_r5/application_models.py)、[单期调度](../applications/single_date/xian_single_date/supervisor.py) | [独立屋顶候选、原生分辨率补检、轮廓精修](../research/building_recognition/) |
| 建筑治理 | [位置治理](../src/xian_r5/alignment/)、[外立面和模型调用](../src/xian_r5/application_models.py)、[空间几何](../src/xian_r5/application_geometry.py) | [2019 全市治理与训练、LOFT、成对分栋和切口选择](../research/building_governance/) |
| 时序追踪 | [跨期编排与状态](../src/xian_r5/application_temporal.py)、[R4 追踪入口](../work/run_xian_r4_tracking_candidate.py)、[追踪实现](../src/xian_r5/tracking/) | [任务对齐、时序升级及原生图匹配](../research/temporal_tracking/) |
| 面积和交付 | [精确面积分数](../src/xian_r5/label_transfer.py)、[单期合并](../applications/single_date/xian_single_date/assemble.py)、[完整应用](../src/xian_r5/application.py) | [模型及数据角色](../applications/xian_r5/MODEL_AND_DATA_ROLES.md) |

## 版本状态

- 建筑候选的保留基线是 B0。应用中的 E3 用于外立面风险诊断，V2 是候选位移。注意研究目录另含 2019 治理专用的立面、位移训练和产品脚本。
- R21 是目前保留的分栋研究对照。R22/R23 属于备选切口实验，R23 在 72 个重复使用的开发窗口上相对 R21 无净增匹配。它们的版本结论限于该评价范围。相同目录内脚本互相导入，保持同轮目录结构。
- 时序使用最近相邻的实际观测，不采用固定一年间隔。R4 追踪与应用状态逻辑在主应用。学习式时序升级和缺失判定实验保留在研究目录，生产推广仍待独立验证。
- Git 版采用应用冻结包中支持投影米制格网的面积实现，兼容原本机 Affine 版本。Git 提交记录供后续跟踪每次模块变化。

## 使用

使用已有的遥感 Python 环境。新环境先按根目录 README 安装依赖，并按模型要求配置 CUDA。以下 PowerShell 命令在仓库根目录运行。

```powershell
$env:PYTHONPATH = "$PWD\src"
python tools/run_xian_r5_integration.py --help
python -m pytest -q applications/xian_r5/tests
python tools/run_xian_r5_integration.py --application validate --manifest <本地清单>
python tools/run_xian_r5_integration.py --application plan --manifest <本地清单>
```

本机 F 盘归档中的迁移版应用清单可作为起点。另存副本，将 execution.code_snapshot 指向 Git 克隆根目录，将 output_root 指向独立的新目录。输入影像、格网、路由清单、权重及原生推理依赖继续由本地清单绑定。validate 和 plan 首先核对这些依赖。单期城市脚本用 applications/single_date 作为其模块根目录，其冻结配置可查本地的 20261028 单期城市运行。

公开仓库发布项目源码、合成测试与模块索引。数据、人工参考、权重、第三方权重包、训练回传和完整运行记录在本地归档中。研究脚本需要对应轮次的本地中间数据与配置。重新训练和复跑全市产品需配置这些外部依赖。

后续更新按模块提交。涉及训练或产品替换时，在提交说明记录当前基线、参数及数据版本、独立验证范围和产出路径。
