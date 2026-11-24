# 模块状态与迭代记录

本系统面向多城市的亚米级建筑制图与变化监测。西安是已实施的应用与验证案例，各模块的通用职责、城市适配范围和 F 盘归档方式见[模型与城市应用范围](MODEL_AND_CITY_SCOPE.md)。

## 项目模块

| 模块 | 代码入口 | 数据与当前状态 |
|---|---|---|
| 影像与日期 | [单期调度](../applications/single_date/xian_single_date/supervisor.py)、[应用契约](../src/xian_r5/application_contract.py)、[输入与有效覆盖](../src/xian_r5/coherent_support.py) | 底图、有效范围、格网、实拍日与混合采集由本地清单绑定。重复实际日期归并，跨期配对按相邻有效观测。 |
| 建筑语义与逐栋识别 | [单期 B0](../applications/single_date/xian_single_date/b0.py)、[建筑基础模型](../src/urban_monitor/building_foundation_v2/)、[建筑识别研究](../research/building_recognition/) | B0/Swin-T 为西安应用保留候选基线。WHU 建筑监督与原生分辨率补检分别登记。候选缺失、粘连、碎裂要分别计数。 |
| 外立面 | [模型适配](../src/xian_r5/application_models.py)、[2019 全市治理](../research/building_governance/xian_2019_citywide_governance/) | E3 在应用中供风险诊断。2019 训练与治理脚本另有立面判决和校准。BONAI 与 BANDON 监督的语义及目标域适用性见模型角色文档。 |
| 位移与地面轮廓 | [alignment](../src/xian_r5/alignment/)、[空间几何](../src/xian_r5/application_geometry.py)、[多视角实验](../research/building_governance/r13_multiview_geometry/) | V2 提供候选位移，西安开发窗以冻结日期位移作为对照。2019 人工图属于参考，拟合结果按参考约束披露。 |
| 分栋与轮廓 | [几何治理研究](../research/building_governance/) | R21 成对分栋为保留对照。R22 比较多条边缘与空隙切口，R23 续训在 72 个旧开发窗相对 R21 未增加净匹配。 |
| 时序身份 | [应用时序](../src/xian_r5/application_temporal.py)、[tracking](../src/xian_r5/tracking/)、[时序实验](../research/temporal_tracking/) | R4 为应用路径。SpaceNet 7 支持公开时序监督。多期学习式关联及原生屋顶关联保留研究状态。 |
| 建筑状态与变化 | [reconstruction](../src/xian_r5/reconstruction/)、[resolver](../src/xian_r5/resolver/)、[变化教师](../src/urban_monitor/change/) | 观测、身份、Presence 和事件各有证据。参考缺失、影像不可用与未解决对应进入未知状态。公开变化教师和西安关联实验分别验收。 |
| 建筑工地与地表 | [工地模块](../src/urban_renewal/construction/)、[工地文档](CONSTRUCTION_ACTIVITY_V1.md)、[地表状态](../extensions/surface_state_v1/) | 城市更新研究含工地与地表状态实验，数据准入和能力状态见现有模块注册表。 |
| 精确面积、矢量与栅格 | [label_transfer](../src/xian_r5/label_transfer.py)、[单期合并](../applications/single_date/xian_single_date/assemble.py)、[完整应用](../src/xian_r5/application.py) | 输出包括单期实例、有效覆盖、逐栋面积及网格并集分数。需区分对象面积和与并集。 |
| 审阅与交付 | [Web 审阅](../web/urrs_review/)、[人工审阅流程](HUMAN_REVIEW_WORKFLOW.md)、[URRS 案例审阅](URRS_TONGZHOU_REVIEW_CONSOLE.md) | 人审纠正、案例图与验收回执作为下次迭代的可追溯证据。 |

公共数据及模型的既有注册表见 [configs/registry](../configs/registry/)、[数据集目录](DATASET_CATALOG.md)和[模型目录](MODEL_CATALOG.md)。西安应用的 B0/E3/V2/R4 角色与训练边界见[应用模型卡](../applications/xian_r5/MODEL_AND_DATA_ROLES.md)。F 盘亚米遥感系统的训练集、权重、城市结果与完整运行回执按目录归档，Git 通过本地清单连接这些资产。公开源码支持在该环境中复跑指定阶段，完整的历史训练与全市生产需要各轮权重、影像、配置、标签和派生数据。

## 每次迭代应记录

| 字段 | 内容 |
|---|---|
| 模块和改动 | 对应入口、算法变化、输入输出协议版本、前一基线提交 |
| 资产 | 训练与验证区、实拍日期、数据版本或哈希、权重哈希、配置和本地回执位置 |
| 比较 | 固定对照、评价口径和分母、是否用了 GT 约束、冻结前后指标 |
| 验证与状态 | 测试命令和结果、完整推理的实际覆盖、已知缺口，以及实验、保留对照或发布状态 |

版本提交同时更新 [CHANGELOG](../CHANGELOG.md)。新增、迁移或删除源码时运行仓库根目录的 `python tools/maintenance/build_code_index.py` 并提交更新后的[文件索引](CODE_INDEX.md)。

## 本地归档的复现入口

| 需要复现的工作 | F 盘亚米遥感系统中的入口 |
|---|---|
| 西安多期应用 | 01_源码与应用 / XIAN_R5_APPLICATION / APPLICATION_V1_20261027_01 / MIGRATION_20261123。先取迁移版 YAML 和模型路由清单。 |
| 城市数据与交付 | 总目录中的城市成果索引.csv 按城市、产出批次和实际拍摄日期定位。西安单期三期批次见 02_城市成果 / 西安 / 西安2026-1028。 |
| 早期建筑识别 | 总目录中的建筑轮廓识别专题.md 与建筑轮廓识别资料清单.csv 连接 WHU、RSBuilding 权重、推理入口、参考矢量和早期 TIF。 |
| 2019 训练与治理 | 05_训练记录与模型 / 西安 / 西安2026-1028 / XIAN_R5_2019_CLOUD_GOVERNANCE_TRAIN_20261028_01。 |
| 11 月研究轮次 | 05_训练记录与模型 / 西安 / 西安2026-1111，按 OPTIMIZE_MODEL_20261111_R 前缀和轮次定位。R7–R12 的历史方案、准备记录和未公开资产从此查阅。R13–R23 在 Git 中按任务归档已发布的实验源码。 |
| 权重、配置、曲线的完整性 | 05_训练记录与模型 / 训练轮次完整性.csv 与训练轮次补齐清单_20261123.csv。部分历史记录需按表核查缺件。 |

研究目录使用“轮次 + 任务”命名。脚本名使用“动作 + 对象”，例如 train_roof_model.py、evaluate_building_association.py。早期脚本的全路径可用 Git 的重命名记录查找，现行工具的稳定命令仍从 tools 进入。
