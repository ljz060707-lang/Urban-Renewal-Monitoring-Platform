# 研究与历史复盘

各目录以研究对象命名，保留 R 轮次以对上 F 盘原回执。各轮脚本共享当时的输入契约、训练数据与权重。复现实验时按同轮回执绑定本地资产。

| 目录 | 研究内容与版本状态 |
|---|---|
| [building_recognition](building_recognition/) | R15 解码器精修、R16 原生窗口屋顶、R17 独立候选补检、R18 小尺度细节。当前应用入口仍以 B0 为保留基线。 |
| [building_governance](building_governance/) | 西安 2019 全市外立面与位移研究，R13 多视角几何，R19 LOFT，R20 分栋/配准/变化分支诊断，R21 成对分栋对照，R22 备选切口，R23 切口选择器续训。 |
| [temporal_tracking](temporal_tracking/) | 10 月 27 日全序列时序关联、训练和部署候选一致性改造，以及 R14 的原生屋顶关联实验。 |
| [historical_xian_reviews](historical_xian_reviews/) | 既有西安诊断、案例图、候选身份和缺失证据复盘脚本。脚本保留真实历史输入前提，云端历史路径需要映射到同一批产物。 |

[模块状态与资产索引](../docs/MODULE_STATUS.md)说明每条路线的代码入口、训练参考和验证范围。[文件索引](../docs/CODE_INDEX.md)可逐文件查找入口和主要符号。
