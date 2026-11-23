# 西安生产核心

| 入口或目录 | 责任 |
|---|---|
| application.py、application_contract.py | 应用调度、清单、输入校验、执行和封存。 |
| application_models.py、cloud_joint | 屋顶、外立面和对位的运行适配，城市分块与模型推理。 |
| alignment、application_geometry.py | 地面配准、实例位移、坐标和轮廓治理。 |
| tracking、roof_identity.py、application_temporal.py | 实际时间排序、屋顶匹配、轨迹和区间状态。 |
| reconstruction、resolver | 多源证据约束、几何重建和状态解析。 |
| training、reference*、transfer_admission.py | 监督数据校核、交叉时相迁移和训练数据准入。 |
| label_transfer.py、fraction_shadow.py | 精确面积分数和面积质量层。 |
| acceptance.py、integration.py | 证据与交付检查、子模块接线。 |

完整模块、应用和研究链接见[模块状态](../../docs/MODULE_STATUS.md)，单文件职责见[文件索引](../../docs/CODE_INDEX.md#src)。
