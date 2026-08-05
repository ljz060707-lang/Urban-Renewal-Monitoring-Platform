# v0.5.1 训练就绪审计

审计日期：2026-07-21。审计范围为公开双时相变化教师和个人 AOI 建筑实例教师；不修改 `D:/Data Set` 中的原始数据。

## 修复前发现的阻断点

- `change-all` 以固定三阶段和 `results["directional"]` 为前提，无法诚实处理 S2Looking-only 或 DSIFN-only 监督。
- validation 与 test 存在合并风险，quick 只缩短 epoch，不能代表受控的小样本通路。
- SECOND 的映射确认只停留在配置层，缺少正式入口的阻断；S2Looking 的 RGB 方向标签也不能被通用二值读取器安全解析。
- 个人 AOI 命令假定仓库内 `data/`，且 GPKG 多图层没有确定性建筑图层选择。

## 本机实测数据与环境

- `D:/Data Set/S2Looking`：train 3500、validation 500、test 1000；全量只读检查发现 5000 对可读取样本，方向标签与总变化标签最大不一致率为 0。
- `D:/Data Set/SECOND`、`D:/Data Set/DSIFN`、`D:/Data Set/Personal`：不存在，故不能报告真实训练成功。
- `.venv-gpu`：Python 3.11.9、Torch 2.12.1+cu130、Torchvision 0.27.1+cu130、CUDA 13.0、RTX 4060 Laptop GPU（8188 MiB）。

## 审计后的验收

完整命令、测试输出、GPU quick run 路径与未执行项记录在 [最终验收报告](03_FINAL_ACCEPTANCE_REPORT.md)。结论为：S2Looking-only quick 已通过；三教师全量训练和个人真实 AOI 训练仍由缺失数据阻断。
