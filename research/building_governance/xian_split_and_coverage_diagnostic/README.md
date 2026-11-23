# 西安分栋可达性与零覆盖诊断

replay.py 从 R23 保存的 INPUTS.tar 和 TRAIN_CACHE.tar 读取 72 个开发窗口。它复现 R21 的原生分栋及冻结位移，对 R22 的已有切口逐条重建治理后几何，按支持区裁剪并计算 IoU。整窗混合整数规划同时约束父对象只能选择一个方案、每个子对象和参考地块最多匹配一次，并保留原匹配。两子块各占同一参考面积至少 10% 的切口被排除。结果衡量已有备选在人工参考引导下的可达性。

summarize.py 连回 R16 的完全零覆盖台账，生成按实拍日期、形状尺度、候选距离、窗口纹理分组的汇总与三期影像对照。纹理使用 RGB 转灰度后的 Laplacian 标准差。R17/R18 读取到的是候选回执计数。

train_cut_selector.py 使用 R23 封存的 MAP_FIT 和 MAP_INNER 训练和评估双线性头，分别估计安全收益和伤害。冻结门槛只由 MAP_INNER 决定，未知结果不参与拟合并单独计入门槛代价。

在项目 Python 环境安装 NumPy、SciPy、Shapely、Rasterio、PyProj、Pyogrio、OpenCV 和 Pillow。位置参数为本地对应轮次的实际目录。

```powershell
python replay.py <R16> <R22> <R23> <新结果目录>
python summarize.py <结果目录> <R7> <R16> <R17> <R18> <R23>
python train_cut_selector.py <R23> <结果目录>
```

本轮输入使用 F 盘 05_训练记录与模型/西安/西安2026-1111 下的对应轮次，结果位于 06_验收与审阅/西安/20261124_分栋与零覆盖诊断。质量判定详见结果目录中的报告。工作用到的参考为指定人工地图，空间接触过的窗口均按开发样本记录。
