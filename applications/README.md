# 应用入口

- [single_date](single_date/) 负责单期城市网格、B0 屋顶推理、参考栅格与产品合并。调度在 xian_single_date/supervisor.py，模型调用在 b0.py，实例与面积产品在 assemble.py，资料绑定辅助程序在 write_bindings.py 和 write_configs.py。
- [xian_r5](xian_r5/) 存放多期应用的输出字典、实现映射、模型与数据角色，以及合成端到端测试。应用实现位于 [src/xian_r5](../src/xian_r5/)，命令入口为 [run_xian_r5_integration.py](../tools/run_xian_r5_integration.py)。

使用方式和本地清单要求见[亚米遥感系统代码基线](../docs/SUBMETER_SYSTEM.md)。
