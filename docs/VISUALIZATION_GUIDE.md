# Visualization Guide

统一入口：`outputs/catalog/visual_gallery.html`。

Gallery 可按 Model、Run、Dataset、Stage、Task 和 Artifact Type 筛选，并提供“打开文件”和“复制路径”。

颜色约定：

- Any Change：orange
- New Building：red
- Demolition：blue
- Vacant Land：earth brown
- Construction：yellow
- Building：cyan
- Uncertain：gray

资产状态：

- `available`：可用于开发、汇报或 PPT。
- `locked_diagnostic_only`：来自已 spent / locked Test，只允许保留审计，不得驱动开发。
- `pending`：接口已定义但尚无真实结果。
- `unsupported`：该模型不提供此输出，不会生成假全零图。

Tongzhou 图中的对象必须标注为 candidate building-change objects。
