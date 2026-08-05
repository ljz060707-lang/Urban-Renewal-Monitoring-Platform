# 新数据集接入

未来北京废弃物、TFR 或 MTI 数据按同一流程接入：

1. 复制 `configs/datasets/*_template.yaml`，填写真实 root、文件模式与日期解析规则。
2. 阅读数据集权威说明，编写 label mapping YAML，并将 `verified` 改为 `true`。
3. 执行 metadata-only audit：

   ```powershell
   python run.py dataset-audit --config configs/datasets/your_dataset.yaml
   ```

4. 核对 label distribution、样例、split safety 和 canonical mapping。
5. 执行 `dataset-prepare` 生成确定性 manifest。
6. 先执行 surface-state smoke；通过后才允许 full training。
7. Validation report 通过后接入统一 Model API、Temporal Engine 与 Gallery。

模板不会假定 native 标签编号，也不会生成伪标签。当前北京废弃物模板因 root 和 label mapping 未配置而应明确返回 blocked。
