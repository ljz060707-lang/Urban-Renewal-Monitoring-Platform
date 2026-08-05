# URRS Building Foundation v2

## 实际收口

本轮新增了可审计的静态建筑基础模型扩展：3D-GloBFP 资产审计、保守配对、WHU train/validation 数据适配、有限值多头网络、训练/验证 checkpoint、候选包和固定人工 QA 队列。

最终收口 run：`outputs/pretraining/building_foundation_v2_20260805_162958`（早先的 `153816` run 保留不覆盖）。

3D-GloBFP 实际为 33 个 Polygon Shapefile，含 `Height` 字段但没有影像文件；全部分类为 `C_FOOTPRINT_HEIGHT_ONLY`，高度单位与影像配对未确认，`trainable_image_pairs=0`，因此没有进入图像分割训练。WHU Test 未枚举、未读取。

配对审计：`PAIR_EXACT=0`、`PAIR_STRONG=0`、`PAIR_GEOMETRY_ONLY=0`、`PAIR_REJECTED=33`，命令以预期的 blocked exit code `3` 结束。

GPU smoke 只使用 WHU train/validation 的 16/8 样本、1 epoch；完整 WHU Validation 使用 1036 张。原生 smoke 模型的验证预测正像素比例为 1.0，属于类别坍缩，未晋升。冻结的 RSBuilding `iter_2000.pth` SHA256 保持为 `f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7`，并以 `baseline_retained_no_material_improvement` 打包为候选参考。

## 验收证据

- 训练 checkpoint：`outputs/pretraining/building_foundation_v2_20260805_162958/06_training/checkpoints/best.pth` 与 `last.pth`，均可加载；best SHA256 `03bb5b8876a3ef1d4550e29576de48d4fe855dc47b4254448f73f8029474a534`，last SHA256 `3dbc6b77258d63126023058d0eacdf97dd6a9a618aece61891a64b789572e400`。
- Validation：precision `0.8900073`、recall `1.0`、F1 `0.9369014`、IoU `0.8900073`；`predicted_positive_fraction=1.0`，故类别坍缩门控拒绝候选。
- 固定 QA：`11_final_review/FINAL_QA_CASES.csv` 共 220 条，`lite_review_database_modified=false`，仅 validation。
- Test 安全：`test_used=false`、`locked_test_pixel_reads=0`；没有启动伪标签、TFR、MTI 或 Construction 训练。
- 保护资产：Full20、registry、原始数据和 RSBuilding checkpoint 未修改；最终 GPU smoke 的 `00_governance/GPU_LOCK_AUDIT.json` 记录 `required=true, acquired=true, released=true`。
- `12_acceptance/SHA256SUMS.txt` 同时记录 best、last 和打包的 iter2000 baseline hash。
- 完整测试：`307 passed, 3 deselected, 56 warnings`；Building v2 定向测试 `15 passed`。
- 配置/CLI 回归 smoke：`outputs/pretraining/building_foundation_v2_20260805_161210`，用 `--config configs/building_foundation_v2_smoke.yaml --device cpu` 完成 1 epoch，生成 `resolved_config.yaml`，无 Test 读取。
- Pilot/controlled full training：未启动；原因是 GloBFP 无可训练影像配对且 native smoke 已触发类别坍缩/源 checkpoint 迁移门控。

## 科学边界

正式科学状态：`baseline_retained_no_material_improvement`。由于 `iter_2000.pth` 是 MMEngine/OpenCD 格式，当前项目 `.venv-gpu` 未导入 RSBuilding 运行时，本轮仅记录外部基线并拒绝静默迁移；严格 source checkpoint migration 与最终人工 QA 仍未完成。南京、通州输出只做冻结后的 proxy 诊断，不能用于 checkpoint、阈值或损失选择；不能声称全国泛化、Ground Truth 或生产可用。
