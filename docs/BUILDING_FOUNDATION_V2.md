# Building Foundation v2

该扩展用于静态建筑基础候选，不替代 Full20、Construction、TFR/MTI 或 RSBuilding 原始 checkpoint。

## 运行顺序

```powershell
$PY = ".venv-gpu\Scripts\python.exe"
& $PY tools\audit_globfp_dataset.py --root "D:\Data Set\3D-GloBFP" --output "outputs\pretraining\building_foundation_v2_<audit>" --no-hash
& $PY tools\pair_globfp_with_local_imagery.py --globfp-root "D:\Data Set\3D-GloBFP" --audit-root "outputs\pretraining\building_foundation_v2_<audit>\01_globfp_audit" --imagery-root "D:\Data Set\Satellite\nanjing_jiangbei 20" --imagery-root "D:\Data Set\Satellite\beijing_tongzhou" --output "<RUN>\02_pairing"
& $PY tools\train_building_foundation_v2.py --config "configs\building_foundation_v2_smoke.yaml" --stage smoke --device cuda
& $PY tools\validate_building_foundation_v2.py --run-dir "<RUN>" --device cuda
& $PY tools\package_building_foundation_v2.py --run-dir "<RUN>" --baseline-checkpoint "outputs\v2_readiness\rsbuilding\whu\work_dirs\swin_t_whu_pilot5000_20260728_110750\iter_2000.pth"
& $PY tools\create_building_v2_qa_queue.py --run-dir "<RUN>"
& $PY tools\finalize_building_v2_run.py --run-dir "<RUN>"
```

训练适配器只访问 WHU `train` 和 `val`，不会枚举 `test`。3D-GloBFP 当前只有省级 Polygon Shapefile 和未确认单位的 `Height` 字段，若没有通过空间/语义审计的影像配对，不能进入图像监督或高度损失。

候选必须通过有限值、类别坍缩、泄漏、checkpoint 迁移和 Test 隔离门控。若没有实质改善，系统保留 iter2000，并标记 `baseline_retained_no_material_improvement`。南京和通州只能在冻结候选后做诊断 proxy，不能选择模型或阈值。

单影像静态推理（不会读取 Test，且无 CRS 时只输出像素坐标结果）：

```powershell
& $PY tools\infer_building_foundation_v2.py --checkpoint "<CHECKPOINT>" --image "<IMAGE>" --output "<OUTPUT>"
```
