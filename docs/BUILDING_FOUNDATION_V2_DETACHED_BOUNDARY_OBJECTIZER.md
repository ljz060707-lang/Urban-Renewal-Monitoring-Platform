# Building Foundation v2：Detached Boundary Objectizer

本实验把 B0 的建筑像素输出与一个独立 Boundary head 分开。Boundary head
只读取 `decode_head.fusion_ab`（256 通道、stride=4）的 detached feature，不能
回传或修改 B0 encoder、decoder、building head 或任何共享参数。Boundary 只能
改变 B0 建筑像素内部的 instance id，不能改变 building probability、mask 或面积。

## 治理与运行

```powershell
$PY = "<repo>\.venv-gpu\Scripts\python.exe"
& $PY tools\init_building_v2_detached_boundary_objectizer.py `
  --root "D:\Data Set\WHU_Building"
& $PY tools\run_building_v2_detached_boundary_objectizer.py `
  --stage all `
  --run-dir "<run>"
```

初始化只读取 `WHU_Building\train`。固定 pilot split 为 Boundary Train 256、
Calibration 128、Diagnostic 32，seed 为 20260805。重复 label/image hash 会整组
排除并写入 registry，避免重复文件跨 split 泄漏。WHU Validation、WHU Test 和
SECOND Test 都不会在初始化阶段读取。

正式 GPU 阶段前有硬门：Building-v2 源码必须已经提交且 GPU lock 可获得。
未提交时命令会生成 `blocked_detached_boundary_training_failed` receipt，附带
`governance_gate=blocked_dirty_source_requires_commit`，不会偷偷训练或写成
候选。用户应先审阅并本地提交源码，再重新运行同一个治理/训练命令；本项目
不会自动 commit 或 push。

## 科学门控

Boundary 使用 symmetric width=2、ignore=255、BCE+Dice(P0/L0)。三个 seed、最多
2000 steps 和固定 checkpoint 间隔由 contract 预注册。Calibration 完成后才可以
冻结 boundary threshold、seed/minimum area、watershed 参数和对象化方法。对象化
比较必须包含 O0 connected components、O1 morphology、O2 learned watershed，O3
GT oracle 只能作上限诊断，不能选择。最终晋升必须由 object F1、split/merge、
count/area error 及 image-level bootstrap 支持；BF1 单独提高不构成晋升。

当前 run `building_foundation_v2_detached_boundary_objectizer_20260806_002117`
已经完成治理和 hash-independent split 审计：4736 条 WHU Train 配对，256/128/32
固定划分，419 条重复 label hash 整组排除，critical leakage=0。正式训练因源码
未提交而阻断，因此当前没有 detached Boundary loss、BF1、对象指标、bootstrap
或 WHU Validation 结果。B0 SHA256 仍为
`f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7`。
