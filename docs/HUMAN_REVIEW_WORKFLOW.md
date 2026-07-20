# 人工审核与修正回流

## 静态审核

训练评估后生成 `review/index.html`。页面完全离线，支持：

- 数据集、stage、错误类型筛选；
- 按 F1、IoU、不确定性排序；
- T1/T2 与阈值预测叠加；
- 透明度控制；
- 审核状态、中文备注、审核人；
- localStorage 自动保留；
- 导出 `review_decisions.csv`。

状态为 `correct`、`prediction_error`、`label_error`、`misregistration`、`uncertain`、`exclude`、`needs_mask_edit`。

## 可选交互审核

```bash
pip install -e ".[review]"
python run.py change-review --run latest --config configs/change_public.yaml --interactive
```

Gradio 工具显示 T1、T2、预览并允许上传修正的 binary/new/demolition mask。所有修正写入：

```text
review/corrections/
├── binary/
├── new/
├── demolition/
└── corrections.jsonl
```

每条修正记录 sample、任务、修正 hash、审核人、时间、备注和版本。原公共数据不修改。

## 回流规则

```bash
python run.py change-retrain --run latest --from-review --device cuda:0
```

优先级：`human_corrected_label > original_dataset_label`。

- `exclude`、`misregistration` 被排除。
- `label_error` 没有修正 mask 时被排除。
- `prediction_error` 保留并提高困难样本权重。
- 重训继承上一阶段 best，不随机初始化。
- 新建 run，旧 run 保持不变。

输出 reviewed index、correction audit 和 `before_after_review.md`。

## 本地变化候选

```bash
python run.py change-labels --aoi beijing_demo
```

相邻日期建筑差分生成 new/demolition/change candidate、建筑 mask 与审核页。candidate 必须人工确认；未确认项不是 GT。

重点检查屋顶侧移、阴影、树木遮挡、registration、merge、split、施工过渡、轮廓轻微差异和分割错误。

## 联合审核

双教师联合审核可设置最终状态：`stable`、`new_building`、`demolished`、`construction_candidate`、`uncertain`、`rejected`，并判断哪个教师正确。

应用审核后：

- `state_final` 更新；
- `rejected` 不进入 `student_fusion_labels.gpkg`；
- reviewer notes/time 保留；
- 重新生成 reviewed fusion report。
