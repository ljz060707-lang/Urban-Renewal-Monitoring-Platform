# Model Catalog

| Model | Task | Status | Test | Checkpoint |
|---|---|---|---|---|
| `dsifn_binary_v1` | any_change | `archived` | `spent_test` | `outputs/pretraining/change/20260725_232317/checkpoints/best_binary_change.pth` |
| `s2looking_directional_v1` | any_change, new_building, demolition | `archived` | `spent_test` | `outputs/models/urban_change_v1_s2looking/checkpoints/best_directional_change.pth` |
| `full20_multisource_v2` | any_change, new_building, demolition, semantic_t1, semantic_t2, uncertainty | `validation_candidate` | `locked_holdout` | `outputs/pretraining/change_v2/20260727_001029/checkpoints/best_v2_multisource_change.pth` |
| `rsbuilding_swin_t_pretrained` | building_probability, building_mask, building_boundary | `external_pretrained` | `locked_holdout` | `D:/Data Set/RSBuilding/weights/Swin_T.pth` |
| `rsbuilding_swin_t_whu_candidate` | building_probability, building_mask, building_boundary | `blocked` | `locked_holdout` | `none` |
