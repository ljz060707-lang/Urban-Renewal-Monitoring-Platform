# Dataset Catalog

| Dataset | Role | Train | Val | Test policy | Status |
|---|---|---:|---:|---|---|
| `dsifn` | binary_change_teacher | 3600 | 340 | `spent_test` | `active` |
| `s2looking` | directional_change_teacher | 3500 | 500 | `spent_test` | `active` |
| `second` | semantic_and_directional_teacher | 2668 | 300 | `locked_holdout` | `development_ready` |
| `whu_building` | geometry_teacher | 4736 | 1036 | `locked_holdout` | `assets_ready` |
| `rsbuilding_pretraining` | external_geometry_teacher | None | None | `external_pretrained` | `blocked` |
| `beijing_tongzhou` | external_real_scene | 0 | 0 | `not_a_benchmark_split` | `external_imagery` |
| `beijing_waste_future` | future_surface_state_teacher | None | None | `undefined_until_onboarding` | `planned` |
