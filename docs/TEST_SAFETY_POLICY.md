# Test Safety Policy

## 当前状态

| Dataset | Test status | Development policy |
|---|---|---|
| DSIFN | `spent_test` | 永久禁止再次读取或用于任何开发决策 |
| S2Looking | `spent_test` | 永久禁止再次读取或用于任何开发决策 |
| SECOND | `locked_holdout` | 训练、验证、调阈值、模型选择、定性审核均禁止 |
| WHU | `locked_holdout` | 训练、验证、调阈值、模型选择、定性审核均禁止 |

`src/urban_monitor/safety/test_guard.py` 会在 training、validation、threshold tuning、model selection 和 qualitative review 阶段阻止 Test 访问。

未来确需一次正式评估时，必须同时满足：

1. 数据集状态仍是 `locked_holdout`，不是 `spent_test`；
2. phase 明确为 `final_test`；
3. 显式传入 `--allow-final-test`；
4. 写出包含路径、时间、决策和读取次数的审计。

已有 change detection final-test 还受其原有一次性 receipt 保护。本轮没有重新执行任何 Test。
