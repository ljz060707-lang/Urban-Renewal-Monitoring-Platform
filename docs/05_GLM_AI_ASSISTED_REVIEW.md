# GLM 视觉辅助审核

GLM 是第二审核员，不是 Ground Truth。它不能修改 S2Looking 标签、不能自动写入人工决定，也不能直接把样本加入训练。

可选依赖和临时密钥：

```powershell
pip install -e ".[ai_review]"
$env:ZAI_API_KEY = "你的API密钥"
python run.py change-ai-review --run 20260721_232528 --config configs/change_public_s2looking_windows.yaml --max-samples 10 --dry-run
```

dry-run 只在本地生成 2×3 联系图。真实调用只发送当前联系图（T1、T2、GT、三类预测），不发送绝对路径、Windows 用户名、EXIF 或完整数据集。请先确认影像许可和隐私要求；敏感影像默认不要启用远程审核。

真实调用建议从 20 张开始：

```powershell
python run.py change-ai-review --run 20260721_232528 --config configs/change_public_s2looking_windows.yaml --model glm-4.6v-flash --max-samples 20 --resume
```

结果在 `review/glm_scores.jsonl`，调用情况/令牌字段在 `review/ai_usage.csv`。缓存以样本、联系图哈希、模型、提示词版本和阈值为键。没有 API key 时人工审核仍完全可用。

