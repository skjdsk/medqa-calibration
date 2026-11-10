# GRPO 试跑（pilot）

**目的不是看效果**，而是在正式训练前测出：每步耗时、显存峰值、训练是否稳定、参数是否合理，并据此估算正式训练要多少 GPU 小时。64 题 × 30 步训练出来的模型没有参考价值，**不要拿这里的 reward 判断 GRPO 是否有效**。

完整输出（含 LoRA adapter 约 120 MB、`train_log.txt`）只保存在本地 `outputs/grpo_pilot/grpo_pilot_Tesla_20261025_213538/` 和 Google Drive，不提交到 git。

## 设置

| 项目 | 值 |
|---|---|
| 模型 | Qwen/Qwen2.5-3B-Instruct，4-bit nf4（QLoRA） |
| LoRA | r = 16，alpha = 32，dropout = 0.05，target = all-linear |
| 数据 | GBaker/MedQA-USMLE-4-options **train** split，seed 42 抽 64 题 |
| 训练 | 30 步；每步 4 题 × 4 个回答 = 16 条（per_device 4 × grad_accum 4）；64 题约过 1.9 遍 |
| 生成 | max_completion_length = 512，temperature = 1.0 |
| 优化 | lr = 1e-5（linear），beta（KL）= 0.04，loss_type = dapo，scale_rewards = group |
| reward | 只看正确性：提取最后一个 `Answer: X`，对 = 1，错或没按格式 = 0 |
| 环境 | Colab Tesla T4（14.56 GB），fp16，gradient checkpointing；TRL 1.16.0，transformers 5.20.2，peft 0.21.2，bitsandbytes 0.50.2，torch 2.11.0 |

完整配置（包括 TRL 实际生效的参数）见 `pilot_report.json` 的 `config` 和 `effective_grpo_config`。TRL 没有忽略任何参数（`dropped_config_keys` 为空）。

## 结果

| 指标 | 结果 |
|---|---|
| 总训练时间 | 48.9 分钟 |
| 每步耗时（去掉前 2 步预热） | **97.7 秒** |
| 显存峰值 allocated / reserved | **3.83 / 4.32 GB**（T4 共 14.56 GB） |
| 稳定性 | 无 NaN/Inf loss；最大 grad_norm 0.295；KL 最大 0.00018 |
| reward（第 1 步 / 最后 5 步均值 / 全程均值） | 0.44 / 0.375 / 0.42 |
| 格式正确率（第 1 步 / 最后 5 步均值） | 1.00 / 0.875 |
| 截断比例（写满 512 token，全程平均） | 0.10 |
| 组内 reward 全相同的比例（全程 / 最后 5 步） | 0.39 / 0.30 |
| 平均回答长度 | 367 token |

## 正式训练估算（T4，按本次每步耗时）

| 方案 | 总步数 | 估计耗时（GPU 小时） |
|---|---|---|
| 全量 10178 题 × 1 epoch，每题 4 个回答 | 2545 | 79.4 |
| 5000 题 × 1 epoch，每题 4 个回答 | 1250 | 39.0 |
| 2000 题 × 1 epoch，每题 4 个回答 | 500 | 15.6 |
| 全量 10178 题 × 1 epoch，每题 8 个回答 | 2545 | 158.8 |

估算假设耗时与每步回答数成正比，并乘了 1.15 的额外开销系数，偏保守。公式见 `train/grpo/estimate.py`。

## 结论

- **代码和配置可以用**：训练稳定，显存余量很大，打分函数能正确提取答案。
- **T4 上跑全量不可行**（约 79 小时）。正式训练需要减少题数，或换 A100 并开启 vLLM 加速生成。
- **截断会拉低 reward**：格式正确率最低的几步（0.75，第 12、17、26、28 步），截断比例也最高（0.25）。写满 512 token 还没写到 `Answer: X` 的回答被记为 0 分，与"答错"混在一起。正式训练建议把 `max_completion_length` 提高到 768 或 1024；显存足够，但每步会变慢，需要重新估算。
- **约 3–4 成的组 reward 全相同**（全对或全错），这些题当步没有梯度。偏高但可接受；如果正式训练中更高，可以考虑增大每题回答数（会使耗时翻倍）。
- reward 在 30 步内没有上升趋势，符合预期（题少、步数少），不说明方法无效。

## 文件

| 文件 | 内容 |
|---|---|
| `pilot_report.json` | 完整设置、TRL 实际配置、耗时、显存、reward、稳定性、训练前后各 2 个生成样例、全量训练估算 |
| `metrics.csv` | 每步一行：耗时、显存、reward 均值/标准差、组内全对/全错/全相同比例、格式正确率、回答长度、截断比例、KL、loss、grad_norm 等 |
| `curves.png` | 上述主要指标随步数变化的曲线 |
