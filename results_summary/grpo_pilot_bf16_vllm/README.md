# GRPO 试跑（pilot）：bf16 LoRA + vLLM（A100）

**目的不是看效果**，而是确认 vLLM 加速下训练是否正常、测出每步耗时和显存，用来估算正式训练的开销。64 题 × 30 步训练出来的模型没有参考价值，**不要拿这里的 reward 判断 GRPO 是否有效**。

和 `results_summary/grpo_pilot/`（T4，QLoRA，不开 vLLM）的唯一实质区别是：**基座不量化（bf16）+ 用 vLLM 生成**。数据、超参数完全相同。

完整输出（含 LoRA adapter、`train_log.txt`）在 Google Drive `medqa-calibration/grpo_pilot/grpo_pilot_bf16_vllm_A100-SXM4-40GB_20261030_004033/`，`train_log.txt` 也复制到了本地 `outputs/grpo_pilot/` 下同名目录，不提交到 git。

## 为什么不用 QLoRA + vLLM

先在 A100 上试过 QLoRA（4-bit）+ vLLM（结果只留在 Drive）。每步约 30.8 秒，但**训练后半程实际上没在学习**：

| 指标 | QLoRA + vLLM | bf16 LoRA + vLLM（本次） |
|---|---|---|
| `sampling/sampling_logp_difference/mean`（vLLM 与训练模型的差异） | 0.025 → 0.27，持续上涨 | 30 步一直 0.022–0.025 |
| `sampling/importance_sampling_ratio/mean` | 0.46 → 2e-7 | 一直在 0.5–1.2 |
| grad_norm（第 14 步以后） | 约 0.001 | 0.06–0.29 |
| 每步耗时 | 30.8 秒 | **8.6 秒** |

原因（推测，与所有指标一致）：TRL 每步把 LoRA 合并进 4-bit 权重同步给 vLLM 再拆开，4-bit 的舍入误差逐步累积，vLLM 生成用的模型和训练模型越差越远，TRL 的 importance sampling 修正把 loss 权重压到接近 0。不量化时没有这个问题。

另外两个环境问题（已在 `notebooks/grpo_pilot.ipynb` 第 4 格处理）：vLLM ≥ 0.28 删除了 bitsandbytes 支持，固定 `vllm==0.27.1`；Colab 预装的 torchao 0.10 会让 peft 给不量化模型加 LoRA 时报错，直接卸载。

## 设置

| 项目 | 值 |
|---|---|
| 模型 | Qwen/Qwen2.5-3B-Instruct，**bf16，不量化**（`--quantization none`） |
| LoRA | r = 16，alpha = 32，dropout = 0.05，target = all-linear |
| 数据 | GBaker/MedQA-USMLE-4-options **train** split，seed 42 抽 64 题 |
| 训练 | 30 步；每步 4 题 × 4 个回答 = 16 条（per_device 4 × grad_accum 4）；64 题约过 1.9 遍 |
| 生成 | **vLLM colocate**（gpu_memory_utilization 0.3），max_completion_length = 512，temperature = 1.0 |
| 优化 | lr = 1e-5（linear），beta（KL）= 0.04，loss_type = dapo，scale_rewards = group |
| reward | 只看正确性：提取最后一个 `Answer: X`，对 = 1，错或没按格式 = 0 |
| 环境 | Colab A100-SXM4-40GB，bf16；TRL 1.16.0，vllm 0.27.1，transformers 5.20.2，peft 0.21.2，torch 2.13.0 |

完整配置见 `pilot_report.json` 的 `config` 和 `effective_grpo_config`。

## 结果

| 指标 | 结果 |
|---|---|
| 总训练时间 | 4.3 分钟 |
| 每步耗时（去掉前 2 步预热） | **8.56 秒**（T4 不开 vLLM 为 97.7 秒，A100 不开 vLLM 约 77 秒） |
| 显存峰值 allocated / reserved | **19.2 / 19.4 GB**（共 39.5 GB，含 vLLM 预留） |
| 稳定性 | 无 NaN/Inf loss；grad_norm 0.06–0.29（第 7 步 0.0009 是因为 4 组全相同，本来就没有梯度）；KL 最大 0.0010 |
| reward（第 1 步 / 最后 5 步均值 / 全程均值） | 0.375 / 0.46 / 0.48 |
| 格式正确率（第 1 步 / 最后 5 步均值） | 0.81 / 0.90 |
| 截断比例（写满 512 token，全程平均） | 0.09 |
| 组内 reward 全相同的比例（全程 / 最后 5 步） | 0.40 / 0.20 |
| 平均回答长度 | 391 token |

## 正式训练估算（A100，bf16 LoRA + vLLM，max_completion_length = 512）

| 方案 | 总步数 | 估计耗时（GPU 小时） |
|---|---|---|
| 2000 题 × 1 epoch，每题 4 个回答 | 500 | 1.4 |
| 5000 题 × 1 epoch，每题 4 个回答 | 1250 | 3.4 |
| 全量 10178 题 × 1 epoch，每题 4 个回答 | 2545 | 7.0 |

估算乘了 1.15 的额外开销系数，公式见 `train/grpo/estimate.py`。**改成 max_completion_length = 1024 后每步会变慢**（一步要等最长的回答写完），粗估每步 12–15 秒；正式训练开始后用前 10 步的实际耗时重新估算。

## 结论

- **bf16 LoRA + vLLM 可以用于正式训练**：训练正常，速度是 QLoRA 不开 vLLM 的约 9 倍，显存余量大。
- **QLoRA + vLLM 不要用**：后半程梯度几乎为 0。
- **可比性**：GRPO 改用 bf16 LoRA 后，SFT / DPO 最好也用 bf16 LoRA，三种方法的差别才只在训练算法上；需要和队友确认。另外 `results_summary/base/` 的基线评估用的是 4-bit 加载，训练后模型的评估设置也要统一。
- 截断仍有约 9%，正式训练建议 `max_completion_length = 1024`。
- 训练后的一个样例里混入了一段中文（Qwen 在 temperature 1.0 采样时偶尔出现），正式训练时留意出现频率。

## 文件

| 文件 | 内容 |
|---|---|
| `pilot_report.json` | 完整设置、TRL 实际配置、耗时、显存、reward、稳定性、训练前后各 2 个生成样例、全量训练估算 |
| `metrics.csv` | 每步一行：耗时、显存、reward 均值/标准差、组内全对/全错/全相同比例、格式正确率、回答长度、截断比例、KL、loss、grad_norm 等 |
| `curves.png` | 上述主要指标随步数变化的曲线 |
| `grpo_pilot.ipynb - Colab.pdf` | Colab 运行记录（本地参考，`*.pdf` 被 git 忽略） |
