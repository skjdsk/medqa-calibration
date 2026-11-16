# GRPO pilot: bf16 LoRA + vLLM (A100)

**This isn't about results.** The point was to check that training works with vLLM speeding up generation, and to measure time per step and memory so we could estimate the cost of the full run. A model trained on 64 questions for 30 steps doesn't tell you anything, so **don't use the reward here to judge whether GRPO works**.

The only real difference from `results_summary/grpo_pilot/` (T4, QLoRA, no vLLM) is: **the base model isn't quantized (bf16) + generation runs on vLLM**. Data and hyperparameters are exactly the same.

The full output (LoRA adapter, `train_log.txt`) isn't in the repo.

## Why not QLoRA + vLLM

We tried QLoRA (4-bit) + vLLM on an A100 first (those results aren't in the repo). It took about 30.8 s per step, but **in the second half of training it wasn't actually learning**:

| Metric | QLoRA + vLLM | bf16 LoRA + vLLM (this run) |
|---|---|---|
| `sampling/sampling_logp_difference/mean` (gap between vLLM and the training model) | 0.025 -> 0.27, kept climbing | stayed at 0.022-0.025 for all 30 steps |
| `sampling/importance_sampling_ratio/mean` | 0.46 -> 2e-7 | stayed between 0.5 and 1.2 |
| grad_norm (after step 14) | about 0.001 | 0.06-0.29 |
| Time per step | 30.8 s | **8.6 s** |

Our best guess (it fits every metric): every step TRL merges LoRA into the 4-bit weights, syncs them to vLLM, then unmerges. The 4-bit rounding error piles up, so the model vLLM generates with drifts further and further from the one being trained, and TRL's importance-sampling correction pushes the loss weights down to almost 0. Without quantization this doesn't happen.

Two other environment issues (handled in cell 4 of `notebooks/grpo_pilot.ipynb`): vLLM >= 0.28 dropped bitsandbytes support, so we pin `vllm==0.27.1`; and Colab's preinstalled torchao 0.10 makes peft error out when adding LoRA to an unquantized model, so we just uninstall it.

## Setup

| | |
|---|---|
| Model | Qwen/Qwen2.5-3B-Instruct, **bf16, no quantization** (`--quantization none`) |
| LoRA | r = 16, alpha = 32, dropout = 0.05, target = all-linear |
| Data | 64 questions from the GBaker/MedQA-USMLE-4-options **train** split, seed 42 |
| Training | 30 steps; 4 questions x 4 answers = 16 per step (per_device 4 x grad_accum 4); about 1.9 passes over the 64 questions |
| Generation | **vLLM colocate** (gpu_memory_utilization 0.3), max_completion_length = 512, temperature = 1.0 |
| Optimization | lr = 1e-5 (linear), beta (KL) = 0.04, loss_type = dapo, scale_rewards = group |
| Reward | correctness only: take the last `Answer: X`, right = 1, wrong or bad format = 0 |
| Environment | Colab A100-SXM4-40GB, bf16; TRL 1.16.0, vllm 0.27.1, transformers 5.20.2, peft 0.21.2, torch 2.13.0 |

The full config is under `config` and `effective_grpo_config` in `pilot_report.json`.

## Results

| Metric | Result |
|---|---|
| Total training time | 4.3 min |
| Time per step (first 2 warmup steps dropped) | **8.56 s** (97.7 s on a T4 without vLLM, about 77 s on an A100 without vLLM) |
| Peak memory allocated / reserved | **19.2 / 19.4 GB** (out of 39.5 GB, including what vLLM reserves) |
| Stability | no NaN/Inf loss; grad_norm 0.06-0.29 (0.0009 at step 7 because all 4 groups had the same reward, so there was no gradient anyway); max KL 0.0010 |
| Reward (step 1 / mean of last 5 / overall mean) | 0.375 / 0.46 / 0.48 |
| Format rate (step 1 / mean of last 5) | 0.81 / 0.90 |
| Truncation rate (hit 512 tokens, overall mean) | 0.09 |
| Share of groups where every reward is the same (overall / last 5) | 0.40 / 0.20 |
| Mean answer length | 391 tokens |

## Full-run estimates (A100, bf16 LoRA + vLLM, max_completion_length = 512)

| Setup | Total steps | Estimated GPU hours |
|---|---|---|
| 2000 questions x 1 epoch, 4 answers each | 500 | 1.4 |
| 5000 questions x 1 epoch, 4 answers each | 1250 | 3.4 |
| all 10178 questions x 1 epoch, 4 answers each | 2545 | 7.0 |

These include a 1.15 overhead factor; the formula is in `train/grpo/estimate.py`. **With max_completion_length = 1024 each step gets slower** (a step waits for the longest answer to finish), roughly 12-15 s per step. Re-estimate from the first 10 steps of the real run.

## Takeaways

- **bf16 LoRA + vLLM is good to go for the real run**: training behaves, it's about 9x faster than QLoRA without vLLM, and there's plenty of memory left.
- **Don't use QLoRA + vLLM**: the gradients go to almost 0 in the second half.
- **Comparability**: now that GRPO uses bf16 LoRA, SFT / DPO should ideally use bf16 LoRA too, so the only difference between the three is the training algorithm. Needs checking with the team. Also, the baseline eval in `results_summary/base/` loads the model in 4-bit, so eval settings for the trained models need to match too.
- About 9% of answers still get cut off, so use `max_completion_length = 1024` for the real run.
- One of the post-training samples slipped into Chinese for a bit (Qwen does this now and then when sampling at temperature 1.0). Worth keeping an eye on how often it happens in the real run.

## Files

| File | What's in it |
|---|---|
| `pilot_report.json` | full setup, the config TRL actually used, timing, memory, reward, stability, 2 sample answers before and after training, full-run estimates |
| `metrics.csv` | one row per step: time, memory, reward mean/std, share of all-right/all-wrong/all-same groups, format rate, answer length, truncation rate, KL, loss, grad_norm, etc. |
| `curves.png` | the main metrics above plotted over steps |
