# GRPO pilot

**This isn't about results.** The point was to measure, before the real run, how long a step takes, how much memory it uses, whether training is stable and whether the settings make sense, and then estimate how many GPU hours the full run would need. A model trained on 64 questions for 30 steps doesn't tell you anything, so **don't use the reward here to judge whether GRPO works**.

The full output (LoRA adapter, about 120 MB, and `train_log.txt`) isn't in the repo.

## Setup

| | |
|---|---|
| Model | Qwen/Qwen2.5-3B-Instruct, 4-bit nf4 (QLoRA) |
| LoRA | r = 16, alpha = 32, dropout = 0.05, target = all-linear |
| Data | 64 questions from the GBaker/MedQA-USMLE-4-options **train** split, seed 42 |
| Training | 30 steps; 4 questions x 4 answers = 16 per step (per_device 4 x grad_accum 4); about 1.9 passes over the 64 questions |
| Generation | max_completion_length = 512, temperature = 1.0 |
| Optimization | lr = 1e-5 (linear), beta (KL) = 0.04, loss_type = dapo, scale_rewards = group |
| Reward | correctness only: take the last `Answer: X`, right = 1, wrong or bad format = 0 |
| Environment | Colab Tesla T4 (14.56 GB), fp16, gradient checkpointing; TRL 1.16.0, transformers 5.20.2, peft 0.21.2, bitsandbytes 0.50.2, torch 2.11.0 |

The full config (including what TRL actually used) is under `config` and `effective_grpo_config` in `pilot_report.json`. TRL didn't ignore any parameters (`dropped_config_keys` is empty).

## Results

| Metric | Result |
|---|---|
| Total training time | 48.9 min |
| Time per step (first 2 warmup steps dropped) | **97.7 s** |
| Peak memory allocated / reserved | **3.83 / 4.32 GB** (out of 14.56 GB on the T4) |
| Stability | no NaN/Inf loss; max grad_norm 0.295; max KL 0.00018 |
| Reward (step 1 / mean of last 5 / overall mean) | 0.44 / 0.375 / 0.42 |
| Format rate (step 1 / mean of last 5) | 1.00 / 0.875 |
| Truncation rate (hit 512 tokens, overall mean) | 0.10 |
| Share of groups where every reward is the same (overall / last 5) | 0.39 / 0.30 |
| Mean answer length | 367 tokens |

## Full-run estimates (T4, based on this time per step)

| Setup | Total steps | Estimated GPU hours |
|---|---|---|
| all 10178 questions x 1 epoch, 4 answers each | 2545 | 79.4 |
| 5000 questions x 1 epoch, 4 answers each | 1250 | 39.0 |
| 2000 questions x 1 epoch, 4 answers each | 500 | 15.6 |
| all 10178 questions x 1 epoch, 8 answers each | 2545 | 158.8 |

These assume time scales with answers per step and include a 1.15 overhead factor, so they're on the safe side. The formula is in `train/grpo/estimate.py`.

## Takeaways

- **The code and config work**: training is stable, there's plenty of memory to spare, and the reward function pulls out answers correctly.
- **A full run on a T4 isn't realistic** (about 79 hours). The real run needs fewer questions, or an A100 with vLLM for faster generation.
- **Truncation drags the reward down**: the steps with the lowest format rate (0.75, steps 12, 17, 26, 28) also had the highest truncation (0.25). Answers that hit 512 tokens before getting to `Answer: X` score 0, same as a wrong answer. For the real run, raise `max_completion_length` to 768 or 1024. Memory is fine, but steps get slower, so the estimate needs redoing.
- **About 30-40% of groups have the same reward for every answer** (all right or all wrong), so those questions give no gradient that step. A bit high but OK; if it gets worse in the real run, more answers per question would help (but doubles the time).
- Reward doesn't trend up in 30 steps. That's expected with so few questions and steps and doesn't mean the method doesn't work.

## Files

| File | What's in it |
|---|---|
| `pilot_report.json` | full setup, the config TRL actually used, timing, memory, reward, stability, 2 sample answers before and after training, full-run estimates |
| `metrics.csv` | one row per step: time, memory, reward mean/std, share of all-right/all-wrong/all-same groups, format rate, answer length, truncation rate, KL, loss, grad_norm, etc. |
| `curves.png` | the main metrics above plotted over steps |
