# GRPO results

Everything after the pilot runs. The pilot results are in `pilot/`: `pilot/qlora_t4/` (first try, QLoRA on a T4) and `pilot/bf16_vllm_a100/` (bf16 LoRA + vLLM on an A100, the setup we kept).

## Setup

| | |
|---|---|
| Model | Qwen2.5-3B-Instruct, bf16 (no quantization) + LoRA (r 16, alpha 32, dropout 0.05, all-linear) |
| Training data | the 5000 training questions in `data/medqa_split_seed42.json`, 1 epoch = 1250 steps (4 questions x 8 answers per step) |
| Reward | 1 if the final answer is right, 0 otherwise. No format reward |
| Other | beta 0.04, temperature 1.0, max 1024 new tokens, TRL 1.16.0 + vLLM 0.27.1, one A100 40GB |
| Final pick | **lr 1e-4, 1250 steps, trained twice (seed 1 and seed 2)** |

## 1. Picking the learning rate (`training/`)

We tried 4 learning rates with seed 1. lr 5e-5 and 2e-4 only ran for 300 steps, then got checked on the 500 validation questions.

| lr | reward (steps 251-300) | KL (steps 251-300) | val direct | val cot | vote8 acc | vote8 conf | vote8 ECE |
|---|---|---|---|---|---|---|---|
| 1e-5 (1250 steps) | 0.428 | 0.0013 | 0.514 | 0.540 | 0.554 | 0.730 | 0.176 |
| 5e-5 (300 steps) | 0.432 | 0.0045 | 0.506 | 0.520 | 0.540 | 0.729 | 0.189 |
| 1e-4 (300 steps) | 0.442 | 0.0161 | 0.516 | 0.522 | 0.554 | 0.724 | 0.187 |
| 2e-4 (300 steps) | 0.441 | 0.0489 | 0.502 | 0.536 | 0.534 | 0.732 | 0.200 |
| base (no training) | - | - | 0.514 | 0.516 | 0.536 | 0.707 | 0.174 |

- All 4 runs see the same questions in the same order, so comparing them over the same steps is fair. Reward bounces around a lot from step to step, mostly because some batches are just harder than others. It doesn't mean the model is getting worse.
- None of the 4 is significantly different from base on the validation set (see `eval_val/paired_vs_base.csv`). We went with 1e-4 because its KL clearly moves away from 0, so the model is actually changing, but it doesn't climb as fast as 2e-4.
- Curves for the first 300 steps: `training/lr_search_curves.png`. Numbers: `training/lr_search.csv`.

## 2. The two final runs (`training/`)

- `training/seed1_vs_seed2_curves.png` compares them. The two seeds shuffle the questions differently, so don't read the reward curves step by step, just look at the overall trend.
- Per-step logs are in `training/<run>/metrics.csv`, and the plots made during training are in `training/<run>/curves.png`.

## 3. Validation set (500 questions, `eval_val/`)

7 models: base, lr 1e-5, the three 300-step checkpoints, and the two lr 1e-4 seeds.

- Accuracy: nothing is significantly different from base.
- vote8 confidence: **goes up significantly for all 6 GRPO models** (+1.7 to +5.5 points). ECE gets worse for all of them too, but none of those changes is significant.

## 4. Test set and transfer (`eval_test/`)

MedQA test (1273 questions), MedMCQA validation (4183), and the expert-labeled PubMedQA set (1000). Models: base and the two lr 1e-4 seeds.

| Dataset | Mode | base | seed 1 | seed 2 |
|---|---|---|---|---|
| MedQA | direct | 0.490 | 0.497 | **0.507** |
| | cot | 0.511 | 0.527 | 0.533 |
| | vote8 | 0.535 | 0.552 | 0.545 |
| PubMedQA | direct | 0.643 | **0.669** | 0.652 |
| | cot | 0.698 | 0.704 | 0.701 |
| | vote8 | 0.735 | 0.732 | 0.741 |
| MedMCQA | direct | 0.476 | **0.484** | **0.482** |
| | cot | 0.469 | **0.491** | **0.483** |
| | vote8 | 0.499 | **0.517** | 0.506 |

Bold means the paired comparison against base is significant: the bootstrap 95% interval doesn't include 0, and the net number of flipped questions clears the sign-test line.

What both seeds agree on:
1. **Accuracy only goes up reliably on MedMCQA**: +0.7 to +0.9 points for direct and +1.4 to +2.2 for cot. On MedQA and PubMedQA the changes are mostly positive, but only one seed is significant each time.
2. **vote8 confidence goes up more than accuracy does**: +3.2 / +5.2 points on MedQA and +2.4 / +3.6 on MedMCQA, significant for both seeds. On PubMedQA only seed 2 is significant.
3. direct and cot confidence was already around 0.94-0.98 and GRPO doesn't move it. So when ECE drops in those modes, it's mostly because accuracy went up a bit, not because calibration actually got better.

Only one seed: vote8 ECE gets significantly worse for seed 2 on all three datasets, but not for seed 1 on any of them.

All paired comparisons are in `eval_test/paired_vs_base.csv`, all metrics in `eval_test/summary.csv`, and the reliability diagrams in `eval_test/reliability_*.png`.

## Things that happened along the way

- lr 1e-4 seed 1: we stopped it at step 300 (`--stop_after_step 300`), evaluated checkpoint-300 on the validation set, then resumed from it and ran to 1250. We only kept the 2 most recent checkpoints (`save_total_limit 2`), so checkpoint-300 got deleted later and `grpo_lr1e-4_step300` can't be re-evaluated. Its results are only in `eval_val/`.
- lr 1e-4 seed 2: the Colab session dropped near the end, so we resumed from checkpoint-1200. `metrics.csv` kept the first 1200 rows and kept going from there.
- lr 5e-5 and 2e-4 stopped at 300 steps on purpose, so they never got a final adapter.

## Not in the repo

The LoRA adapters, checkpoints, full training logs, per-question predictions, and generated reasoning are too big to commit. Every table and figure here was computed from those per-question predictions with `eval/metrics.py`.
