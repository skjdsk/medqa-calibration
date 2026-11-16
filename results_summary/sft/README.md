# SFT Results — Qwen2.5-3B-Instruct

## Overview

Supervised Fine-Tuning (SFT) on MedQA using correct reasoning
responses selected from a shared candidate-generation pipeline.

This report covers direct-answer calibration evaluation.
The same evaluation protocol was used for the BF16 base model
and both SFT seeds.

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct |
| Training method | BF16 LoRA SFT |
| Selected training examples | 3,322 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA targets | all-linear |
| Max sequence length | 2048 |
| Epochs | 1 |
| Effective batch size | 4 |
| Selected learning rate | 1e-4 |
| Final seeds | 42, 123 |
| Hardware | NVIDIA L4 |
| Loss | Completion-only cross entropy |

### Hyperparameter Tuning

Two configurations were evaluated on the fixed 500-question
MedQA validation subset:

- Run A: learning rate 5e-5, seed 42
- Run B: learning rate 1e-4, seed 42

Run B was retained for the calibration-focused final analysis
because it had lower validation NLL and Brier score.

The selection criterion was confirmed after inspecting validation
results and should be considered exploratory rather than preregistered.

The seed-42 tuning run was reused as one final replicate;
the second run was independently trained with seed 123.

## Evaluation Protocol

- Mode: Direct answer (letter probabilities)
- Precision: BF16, no quantization
- Model comparison: Base vs. SFT seed 42 vs. SFT seed 123
- Datasets: MedQA test, MedMCQA validation, PubMedQA labeled
- Metrics: Accuracy, ECE (15 bins), Brier, NLL,
  high-confidence wrong rate
- High-confidence threshold: 0.9

The baseline reported here was reevaluated in BF16.
It must not be confused with the repository's separate
4-bit baseline in results_summary/base/.

## Final Results

SFT results below are averaged across the two final seeds.
Change = SFT mean minus BF16 base.

| Dataset | Metric | BF16 Base | SFT Mean | Change |
|---|---|---:|---:|---:|
| medqa | accuracy | 0.4910 | 0.4906 | -0.0004 |
| medqa | ece | 0.4688 | 0.4609 | -0.0078 |
| medqa | brier | 0.9649 | 0.9522 | -0.0127 |
| medqa | nll | 5.3796 | 4.7795 | -0.6001 |
| medqa | hc_wrong_rate | 0.4305 | 0.4152 | -0.0153 |
| medmcqa | accuracy | 0.4748 | 0.4745 | -0.0002 |
| medmcqa | ece | 0.4834 | 0.4766 | -0.0069 |
| medmcqa | brier | 0.9924 | 0.9822 | -0.0102 |
| medmcqa | nll | 6.3982 | 5.7713 | -0.6269 |
| medmcqa | hc_wrong_rate | 0.4341 | 0.4212 | -0.0129 |
| pubmedqa | accuracy | 0.6410 | 0.6495 | +0.0085 |
| pubmedqa | ece | 0.2982 | 0.2745 | -0.0237 |
| pubmedqa | brier | 0.6380 | 0.5982 | -0.0398 |
| pubmedqa | nll | 2.4686 | 2.0528 | -0.4157 |
| pubmedqa | hc_wrong_rate | 0.2430 | 0.2020 | -0.0410 |

## Findings

1. Accuracy remained approximately unchanged on MedQA and
   MedMCQA. PubMedQA accuracy increased slightly.
2. NLL and Brier improved across all three datasets
   for both final SFT seeds.
3. ECE and the share of high-confidence wrong predictions
   decreased across all three datasets.
4. Substantial overconfidence remains after SFT.

These results describe observed differences, not claims
of statistical significance.

## Limitations

- Only two final training seeds were evaluated.
- Seed 42 also participated in learning-rate tuning.
- The evaluation uses normalized letter probabilities.
- SFT trained on reasoning traces, while the reported
  primary evaluation uses direct-answer scoring.
- Paired bootstrap uncertainty estimates and additional
  qualitative error analysis remain to be completed.

## Files

- `summary.csv`, `summary.json`: complete metric summaries
- `bins_*.csv`: 15-bin calibration tables
- `reliability_*.png`: reliability diagrams
- `preds_*_meta.json`: evaluation configuration metadata

Full per-question predictions, LoRA adapters, and training
checkpoints are retained in Google Drive rather than GitHub.

Drive root:
`MyDrive/medqa-calibration/sft/`

Relevant folders:
- `tuning_lr5e5/`
- `tuning_lr1e4/`
- `final_seed123/`
- `final_evaluation/`

## Reproduction

Evaluation uses the shared scripts:

```bash
python -m eval.evaluate --dataset medqa medmcqa pubmedqa \
    --adapter /path/to/sft_adapter \
    --model_name sft_seed42 \
    --quantization none \
    --out_dir results/sft

python -m eval.metrics \
    --pred_csv "results/sft/preds_*.csv" \
    --out_dir results/sft/metrics
```

The repository notebook documents the SFT training workflow.
