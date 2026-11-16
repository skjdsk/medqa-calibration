# Base model baseline (Qwen2.5-3B-Instruct, no training)

Eval setup: direct answer (direct), probabilities from the logits on the option letters, normalized; prompt version `direct-v1`; 4-bit nf4 + double quant, compute dtype = bfloat16; seed 42; Colab T4.
The full setup is in each `preds_base_*_meta.json`. **Trained models have to be evaluated with the same setup for the results to be comparable.**

## Results

| Dataset | Split | n | Accuracy | Mean confidence | ECE | Brier | NLL | Confidence >= 0.9 and wrong | Accuracy when confidence >= 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| MedQA (4 options) | test | 1273 | 0.465 | 0.956 | 0.491 | 0.999 | 5.618 | 565 (44.4%) | 0.493 (1115 questions) |
| MedMCQA | validation | 4183 | 0.448 | 0.957 | 0.509 | 1.044 | 6.831 | 1919 (45.9%) | 0.471 (3627 questions) |
| PubMedQA | pqa_labeled (all 1000) | 1000 | 0.649 | 0.925 | 0.280 | 0.615 | 2.408 | 219 (21.9%) | 0.717 (775 questions) |

`mean_letter_mass` is >= 0.989 on all three datasets (about 1.0 on average), so the model really is picking a letter right away and the probabilities can be trusted.

## Takeaways

- **Badly overconfident** on all three datasets: mean confidence is about 0.93-0.96 while accuracy is only 0.45-0.65.
- On MedQA and MedMCQA the model gets about half the questions wrong but is very confident on almost all of them (most questions have confidence >= 0.9), and accuracy on those high-confidence questions is under 0.5.
- PubMedQA only has three options (yes/no/maybe) and the answers aren't balanced, so 0.649 accuracy doesn't by itself mean the model is "better" there. Keep that in mind when comparing against an always-guess-the-majority baseline.
- What we'd hope to see after training: lower ECE, Brier, NLL and share of "confidence >= 0.9 and wrong", with the reliability diagram bars moving toward the diagonal.

## Files

| File | What's in it |
|---|---|
| `summary.csv` / `summary.json` | every metric in the table above (plus ones not shown, like `mean_conf_wrong` and `min_letter_mass`) |
| `bins_base_*_direct.csv` | 15 equal-width confidence bins: count, mean confidence and accuracy per bin (used for the reliability diagrams) |
| `reliability_base_*_direct.png` | reliability diagrams |
| `preds_base_*_meta.json` | the full setup for each eval run (model, prompt version, quantization, seed, library versions, GPU) |

The per-question predictions (`preds_base_*_direct.csv`, about 717 KB for MedMCQA) aren't in the repo.
