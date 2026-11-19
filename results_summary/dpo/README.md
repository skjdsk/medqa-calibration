# DPO results

A copy of the DPO run outputs, put here so all methods' results sit in `results_summary/`. The originals are still in `outputs/` (the DPO notebooks read from there). Training and eval code is in `train/dpo/`, and the notebooks are in `notebooks/dpo/`.

## `training_seed1/`

One DPO run: bf16 LoRA (r 16, dropout 0.05), lr 1e-5, beta 0.1, 3 epochs, seed 1, batch 4 x grad accum 4. The model was checked on the 500 MedQA validation questions after each epoch (steps 208, 416 and 624).

| File | What's in it |
|---|---|
| `effective_training_config.json`, `run_manifest.json` | full training config and run info |
| `training_metrics.jsonl`, `step_times.jsonl`, `training_summary.json` | per-step training logs and an overall summary |
| `best_validation.json` | the checkpoint with the best validation result and its metrics |
| `validation/step_*_metrics.json` | validation metrics at each epoch |
| `validation/step_*_predictions.csv` | per-question validation predictions |
| `validation/step_*_reliability.csv` | calibration bins for the reliability diagrams |
| `loss_curves.png` | loss, preference accuracy, reward margin and gradient norm over the 3 epochs |

## `temperature_scaling/`

Temperature scaling on top of each checkpoint above (direct answer). The 500 validation questions are split in half: 250 to fit the temperature and 250 held out to check it.

| File | What's in it |
|---|---|
| `comparison.csv` | before vs after temperature scaling on the held-out half, for each checkpoint |
| `partition.json` | which questions went into the fit half and the held-out half |
| `step_*/direct/temperature.json` | the fitted temperature and the fit details |
| `step_*/direct/holdout/metrics.*` | held-out metrics before (raw) and after (temperature) |
| `step_*/direct/*predictions.csv` | per-question predictions used for fitting and checking |
| `step_*/direct/holdout/*_reliability.csv` | calibration bins before and after |
