# DPO post-training evaluation

Open `notebooks/dpo_evaluate.ipynb` from the repository. It runs locally in
`DL_Final` or in Colab. Execution flags default to `False`. The completed training
notebook retains its Colab outputs, but baseline/training flags are disabled.

## Synced experiment

The latest DPO preparation/training notebooks and small artifacts from
`base_medqa_4bit` and `dpo_full_3epochs_seed1_lr5e-6_beta0p1_r16_drop0p05`
were imported from the November 9 Colab download. Adapter weights, checkpoints,
optimizer states, and pilot results were not imported. Previous notebook versions
and the synchronization manifest are in the sibling backup directory under
`before_cot_temperature_sync`. The original download remains unchanged.

Validation accuracies at steps 208, 416, and 624 were 0.498, 0.498, and 0.496.
The existing accuracy/NLL selection rule chose step 208. This adapter choice is
fixed before any post-hoc test evaluation.

## CoT protocol

`eval.evaluate.score_cot` delegates to `eval.reasoning.score_cot` and shares
`score_letters` with direct evaluation. For four-option MedQA the CoT prompt
matches the frozen DPO training prompt, including its medical system instruction.
Direct prompting is unchanged. This comparison therefore changes both reasoning
instructions and the system prompt; it is not a pure isolation of reasoning.

1. Greedily generate up to 512 new tokens by default, with batched left padding.
2. Remove the first final-answer line and all subsequent text, so the generated
   final answer is not fed back into the answer scoring position.
3. Append `\nAnswer:` to the remaining reasoning and score all option letters
   using full-vocabulary logits. Merge single-token letter/space-letter variants.
4. Export normalized probabilities, stable log probabilities, raw generation,
   reasoning, parsed generated answer, format status, and truncation status.

`accuracy` is the forced-answer/rescored accuracy. `generated_accuracy` is based
on the parsed generated final answer; malformed generations count as incorrect.
All questions remain in both denominators. Reasoning can itself mention an
option, so this is conditional answer confidence, not independent verification
of the rationale. Missing final lines are rescored after the available reasoning;
inspect format/truncation rates before interpreting results. Context overflow
raises an error instead of silently dropping question content.

The existing GRPO `eval/cot_eval.py` vLLM implementation is unchanged. Its prompt,
parsing and top-k probability protocol differ; do not pool its outputs with this
new full-vocabulary protocol. Compare base and DPO with identical settings here.

From the repository root:

```powershell
python -m train.dpo.evaluate --source hf --adapter PATH_TO_BEST_ADAPTER --model_name step_000208 --modes direct cot --splits validation test --batch_size 2 --output_dir outputs/posthoc_dpo
```

Omit `--adapter` and use `--model_name base` for the original model. For local
files use `--source local`, optionally `--model`, `--local_train_file`, and
`--local_test_file`. `--n 8` limits each requested split for a smoke test.
`--cot_max_new_tokens` and `--cot_max_context_tokens` control generation limits.
The notebook can reference adapter weights directly in the retained download
when they are absent from the main repository; it never copies them.

The shared CLI also supports CoT, for example:

```powershell
python -m eval.evaluate --dataset medqa --medqa_subset val --mode cot --batch_size 2 --out_dir results/cot
```

## Temperature scaling

`eval/temperature.py` is CPU-only and needs no model download. It fits a positive
scalar T by minimizing validation NLL; the objective is convex in inverse
temperature. Search bounds are 0.05 to 100 and boundary optima are flagged.
Probabilities become `softmax(log(p) / T)`. Positive scalar T preserves the
predicted answer and accuracy. This is unrelated to generation temperature.

New prediction CSVs contain log probabilities to avoid underflow. Older Colab
CSVs are supported by taking logs of probabilities clipped at 1e-30; exact
information lost to earlier underflow cannot be recovered. The artifact records
which input representation was used.

Fit a separate temperature per checkpoint and evaluation mode. Inputs must use
one consistent `model_name` and `eval_mode`; use distinct names for distinct
checkpoints. IDs and modes are checked, but legacy CSVs do not cryptographically
identify the model weights, so names must be assigned correctly by the caller.
Known test IDs/non-validation split labels are rejected during fitting. The
caller must also ensure that arbitrary custom CSVs truly contain held-out
validation data. Applying a temperature to its fitting questions is rejected.

```powershell
python -m eval.temperature fit --calibration_csv calibration_fit.csv --calibration_split validation --output outputs/temperature.json
python -m eval.temperature apply --predictions_csv independent_test.csv --temperature_file outputs/temperature.json --output_dir outputs/calibrated_test
```

The notebook uses the same seeded 250/250 split of the fixed 500 validation IDs
for all model/mode combinations. It fits on 250 and reports a diagnostic holdout
on the remaining 250. Because the full validation set already selected the
checkpoint, this holdout is exploratory. Official test predictions, if available,
provide separate reporting without refitting T. Do not tune configurations on
these test results. There is no automatic transfer-domain refitting.

Outputs include fitted temperature/IDs, raw versus calibrated accuracy, NLL,
Brier, ECE, high-confidence error/coverage, and reliability-bin CSVs. Raw input
files remain unchanged. Nonempty output directories are rejected. Calibration
results supplement the original project's raw calibration comparison; they do
not replace it or feed back into training/checkpoint selection.

## Verification

Install `pytest` in the development environment, then run:

```powershell
python -m pytest -q eval/test_posthoc.py eval/test_metrics.py eval/test_cot_eval.py train/dpo/test_training.py
```

Tests cover NLL fitting, unchanged argmax, fit/evaluation overlap, protocol
mismatches, invalid probabilities, missing options, prompt parity, final-answer
removal, malformed/truncated generations, model-state restoration, and the
existing training/evaluation regression checks. No pretrained training is run.
