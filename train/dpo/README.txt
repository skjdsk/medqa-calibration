DPO training workflow
=====================

Scope
-----
This workflow trains a fresh LoRA adapter on the original Qwen2.5-3B-Instruct
model. It does not initialize from the project's SFT adapter. The frozen base
model, with adapters disabled, supplies the DPO reference probabilities.
The verified preference file is reused across epochs and training seeds.
No online generation, reward model training, or preference rebuilding occurs.
The shared SFT, GRPO, data split, and evaluation implementations are unchanged.

Environment
-----------
Use Python 3.11 or a compatible Colab Python runtime and a CUDA-enabled PyTorch.
Activate DL_Final locally, then install the DPO-specific dependencies:

    python -m pip install -r train/dpo/requirements.txt

The dependency file pins TRL 1.16.0, Transformers 4.57.6, and PEFT 0.18.0.
Do not install a second PyTorch build over Colab's working CUDA installation.
The entry point supports exactly one visible GPU. Set CUDA_VISIBLE_DEVICES
when a machine has multiple GPUs. No FlashAttention or vLLM is required.

Data and tokenizer checks (no model weights loaded)
--------------------------------------------------
Run all commands from the repository root:

    python -m train.dpo.train --source local --check_data

Local mode uses ../Qwen2.5-3B-Instruct and the original JSONL under
../MedQA-USMLE-4-options by default. Override --model and --local_train_file
when needed. HF mode loads the original model/tokenizer and MedQA train split
from Hugging Face; it still reads the generated preference pairs from disk:

    python -m train.dpo.train --source hf --check_data

The default preference file is:
outputs/medqa5000_L4_seed42_qpb2/selection_seed42/dpo.jsonl

Keep its build_manifest.json and retained_questions.jsonl sidecars, plus the
sampling_manifest.json one directory above. Archived raw responses and SFT
exports are not required. The validator checks source content/order hashes,
fixed train/validation separation, original question text, and final-answer
correctness in every chosen/rejected pair. It does not certify every clinical
reasoning statement. The default 2048-token budget includes prompt and answer.
An overlength example raises an error; no pairs or answer endings are dropped.

Pilot, full training, and resume
-------------------------------
Use a new output directory for a separate pilot:

    python -m train.dpo.train --source local --n_pairs 64 --n_val 32 --max_steps 10 --output_dir outputs/dpo_pilot_seed1

For a complete one-epoch run on all 3322 pairs and all 500 validation questions:

    python -m train.dpo.train --source local --seed 1 --output_dir outputs/dpo_train_seed1

For Colab, set --source hf and put --output_dir on mounted Google Drive.
Use --quantization none for unquantized BF16/FP16 LoRA; the default is 4-bit
NF4 QLoRA with double quantization. Compute precision is selected from the GPU
or explicitly controlled by --compute_dtype. Trainable adapter parameters are
stored in FP32; gradients use the configured mixed precision.

Defaults: rank 16, alpha 32, all-linear targets, learning rate 5e-6, beta 0.1,
one epoch, one pair per microbatch, 16 accumulation steps, cosine learning-rate
schedule, 10 percent warmup, and gradient checkpointing. These are starting
settings, not tuned results. Each pair contains two answer sequences.
LoRA dropout is active during training: --lora_dropout defaults to 0.05.
Use --lora_dropout 0 to disable it. Base-model dropout remains disabled.
TRL disable_dropout=False preserves adapter dropout; evaluation disables it.
Old checkpoints created before this behavior change cannot resume into this implementation.
Coordinate quantization, rank, targets, and effective dropout across methods
before interpreting comparisons as controlled experiments.

Reference log probabilities are precomputed once per trainer construction.
This avoids reference forward passes on every training step. Recreating the
trainer on resume may repeat preprocessing; the reference still stays fixed.
Use --no-precompute_ref_log_probs to compute them online with adapters disabled.

Checkpoint files include adapter, optimizer, scheduler, RNG, trainer state, and
the run signature. Checkpoints are saved every 50 updates and at epoch ends.
Two recent checkpoints are retained by default; best_adapter is separate.

    python -m train.dpo.train --source local --seed 1 --output_dir outputs/dpo_train_seed1 --resume

Resume requires the same data and training configuration. A missing or
incomplete checkpoint is an error, never an implicit restart. best_adapter and
last_adapter are inference artifacts, not optimizer-resume checkpoints.

To measure part of the real run without changing its learning-rate schedule:

    python -m train.dpo.train --source local --stop_after_step 20 --output_dir outputs/dpo_train_seed1

Then resume with the same arguments except --stop_after_step. A separate pilot
using --max_steps changes the schedule and should not be resumed as a full run.

Validation and checkpoint selection
-----------------------------------
The original 500 held-out MedQA questions are evaluated before training and
after every epoch, including the final partial epoch of a stopped pilot.
Validation uses the shared direct-answer option-probability scoring and metric
functions: accuracy, 15-bin ECE, multiclass Brier, NLL, high-confidence error rate
(confidence >= 0.9 divided by all questions), wrong-answer mean confidence,
high-confidence accuracy/count, and letter probability mass. Prediction CSVs
and reliability-bin CSVs are saved. No validation preferences are manufactured.

best_adapter is selected by highest validation accuracy, with lower NLL as the
tie breaker. The original base-model result is recorded separately. The script
does not select using test or transfer data and does not automatically stop on
an unfavorable metric. Training loss/reward accuracy are not medical accuracy.

Artifacts under --output_dir
----------------------------
run_manifest.json               Data/configuration hashes, versions, GPU, and selection policy.
effective_training_config.json  Resolved TRL training configuration.
training_metrics.jsonl          DPO loss, reward margins, optimizer logs, and steps.
step_times.jsonl                Per-update timing and peak CUDA allocation/reservation.
baseline_validation.json        Held-out metrics before training.
validation/                    Per-checkpoint predictions, metrics, and reliability bins.
best_adapter/                  Adapter selected on held-out accuracy/NLL, plus tokenizer.
best_validation.json           Selected adapter's validation step and scores.
last_adapter/                  Last adapter and tokenizer, regardless of selection.
checkpoints/checkpoint-*/       Full resumable training state.
training_summary.json          Session timing, memory, progress, and per-epoch estimate.

The epoch-time estimate uses optimizer-update times after three warmup updates;
it excludes validation, checkpoint I/O, loading, and reference preprocessing.
Peak CUDA allocation is not total GPU memory reported by nvidia-smi.
For final experiments, run a second seed in a separate directory with the same
frozen pairs and selected hyperparameters.

Final evaluation (only after selecting the configuration)
--------------------------------------------------------
Reuse the existing evaluator; match the training quantization mode:

    python -m eval.evaluate --dataset medqa medmcqa pubmedqa --base_model Qwen/Qwen2.5-3B-Instruct --adapter outputs/dpo_train_seed1/best_adapter --model_name dpo_seed1 --quantization 4bit --out_dir results/dpo_seed1
    python -m eval.metrics --pred_csv results/dpo_seed1/preds_*.csv --out_dir results/dpo_seed1/metrics

The shared evaluator downloads its evaluation datasets. Its existing --mode cot
option can be used for the secondary reasoning-based evaluation. It is not part
of per-epoch checkpoint selection. The shared evaluator retains its original
interface and messages; new DPO-specific code and notebook text are English.

Notebook
--------
notebooks/dpo_train.ipynb supports local and Colab configuration, optional
dependency installation, data-only checks, guarded training, and result plots.
Training is disabled by default until RUN_TRAINING is explicitly set to True.
The notebook stores no executed training results in the repository initially.

Tests
-----
    python -m unittest train.dpo.test_training train.dpo.test_prepare -v

The optional integration test uses a tiny random CPU model and synthetic data.
It checks adapter updates, frozen base/reference behavior, and checkpoint-resume
equivalence. It downloads no model weights and does not train Qwen.

One-time original-model baseline
--------------------------------
The first optional stage in notebooks/dpo_train.ipynb runs train.dpo.baseline.
Set RUN_BASELINE=True for the first experiment, then leave it False to display
saved results without loading a model. It evaluates all 500 fixed validation
questions and the full official MedQA test split using direct-answer option
probabilities. It saves summary.json, manifest.json, predictions, metrics, and
reliability tables under outputs/base_medqa_4bit. Test metrics are descriptive
only; do not use them to tune parameters or select adapters.

Example from the repository root:
python -m train.dpo.baseline --source hf --quantization 4bit --output_dir outputs/base_medqa_4bit

Local mode also accepts --local_test_file. Existing baseline directories are
protected from overwrite. For different model revisions or quantization, use
a new baseline directory. Keep precision and quantization consistent with the
trained-model evaluation. The notebook passes --no-baseline_validation to
training to avoid repeating the initial baseline. The command-line trainer
still runs its per-run initial validation by default. Epoch-end adapter
validation is always retained. No GPU evaluation is performed by creating
or editing this notebook.
