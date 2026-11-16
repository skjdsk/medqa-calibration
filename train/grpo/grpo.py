"""
train/grpo/grpo.py -- GRPO training script (used for both the pilot and the full runs)

[What this file does]
    1. Prints the environment (GPU, TRL / transformers / peft / bitsandbytes versions) and picks bf16 or fp16
    2. Takes N questions from the MedQA **training set** (default 64) and builds Qwen chat-template prompts
    3. Loads Qwen2.5-3B-Instruct in 4-bit (nf4) with a LoRA adapter (QLoRA); with --quantization none it's bf16 + LoRA
    4. Before training: samples 2 answers and prints them with their reward
    5. Trains with TRL's GRPOTrainer for max_steps steps (default 30), logging every step:
                time, peak memory, mean reward / std, share of all-same groups, answer length, truncation rate, format rate, KL
          -> written to metrics.csv (one row per step, written as it goes, so a crash doesn't lose everything)
    6. After training: generates answers for the same 2 questions again for comparison
    7. Saves the LoRA adapter, metrics.csv, pilot_report.json and curves.png; with --drive_dir it also copies the folder to Google Drive

[What one GRPO step does] (helps explain where the time and memory go)
    Each optimizer step:
      a) take P questions (P = answers per step / num_generations, default 16/4 = 4)
      b) generate num_generations answers per question (default 4) -> 16 answers total        <- usually the slowest part (token-by-token generation)
      c) score each answer 0/1 with rewards.py
      d) compare within each group to get advantage = (reward - group mean) / group std (all-right / all-wrong groups get 0)
      e) forward + backward in gradient_accumulation_steps chunks (default 4 chunks of 4 answers), then one LoRA update
            if beta > 0, there's one more forward pass with LoRA turned off ("the original model") to compute the KL penalty

[What affects memory / speed]
    memory: per_device_train_batch_size (answers per forward/backward, the big one), max_completion_length (sequence length),
                gradient_checkpointing (saves memory but ~20-30% slower), lora_r (barely matters), use_vllm (takes extra memory)
    speed: max_completion_length (generation runs token by token, so the longest answer sets the time),
                answers per step = per_device_train_batch_size x gradient_accumulation_steps, num_generations,
                beta > 0 (one extra reference forward pass, about +10-20%), use_vllm (generation can be several times faster on an A100)

[Running] (from the repo root)
    python -m train.grpo.grpo --n_questions 64 --max_steps 30 --output_dir outputs/grpo_pilot \\
        --drive_dir /content/drive/MyDrive/medqa-calibration/grpo_pilot
    Full run (steps come from epochs, checkpoint every 100 steps; if Colab disconnects, rerun the same command with --resume):
    python -m train.grpo.grpo --n_questions 0 --seed 1 --num_train_epochs 1 --num_generations 8 \\
        --gradient_accumulation_steps 8 --max_completion_length 1024 --quantization none --use_vllm \\
        --output_dir <output folder>/<run name> --save_steps 100 --resume
    To run just the first 300 steps (same LR schedule; drop --stop_after_step and rerun to finish): add --stop_after_step 300
    All options: python -m train.grpo.grpo --help
"""

from __future__ import annotations

import os

# cuts down on "out of memory even though there's room" from fragmentation; has to be set before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import dataclasses
import json
import math
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# make "from train.xxx import ..." work with both python -m train.grpo.grpo and python train/grpo/grpo.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from data.medqa_split import N_TRAIN  # noqa: E402
from train.grpo.estimate import FORMULA, estimate_gpu_hours  # noqa: E402
from train.grpo.grpo_data import build_grpo_dataset, print_samples, prompt_token_stats  # noqa: E402
from train.grpo.rewards import (  # noqa: E402
    RewardTracker,
    extract_answer,
    make_correctness_reward,
    score_completion,
)

FULL_TRAIN_SIZE = N_TRAIN  # number of training questions (the 5000 in data/medqa_split_seed42.json), used for the full-run estimate in the report


# ---------------------------------------------------------------------------
# Command-line options (everything is configurable)
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO pilot on MedQA (QLoRA + TRL GRPOTrainer)")

    # data
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument(
        "--n_questions",
        type=int,
        default=64,
        help="take the first N training questions (from the 5000 in data/medqa_split_seed42.json); 0 = all 5000",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="training seed (LoRA init, sampling, data order). Which questions get used comes from the split file, not this seed",
    )

    # quantization
    p.add_argument(
        "--quantization",
        choices=["4bit", "none"],
        default="4bit",
        help="4bit (default) = QLoRA, nf4 base; none = no quantization, bf16 (A100) / fp16 (T4) base + LoRA. "
        "With QLoRA + --use_vllm, merging LoRA into the 4-bit weights for vLLM every step piles up rounding error; "
        "in an A100 pilot importance_sampling_ratio dropped to near 0 and gradients vanished. Use none with vLLM",
    )

    # LoRA (SFT / DPO use the same settings)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        default="all-linear",
        help='"all-linear" or comma-separated module names, e.g. "q_proj,k_proj,v_proj,o_proj"',
    )
    p.add_argument(
        "--lora_param_dtype",
        choices=["fp32", "trl_default"],
        default="fp32",
        help="storage precision for the LoRA params. TRL 1.16 forces QLoRA's LoRA params to bf16; "
        "fp32 (default) casts them back: required for fp16 mixed precision on a T4, and on an A100 it keeps small-LR updates from being rounded away in bf16.",
    )

    # core GRPO settings
    p.add_argument("--num_generations", type=int, default=4, help="answers per question (group size G)")
    p.add_argument("--max_completion_length", type=int, default=512, help="max tokens per answer")
    p.add_argument("--max_steps", type=int, default=30, help="number of training steps; ignored when --num_train_epochs > 0")
    p.add_argument(
        "--num_train_epochs",
        type=float,
        default=0,
        help="> 0 trains by epoch: total steps = ceil(questions x epochs / questions per step), overrides --max_steps (used for full runs)",
    )
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="KL coefficient. > 0 also logs KL (the model with LoRA off is the reference, so no extra memory, but one more forward pass per step). "
        "Note TRL 1.16 defaults to 0.0.",
    )
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument(
        "--loss_type",
        default=None,
        help="empty = TRL default ('dapo' in 1.16). Also 'grpo' (original paper, normalized by sequence length), 'dr_grpo', etc.",
    )
    p.add_argument(
        "--scale_rewards",
        default=None,
        help="empty = TRL default ('group': advantage divided by the group std). Also 'batch' / 'none'",
    )

    # Batch settings. Note: in TRL GRPO, per_device_train_batch_size counts answers
    #   answers per step = per_device_train_batch_size x gradient_accumulation_steps x number of GPUs
    #   has to be divisible by num_generations; questions per step = answers per step / num_generations
    #   default 4 x 4 = 16 answers = 4 questions x 4 answers
    p.add_argument("--per_device_train_batch_size", type=int, default=4, help="answers per forward/backward (the main memory knob)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--no_gradient_checkpointing", action="store_true", help="turn off gradient checkpointing (faster but uses more memory)")

    # vLLM (optional, faster generation on an A100)
    p.add_argument("--use_vllm", action="store_true", help="generate with vLLM in colocate mode (needs pip install 'trl[vllm]')")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    p.add_argument("--vllm_max_model_length", type=int, default=2048)

    # output
    p.add_argument("--output_dir", default="outputs/grpo_pilot")
    p.add_argument("--drive_dir", default="", help="Google Drive folder to copy results to; leave empty to skip")
    p.add_argument(
        "--run_name",
        default="",
        help="subfolder name on Drive; empty = output_dir folder name + GPU + time (e.g. grpo_Tesla_20261025_213538)",
    )

    # Checkpoints / resuming (for full runs; the pilot doesn't save checkpoints by default)
    p.add_argument(
        "--save_steps",
        type=int,
        default=0,
        help="save a checkpoint every N steps (with optimizer state, for resuming); 0 = don't save (pilot default)",
    )
    p.add_argument("--save_total_limit", type=int, default=2, help="max checkpoints to keep; older ones get deleted")
    p.add_argument(
        "--checkpoint_dir",
        default="",
        help="where checkpoints go; empty = <output_dir>/checkpoints. "
        "On Colab, put it on Google Drive so a disconnect doesn't lose it",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest checkpoint in checkpoint_dir; starts from scratch if there isn't one",
    )
    p.add_argument(
        "--stop_after_step",
        type=int,
        default=0,
        help="> 0 saves a checkpoint and stops at this step (the LR schedule still assumes the full run); "
        "use it to look at the first N steps, then drop it and add --resume to finish. 0 = don't stop early",
    )
    p.add_argument("--num_samples", type=int, default=2, help="how many sample answers to show before and after training")
    p.add_argument("--warmup_skip", type=int, default=2, help="steps to skip (warmup) when averaging time per step")
    p.add_argument("--report_last_k", type=int, default=5, help="how many final steps to average for 'last steps' in the report")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------
def detect_environment() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. In Colab: Runtime -> Change runtime type -> GPU.")

    import bitsandbytes
    import peft
    import transformers
    import trl

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # go by compute capability: >= 8.0 (A100/L4/H100, Ampere and later) has native bf16; the T4 is 7.5, so fp16 only.
    # (not torch.cuda.is_bf16_supported(): newer PyTorch can return True on a T4 because of software emulation.)
    use_bf16 = major >= 8
    env = {
        "gpu_name": name,
        "gpu_compute_capability": f"{major}.{minor}",
        "gpu_total_memory_gb": round(total_gb, 2),
        "precision": "bf16" if use_bf16 else "fp16",
        "use_bf16": use_bf16,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "trl": trl.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
    }
    print("=" * 80)
    print(f"TRL version: {trl.__version__}   (this code uses TRL 1.16 parameter names)")
    print(
        f"GPU: {name} (compute {major}.{minor}, {total_gb:.1f} GB) -> precision {env['precision']}\n"
        f"torch {env['torch']} | CUDA {env['cuda']} | transformers {env['transformers']} | "
        f"peft {env['peft']} | bitsandbytes {env['bitsandbytes']} | python {env['python']}"
    )
    print("=" * 80)
    return env


# ---------------------------------------------------------------------------
# GRPOConfig: built with the current TRL version's parameter names; unknown ones get dropped with a warning
# ---------------------------------------------------------------------------
def check_batch_settings(args) -> dict[str, int]:
    n_gpus = int(os.environ.get("WORLD_SIZE", "1"))  # single GPU on Colab = 1
    completions_per_gpu = args.per_device_train_batch_size * args.gradient_accumulation_steps
    completions_per_step = completions_per_gpu * n_gpus
    if args.num_generations < 2:
        raise SystemExit("num_generations has to be at least 2 (GRPO compares answers within a group).")
    if completions_per_step % args.num_generations != 0:
        raise SystemExit(
            f"Bad batch settings: answers per step = per_device_train_batch_size({args.per_device_train_batch_size}) x "
            f"gradient_accumulation_steps({args.gradient_accumulation_steps}) x GPUs({n_gpus}) = "
            f"{completions_per_step}, which has to be divisible by num_generations({args.num_generations}).\n"
            f"Tip: make per_device_train_batch_size x gradient_accumulation_steps a multiple of num_generations, "
            f"e.g. 4x4 or 2x8 for num_generations=4, and 4x8 or 8x4 for num_generations=8."
        )
    info = {
        "num_gpus": n_gpus,
        "completions_per_step_per_gpu": completions_per_gpu,
        "completions_per_step": completions_per_step,
        "prompts_per_step": completions_per_step // args.num_generations,
    }
    print(
        f"[batch] {info['completions_per_step']} answers per step = {info['prompts_per_step']} questions x "
        f"{args.num_generations} answers; {args.per_device_train_batch_size} per forward/backward, "
        f"one update every {args.gradient_accumulation_steps} of those"
    )
    return info


def make_grpo_config(args, env):
    from trl import GRPOConfig

    dtype_str = "bfloat16" if env["use_bf16"] else "float16"
    kwargs: dict[str, Any] = dict(
        output_dir=args.checkpoint_dir,  # Trainer saves checkpoint-<step>/ here
        # ---- training steps / learning rate ----
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        # ---- batch (the main memory and speed knobs) ----
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # TRL 1.16: generation_batch_size defaults to per_device x GPUs x steps_per_generation,
        #           steps_per_generation defaults to gradient_accumulation_steps
        #           -> generation happens once per optimizer step, all 16 answers at once. Left at the default here.
        # ---- GRPO ----
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,  # no top-k cutoff (TRL default). Qwen's own generation_config uses top_k=20, so we override it here
        repetition_penalty=1.0,
        beta=args.beta,
        num_iterations=1,  # each batch of generated data is used for one update (standard GRPO)
        # ---- precision / memory ----
        bf16=env["use_bf16"],
        fp16=not env["use_bf16"],
        gradient_checkpointing=not args.no_gradient_checkpointing,
        # TRL loads the model with these (since we hand GRPOTrainer the model name as a string)
        model_init_kwargs={"dtype": dtype_str},
        # ---- logging ----
        logging_steps=1,  # log every step so the callback gets per-step KL etc.
        report_to="none",
        # pilot (save_steps=0) saves no intermediate checkpoints, just the adapter at the end by hand;
        # full runs save every save_steps steps (with optimizer state) and keep only the latest save_total_limit
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=args.save_total_limit,
        log_completions=False,
        remove_unused_columns=False,  # keep the "answer" column, the reward function needs it
        # ---- vLLM (optional) ----
        use_vllm=args.use_vllm,
    )
    if args.use_vllm:
        kwargs.update(
            vllm_mode="colocate",  # shares the training GPU, no separate server needed
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_length=args.vllm_max_model_length,
        )
    if args.loss_type:
        kwargs["loss_type"] = args.loss_type
    if args.scale_rewards:
        kwargs["scale_rewards"] = args.scale_rewards

    # version compat: only pass parameters this TRL version knows, drop the rest with a warning (the TRL API changes fast)
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in kwargs if k not in valid)
    if dropped:
        print(f"[warning] GRPOConfig in this TRL version doesn't know these, ignored: {dropped}. Check the new names.")
    config = GRPOConfig(**{k: v for k, v in kwargs.items() if k in valid})
    return config, dropped


# ---------------------------------------------------------------------------
# Per-step metric logging: TrainerCallback
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "step",
    "step_time_s",  # time for this step (generate + score + forward/backward + update)
    "peak_mem_allocated_gb",  # torch.cuda.max_memory_allocated (peak within this step)
    "peak_mem_reserved_gb",  # peak memory PyTorch actually reserved (closer to what nvidia-smi shows)
    "reward_mean",
    "reward_std",
    "frac_groups_all_same",  # share of groups where every reward is the same (all right or all wrong) -> advantage 0
    "frac_groups_all_correct",
    "frac_groups_all_wrong",
    "format_rate",  # share where we can pull out "Answer: X"
    "mean_completion_tokens",
    "max_completion_tokens",
    "truncated_frac",  # share that hit max_completion_length without finishing
    "n_completions",
    "n_groups",
    "kl",  # KL as logged by TRL (only when beta > 0)
    "loss",
    "grad_norm",
    "entropy",
    "learning_rate",
    # TRL's own versions of similar metrics (to cross-check ours)
    "trl_reward",
    "trl_reward_std",
    "trl_frac_reward_zero_std",
    "trl_mean_length",
    "trl_clipped_ratio",
]

# TRL log key -> CSV column (TRL 1.16 keys; if a newer version renames one, that column is just empty, training is fine)
TRL_LOG_KEYS = {
    "kl": "kl",
    "loss": "loss",
    "grad_norm": "grad_norm",
    "entropy": "entropy",
    "learning_rate": "learning_rate",
    "reward": "trl_reward",
    "reward_std": "trl_reward_std",
    "frac_reward_zero_std": "trl_frac_reward_zero_std",
    "completions/mean_length": "trl_mean_length",
    "completions/clipped_ratio": "trl_clipped_ratio",
}


def _read_metrics_csv(path: Path) -> list[dict[str, Any]]:
    """Read metrics.csv back in (for resuming): numeric columns back to int/float, empty cells to None."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: dict[str, Any] = {}
            for k, v in raw.items():
                if v in ("", None):
                    row[k] = None
                    continue
                try:
                    row[k] = int(v) if k in ("step", "n_completions", "n_groups") else float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def _make_callback_class():
    from transformers import TrainerCallback

    class PilotMetricsCallback(TrainerCallback):
        """Each optimizer step: reset peak memory and start the timer at the beginning; summarize the reward-function metrics at the end;
        merge in TRL's KL etc. on on_log and write one CSV row."""

        def __init__(self, tracker: RewardTracker, csv_path: Path, max_steps: int, resume: bool = False):
            self.tracker = tracker
            self.csv_path = Path(csv_path)
            self.max_steps = max_steps
            self.resume = resume
            self.rows: list[dict[str, Any]] = []
            self._pending: dict[int, dict[str, Any]] = {}
            self._t0 = None
            self._file = None
            self._writer = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            # resuming: keep the rows before the checkpoint (later steps get rerun, so drop them), then keep appending
            if self.resume and self.csv_path.exists() and state.global_step > 0:
                self.rows = [r for r in _read_metrics_csv(self.csv_path) if r["step"] <= state.global_step]
                print(f"[resume] continuing from step {state.global_step}; keeping the first {len(self.rows)} rows of metrics.csv")
            self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS, extrasaction="ignore")
            self._writer.writeheader()
            self._writer.writerows(self.rows)
            self._file.flush()
            self.tracker.pop_step_stats()  # clear anything left over from before training started

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._t0 = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            dt = time.perf_counter() - self._t0 if self._t0 is not None else float("nan")
            row: dict[str, Any] = {
                "step": state.global_step,
                "step_time_s": round(dt, 3),
                "peak_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "peak_mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            }
            row.update(self.tracker.pop_step_stats())
            self._pending[state.global_step] = row

        def on_log(self, args, state, control, logs=None, **kwargs):
            row = self._pending.pop(state.global_step, None)
            if row is None:
                return
            for src, dst in TRL_LOG_KEYS.items():
                if logs and src in logs and logs[src] is not None:
                    row[dst] = logs[src]
            self._write(row)

        def on_train_end(self, args, state, control, **kwargs):
            for step in sorted(self._pending):
                self._write(self._pending[step])
            self._pending.clear()
            if self._file:
                self._file.close()

        def _write(self, row):
            self.rows.append(row)
            self._writer.writerow(row)
            self._file.flush()

            def f(key, fmt):
                v = row.get(key)
                return format(v, fmt) if isinstance(v, (int, float)) and not math.isnan(v) else "-"

            print(
                f"[step {row['step']}/{self.max_steps}] {f('step_time_s', '.1f')}s | "
                f"mem {f('peak_mem_allocated_gb', '.2f')}GB | reward {f('reward_mean', '.3f')}"
                f"±{f('reward_std', '.3f')} | all-same groups {f('frac_groups_all_same', '.2f')} | "
                f"format {f('format_rate', '.2f')} | length {f('mean_completion_tokens', '.0f')} | "
                f"truncated {f('truncated_frac', '.2f')} | kl {f('kl', '.4f')}",
                flush=True,
            )

    class StopAfterStepCallback(TrainerCallback):
        """Save a checkpoint and stop at step stop_step. max_steps stays the same, so the LR schedule matches the full run exactly."""

        def __init__(self, stop_step: int):
            self.stop_step = stop_step

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= self.stop_step:
                control.should_save = True
                control.should_training_stop = True
            return control

    return PilotMetricsCallback, StopAfterStepCallback


# ---------------------------------------------------------------------------
# Sample answers before and after training
# ---------------------------------------------------------------------------
@torch.no_grad()
def show_samples(model, tokenizer, examples, args, title: str) -> list[dict[str, Any]]:
    """Generate 1 answer per question (same sampling settings as training) and print the answer and reward."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    prompts = [ex["prompt"] for ex in examples]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    t0 = time.perf_counter()
    out = model.generate(
        **enc,
        max_new_tokens=args.max_completion_length,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    dt = time.perf_counter() - t0
    gen = out[:, enc["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

    results = []
    bar = "#" * 80
    print(f"\n{bar}\n{title} ({len(examples)} questions, generated in {dt:.1f}s)\n{bar}")
    for i, (ex, text, ids) in enumerate(zip(examples, texts, gen)):
        n_tokens = int((ids != tokenizer.pad_token_id).sum().item())
        pred = extract_answer(text)
        reward = score_completion(text, ex["answer"])
        results.append(
            {"answer": ex["answer"], "pred": pred, "reward": reward, "num_tokens": n_tokens, "completion": text}
        )
        print(f"--- sample {i + 1} | gold {ex['answer']} | extracted {pred} | reward = {reward} | {n_tokens} tokens ---")
        print(text.strip())
        print()
    if was_training:
        model.train()
    return results


# ---------------------------------------------------------------------------
# Report / plots
# ---------------------------------------------------------------------------
def _nanmean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else None


def build_report(args, env, batch_info, trainer_args, rows, dropped, samples, train_seconds, prompt_stats):
    times = [r["step_time_s"] for r in rows]
    skip = min(args.warmup_skip, max(len(times) - 1, 0))
    avg_excl = _nanmean(times[skip:])
    k = min(args.report_last_k, len(rows))
    col = lambda key: [r.get(key) for r in rows]  # noqa: E731

    losses = [r.get("loss") for r in rows if r.get("loss") is not None]
    grad_norms = [r.get("grad_norm") for r in rows if r.get("grad_norm") is not None]

    effective = {}
    for key in [
        "num_generations", "max_completion_length", "per_device_train_batch_size",
        "gradient_accumulation_steps", "generation_batch_size", "steps_per_generation",
        "learning_rate", "lr_scheduler_type", "beta", "temperature", "top_p", "top_k",
        "loss_type", "scale_rewards", "num_iterations", "epsilon", "bf16", "fp16",
        "gradient_checkpointing", "use_vllm", "max_steps", "seed",
    ]:
        if hasattr(trainer_args, key):
            v = getattr(trainer_args, key)
            effective[key] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gpu_name": env["gpu_name"],
        "environment": env,
        "trl_version": env["trl"],
        "config": vars(args),
        "effective_grpo_config": effective,  # the values TRL actually used (defaults included)
        "dropped_config_keys": dropped,
        "derived": {
            **batch_info,
            "pilot_passes_over_subset": args.max_steps * batch_info["prompts_per_step"] / args.n_questions,
            "prompt_tokens": prompt_stats,
        },
        "timing": {
            "num_steps_logged": len(rows),
            "warmup_steps_skipped": skip,
            "avg_step_time_s_excl_warmup": avg_excl,
            "avg_step_time_s_all": _nanmean(times),
            "first_step_time_s": times[0] if times else None,
            "max_step_time_s": max(times) if times else None,
            "total_train_time_s": train_seconds,
        },
        "memory": {
            "peak_mem_allocated_gb_max_over_steps": max(col("peak_mem_allocated_gb"), default=None),
            "peak_mem_reserved_gb_max_over_steps": max(col("peak_mem_reserved_gb"), default=None),
            "gpu_total_memory_gb": env["gpu_total_memory_gb"],
        },
        "reward": {
            "first_step": rows[0].get("reward_mean") if rows else None,
            f"last_{k}_steps_mean": _nanmean(col("reward_mean")[-k:]) if rows else None,
            "all_steps_mean": _nanmean(col("reward_mean")),
        },
        "format_rate": {
            "first_step": rows[0].get("format_rate") if rows else None,
            f"last_{k}_steps_mean": _nanmean(col("format_rate")[-k:]) if rows else None,
        },
        "truncated_frac_mean": _nanmean(col("truncated_frac")),
        "frac_groups_all_same_mean": _nanmean(col("frac_groups_all_same")),
        "frac_groups_all_same_last_k_mean": _nanmean(col("frac_groups_all_same")[-k:]) if rows else None,
        "mean_completion_tokens_mean": _nanmean(col("mean_completion_tokens")),
        "stability": {
            "nan_or_inf_loss_steps": sum(1 for x in losses if isinstance(x, float) and not math.isfinite(x)),
            "max_grad_norm": max(grad_norms) if grad_norms else None,
            "kl_last": rows[-1].get("kl") if rows else None,
            "kl_max": max([x for x in col("kl") if x is not None], default=None),
        },
        "samples_before": samples.get("before"),
        "samples_after": samples.get("after"),
    }

    if avg_excl:
        report["full_training_estimate_1_epoch"] = estimate_gpu_hours(
            num_questions=FULL_TRAIN_SIZE,
            epochs=1,
            num_generations=args.num_generations,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            sec_per_step_pilot=avg_excl,
            pilot_completions_per_step=batch_info["completions_per_step_per_gpu"],
        )
        report["estimate_formula"] = FORMULA
    return report


def plot_curves(rows, max_completion_length: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    get = lambda key: [r.get(key, float("nan")) for r in rows]  # noqa: E731

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ink, muted, accent = "#1f2937", "#9ca3af", "#2563eb"

    ax1.plot(steps, get("reward_mean"), color=accent, marker="o", ms=3, lw=1.8, label="mean reward (accuracy)")
    ax1.plot(steps, get("format_rate"), color=muted, lw=1.2, ls="--", label="format rate")
    ax1.plot(steps, get("frac_groups_all_same"), color="#f59e0b", lw=1.2, ls=":", label="groups all-same (adv=0)")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_xlabel("step")
    ax1.set_title("Reward", loc="left", color=ink)
    ax1.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    ax2.plot(steps, get("mean_completion_tokens"), color=accent, marker="o", ms=3, lw=1.8, label="mean length")
    ax2.plot(steps, get("max_completion_tokens"), color=muted, lw=1.2, label="max length")
    ax2.axhline(max_completion_length, color="#ef4444", lw=1, ls="--", label=f"limit = {max_completion_length}")
    ax2.set_xlabel("step")
    ax2.set_ylabel("tokens")
    ax2.set_title("Completion length", loc="left", color=ink)
    ax2.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def copy_to_drive(output_dir: Path, drive_dir: str, run_name: str) -> Path | None:
    if not drive_dir:
        return None
    if drive_dir.startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        print("[warning] Google Drive doesn't seem to be mounted (no /content/drive/MyDrive), skipping the copy.")
        return None
    dest = Path(drive_dir) / run_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output_dir, dest, dirs_exist_ok=True)
    print(f"[save] copied to Google Drive: {dest}")
    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    args = parse_args(argv)
    env = detect_environment()
    batch_info = check_batch_settings(args)

    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig, set_seed
    from trl import GRPOTrainer

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = args.checkpoint_dir or str(output_dir / "checkpoints")
    gpu_tag = env["gpu_name"].replace("NVIDIA", "").strip().split()[0].replace("/", "-")
    run_name = args.run_name or f"{output_dir.name}_{gpu_tag}_{datetime.now():%Y%m%d_%H%M%S}"

    # ---------------- 1. tokenizer + data ----------------
    # padding_side="left": for batched generation the padding goes on the left so the model continues right from the end of the prompt
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_ds, train_ds = build_grpo_dataset(tokenizer, n=args.n_questions)
    print_samples(raw_ds, train_ds, tokenizer, index=0)
    prompt_stats = prompt_token_stats(tokenizer, train_ds)
    args.n_questions = len(train_ds)  # n_questions=0 means all training questions; record the real count here
    if args.num_train_epochs > 0:
        args.max_steps = math.ceil(args.n_questions * args.num_train_epochs / batch_info["prompts_per_step"])
        print(
            f"[steps] {args.n_questions} questions x {args.num_train_epochs} epoch / "
            f"{batch_info['prompts_per_step']} questions per step -> max_steps = {args.max_steps}"
        )

    # ---------------- 2. quantization + LoRA config ----------------
    compute_dtype = torch.bfloat16 if env["use_bf16"] else torch.float16
    if args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # the 4-bit format from the QLoRA paper
            bnb_4bit_use_double_quant=True,  # quantize the quantization constants too, saves a bit more memory
            bnb_4bit_compute_dtype=compute_dtype,  # the actual matmuls run in bf16 (A100) or fp16 (T4)
        )
        print("[quantization] 4-bit nf4 (QLoRA)")
    else:
        # no quantization: the base loads in the dtype from model_init_kwargs (bf16 / fp16), about 6 GB for a 3B model
        bnb_config = None
        print(f"[quantization] none, base in {'bf16' if env['use_bf16'] else 'fp16'} + LoRA")
    target_modules = (
        "all-linear"
        if args.lora_target_modules.strip() == "all-linear"
        else [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---------------- 3. reward function ----------------
    grpo_config, dropped = make_grpo_config(args, env)
    tracker = RewardTracker()
    # "answer finished" is checked the same way TRL does: last token is EOS or pad -> finished; otherwise it got cut off
    # Qwen's end tokens come from generation_config (<|im_end|> and <|endoftext|>)
    gen_cfg = GenerationConfig.from_pretrained(args.model_name)
    eos_ids = gen_cfg.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
    eos_ids += [i for i in (tokenizer.eos_token_id, tokenizer.pad_token_id) if i is not None]
    reward_fn = make_correctness_reward(
        tracker=tracker,
        max_completion_length=args.max_completion_length,
        eos_token_ids=eos_ids,
        tokenizer=tokenizer,
    )

    # ---------------- 4. build GRPOTrainer ----------------
    # hand TRL the model name string + quantization_config + peft_config, and TRL takes care of:
    #   loading the model (4-bit or bf16/fp16) -> adding LoRA -> when beta>0, using "the same model with LoRA off" as the reference (no extra memory)
    PilotMetricsCallback, StopAfterStepCallback = _make_callback_class()
    resume_ckpt = None
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint

        if Path(args.checkpoint_dir).is_dir():
            resume_ckpt = get_last_checkpoint(args.checkpoint_dir)
        if resume_ckpt:
            print(f"[resume] found checkpoint: {resume_ckpt}")
        else:
            print(f"[resume] no checkpoint in {args.checkpoint_dir}, starting from scratch")
    callback = PilotMetricsCallback(
        tracker, output_dir / "metrics.csv", args.max_steps, resume=resume_ckpt is not None
    )
    callbacks = [callback]
    stop_early = 0 < args.stop_after_step < args.max_steps
    if stop_early:
        if args.save_steps <= 0:
            raise SystemExit("--stop_after_step needs --save_steps > 0 (it has to save a checkpoint before stopping so you can pick it back up)")
        if resume_ckpt and int(Path(resume_ckpt).name.split("-")[-1]) >= args.stop_after_step:
            raise SystemExit(f"[early stop] {Path(resume_ckpt).name} already exists, so the first {args.stop_after_step} steps are done; "
                             f"set --stop_after_step to 0 to finish the run")
        callbacks.append(StopAfterStepCallback(args.stop_after_step))
        print(f"[early stop] will save a checkpoint and stop at step {args.stop_after_step}/{args.max_steps} (LR schedule unchanged)")
    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=[reward_fn],  # just the one 0/1 correctness reward, no format reward
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        quantization_config=bnb_config,  # None = no quantization
        peft_config=lora_config,
        callbacks=callbacks,
    )

    model = trainer.model
    # TRL 1.16 casts QLoRA's LoRA params to bf16 (without quantization PEFT already uses fp32). Cast back to fp32 if needed:
    #  - T4 (fp16 mixed precision): has to be fp32, otherwise fp16 grad scaling / the optimizer break
    #  - A100: fp32 storage keeps small-LR updates like 1e-5 from being "rounded away" in bf16; costs < 0.5 GB extra
    if args.lora_param_dtype == "fp32":
        n_cast = 0
        for _, p in model.named_parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
                n_cast += 1
        print(f"[LoRA] cast {n_cast} trainable tensors to fp32")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    packed_note = " (4-bit weights counted as packed)" if args.quantization == "4bit" else ""
    print(f"[LoRA] trainable params {n_train / 1e6:.1f}M / total {n_total / 1e6:.0f}M{packed_note}")
    print(f"[memory] allocated after loading the model: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # ---------------- 5. samples before training ----------------
    sample_examples = [train_ds[i] for i in range(min(args.num_samples, len(train_ds)))]
    samples = {"before": show_samples(model, tokenizer, sample_examples, args, "[before training] sample answers")}

    # ---------------- 6. training ----------------
    print(f"\nStarting GRPO training: {args.max_steps} steps ...")
    if args.save_steps > 0:
        print(f"[checkpoint] every {args.save_steps} steps to {args.checkpoint_dir} (keeping the latest {args.save_total_limit})")
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_ckpt)  # when resuming, the time only counts this session
    train_seconds = time.perf_counter() - t0
    print(f"Training done in {train_seconds / 60:.1f} minutes")
    if stop_early and trainer.state.global_step < args.max_steps:
        # not finished yet: don't save a final adapter / report (so nobody mistakes it for a finished model), just plot the curves so far
        if callback.rows:
            plot_curves(callback.rows, args.max_completion_length, output_dir / "curves.png")
        print(f"[early stop] stopped at step {trainer.state.global_step}/{args.max_steps}, checkpoint is in {args.checkpoint_dir}. "
              f"To finish: drop --stop_after_step, add --resume and rerun the same command")
        return

    # ---------------- 7. samples after training ----------------
    samples["after"] = show_samples(trainer.model, tokenizer, sample_examples, args, "[after training] sample answers")

    # ---------------- 8. save ----------------
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))  # a PEFT model only saves the LoRA adapter (tens of MB)
    tokenizer.save_pretrained(str(adapter_dir))

    report = build_report(
        args, env, batch_info, trainer.args, callback.rows, dropped, samples, train_seconds, prompt_stats
    )
    report["run_name"] = run_name
    with open(output_dir / "pilot_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    if callback.rows:
        plot_curves(callback.rows, args.max_completion_length, output_dir / "curves.png")

    print("\n" + "=" * 80)
    print("Run summary")
    t = report["timing"]
    m = report["memory"]
    print(f"  GPU: {env['gpu_name']} | precision {env['precision']} | TRL {env['trl']}")
    print(f"  avg time per step {t['avg_step_time_s_excl_warmup']}s (skipping the first {t['warmup_steps_skipped']} steps)")
    print(f"  peak memory allocated {m['peak_mem_allocated_gb_max_over_steps']} GB / reserved "
          f"{m['peak_mem_reserved_gb_max_over_steps']} GB / total {m['gpu_total_memory_gb']} GB")
    print(f"  reward: first step {report['reward']['first_step']} -> last steps "
          f"{list(report['reward'].values())[1]}")
    print(f"  truncation rate {report['truncated_frac_mean']} | all-same group share {report['frac_groups_all_same_mean']}")
    if "full_training_estimate_1_epoch" in report:
        e = report["full_training_estimate_1_epoch"]
        print(f"  full-run estimate ({FULL_TRAIN_SIZE} questions x 1 epoch, same batch): {e['total_steps']} steps, "
              f"~ {e['gpu_hours']:.1f} GPU hours")
    print(f"  output folder: {output_dir.resolve()}")
    print("=" * 80)

    copy_to_drive(output_dir, args.drive_dir, run_name)


if __name__ == "__main__":
    main()
