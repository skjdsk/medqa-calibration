"""
train/grpo/estimate.py -- estimate GPU hours for a full run from the time per step measured in the pilot

Pilot only: used after the GRPO pilot to estimate the cost of the full run. Not part of training itself.
      (grpo.py imports FORMULA and estimate_gpu_hours from here for the pilot report, so the file has to stay.)

[What this file does]
In: number of questions, epochs, generations per question, batch settings (+ the pilot's pilot_report.json)
Out: expected total steps, time per step, wall-clock hours, GPU hours

[Formula] (in TRL GRPO, per_device_train_batch_size counts answers, not questions)

    answers per step     C   = per_device_train_batch_size x gradient_accumulation_steps x num_gpus
    questions per step   P   = C / num_generations                  (has to divide evenly; TRL requires it)
    total steps          S   = ceil(questions x epochs / P)

    time per step        t   = t_pilot x (C_per_gpu / C_pilot_per_gpu) x length_factor
            t_pilot         : average seconds per step in the pilot, after dropping the first few warmup steps
            C_per_gpu       : answers per GPU per step in the full run = per_device x grad_accum
            C_pilot_per_gpu : answers per step in the pilot (default 4 x 4 = 16)
            assumes time scales with answers per step -> on the safe side (generation grows slower than linear with batch size)
            length_factor   : if the full run uses a different max_completion_length, put new length / pilot length (also on the safe side)

    wall-clock hours     H   = S x t x overhead_factor / 3600
            overhead_factor: extra overhead (saving checkpoints, the odd long answer, Colab disconnects and reruns), default 1.15
    GPU hours                = H x num_gpus

Note: answers can get longer or shorter during training, and time per step changes with them.
      If the pilot's truncation rate (truncated_frac) is > 0, some answer in every batch hits 512 tokens,
      and HF generate waits for the longest one -> generation time is basically set by the longest answer.

[Command line]
    python -m train.grpo.estimate --report outputs/grpo_pilot/pilot_report.json \\
        --num_questions 10178 --epochs 1 --num_generations 4 \\
        --per_device_train_batch_size 4 --gradient_accumulation_steps 4
    # add --grid to print a comparison table of several setups
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

FORMULA = (
    "C = per_device_bs × grad_accum × num_gpus (answers per step); "
    "P = C / num_generations (questions per step); "
    "S = ceil(num_questions × epochs / P) (total steps); "
    "t = t_pilot × (per_device_bs × grad_accum) / C_pilot_per_gpu × length_factor (seconds per step); "
    "wall_hours = S × t × overhead_factor / 3600; gpu_hours = wall_hours × num_gpus"
)


def estimate_gpu_hours(
    num_questions: int,
    epochs: float,
    num_generations: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    sec_per_step_pilot: float,
    pilot_completions_per_step: int,
    num_gpus: int = 1,
    length_factor: float = 1.0,
    overhead_factor: float = 1.15,
) -> dict[str, Any]:
    """Estimate using the formula above. All inputs are plain numbers so it's easy to call from a notebook."""
    completions_per_gpu = per_device_train_batch_size * gradient_accumulation_steps
    completions_per_step = completions_per_gpu * num_gpus
    if completions_per_step % num_generations != 0:
        raise ValueError(
            f"answers per step {completions_per_step} (= {per_device_train_batch_size}x"
            f"{gradient_accumulation_steps}x{num_gpus}) has to be divisible by num_generations={num_generations}"
        )
    prompts_per_step = completions_per_step // num_generations
    total_steps = math.ceil(num_questions * epochs / prompts_per_step)

    sec_per_step = (
        sec_per_step_pilot * (completions_per_gpu / pilot_completions_per_step) * length_factor
    )
    wall_hours = total_steps * sec_per_step * overhead_factor / 3600.0
    gpu_hours = wall_hours * num_gpus

    return {
        "num_questions": num_questions,
        "epochs": epochs,
        "num_generations": num_generations,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_gpus": num_gpus,
        "completions_per_step": completions_per_step,
        "prompts_per_step": prompts_per_step,
        "total_steps": total_steps,
        "sec_per_step_pilot": sec_per_step_pilot,
        "pilot_completions_per_step": pilot_completions_per_step,
        "length_factor": length_factor,
        "overhead_factor": overhead_factor,
        "sec_per_step_estimated": sec_per_step,
        "wall_hours": wall_hours,
        "gpu_hours": gpu_hours,
        "formula": FORMULA,
    }


def load_pilot_timing(report_path: str | Path) -> tuple[float, int, int, str]:
    """Read (seconds per step, pilot answers per GPU per step, pilot max_completion_length, GPU name) from pilot_report.json."""
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    t = report["timing"]["avg_step_time_s_excl_warmup"]
    c = report["derived"]["completions_per_step_per_gpu"]
    L = report["config"]["max_completion_length"]
    return float(t), int(c), int(L), report.get("gpu_name", "unknown GPU")


def estimate_from_report(report_path: str | Path, max_completion_length: int | None = None, **kwargs):
    t, c, pilot_len, gpu = load_pilot_timing(report_path)
    length_factor = kwargs.pop("length_factor", None)
    if length_factor is None:
        length_factor = (max_completion_length / pilot_len) if max_completion_length else 1.0
    result = estimate_gpu_hours(
        sec_per_step_pilot=t, pilot_completions_per_step=c, length_factor=length_factor, **kwargs
    )
    result["pilot_gpu"] = gpu
    return result


def format_estimate(r: dict[str, Any]) -> str:
    return (
        f"questions={r['num_questions']} x epochs={r['epochs']} | {r['num_generations']} generations per question | "
        f"batch {r['per_device_train_batch_size']}x{r['gradient_accumulation_steps']}x{r['num_gpus']} GPU "
        f"-> {r['prompts_per_step']} questions/{r['completions_per_step']} answers per step | "
        f"total steps {r['total_steps']} | per step ~{r['sec_per_step_estimated']:.1f}s | "
        f"wall-clock ~{r['wall_hours']:.1f}h | GPU hours ~{r['gpu_hours']:.1f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate GPU hours for a full GRPO run from the pilot results")
    p.add_argument("--report", required=True, help="path to pilot_report.json")
    p.add_argument("--num_questions", type=int, default=10178)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--max_completion_length", type=int, default=None, help="defaults to the pilot's value")
    p.add_argument("--overhead_factor", type=float, default=1.15)
    p.add_argument("--grid", action="store_true", help="print a comparison table of several setups")
    args = p.parse_args()

    t, c, pilot_len, gpu = load_pilot_timing(args.report)
    print(f"[pilot] GPU={gpu}  {t:.1f}s per step  {c} answers per step  max_completion_length={pilot_len}")
    print(f"[formula] {FORMULA}\n")

    common = dict(
        num_gpus=args.num_gpus,
        overhead_factor=args.overhead_factor,
        max_completion_length=args.max_completion_length,
    )
    r = estimate_from_report(
        args.report,
        num_questions=args.num_questions,
        epochs=args.epochs,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **common,
    )
    print("[your setup] " + format_estimate(r))

    if args.grid:
        print("\n[comparison] (batch fixed at per_device x grad_accum answers per GPU per step)")
        for nq in (2000, 5000, args.num_questions):
            for gens in (4, 8):
                # keep questions per step the same: double the generations -> double grad_accum
                ga = args.gradient_accumulation_steps * gens // args.num_generations
                try:
                    rr = estimate_from_report(
                        args.report,
                        num_questions=nq,
                        epochs=args.epochs,
                        num_generations=gens,
                        per_device_train_batch_size=args.per_device_train_batch_size,
                        gradient_accumulation_steps=max(ga, 1),
                        **common,
                    )
                    print("  " + format_estimate(rr))
                except ValueError as e:
                    print(f"  questions={nq} generations={gens}: {e}")


if __name__ == "__main__":
    sys.exit(main())
