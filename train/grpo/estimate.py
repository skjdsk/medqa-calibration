"""
train/grpo/estimate.py —— 用试跑测到的"每步耗时"估算正式训练要多少 GPU 小时

⚠️ 试跑专用：只在 GRPO 试跑（pilot）之后用来估算正式训练的开销，不是训练流程本身的一部分。
   （grpo.py 会 import 这里的 FORMULA 和 estimate_gpu_hours 写进试跑报告，所以文件需要保留。）

【这个文件做什么（简单版）】
输入：正式训练的题数、epoch 数、每题生成数、batch 设置（+ 试跑报告 pilot_report.json）
输出：预计总步数、每步耗时、墙钟小时数、GPU 小时数

【公式】（TRL GRPO 里 per_device_train_batch_size 的单位是"回答条数"，不是"题数"）

  每步回答数   C   = per_device_train_batch_size × gradient_accumulation_steps × num_gpus
  每步题数     P   = C / num_generations                  （必须整除，TRL 的硬性要求）
  总步数       S   = ceil(题数 × epochs / P)

  每步耗时     t   = t_pilot × (C_per_gpu / C_pilot_per_gpu) × length_factor
      t_pilot         ：试跑中去掉前几步预热后的平均每步秒数
      C_per_gpu       ：正式训练每张卡每步处理的回答数 = per_device × grad_accum
      C_pilot_per_gpu ：试跑每步回答数（默认 4 × 4 = 16）
      假设耗时与"每步回答数"成正比 → 偏保守（生成阶段 batch 变大时实际增长慢于线性）
      length_factor   ：如果正式训练的 max_completion_length 不同，填 新长度/试跑长度（同样偏保守）

  墙钟小时     H   = S × t × overhead_factor / 3600
      overhead_factor：额外开销（保存 checkpoint、偶尔变长的回答、Colab 断连重跑），默认 1.15
  GPU 小时         = H × num_gpus

⚠️ 注意：训练过程中模型回答可能变长或变短，每步耗时会随之变化。
   如果试跑里截断比例（truncated_frac）> 0，说明每批都有回答写满 512 token，
   而 HF generate 要等最长的那条写完 → 生成时间基本被"最长回答"决定。

【命令行用法】
    python -m train.grpo.estimate --report outputs/grpo_pilot/pilot_report.json \\
        --num_questions 10178 --epochs 1 --num_generations 4 \\
        --per_device_train_batch_size 4 --gradient_accumulation_steps 4
    # 加 --grid 打印多种方案对比表
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

FORMULA = (
    "C = per_device_bs × grad_accum × num_gpus（每步回答数）; "
    "P = C / num_generations（每步题数）; "
    "S = ceil(num_questions × epochs / P)（总步数）; "
    "t = t_pilot × (per_device_bs × grad_accum) / C_pilot_per_gpu × length_factor（每步秒数）; "
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
    """按上面的公式估算。所有输入都是普通数字，方便在 notebook 里直接调用。"""
    completions_per_gpu = per_device_train_batch_size * gradient_accumulation_steps
    completions_per_step = completions_per_gpu * num_gpus
    if completions_per_step % num_generations != 0:
        raise ValueError(
            f"每步回答数 {completions_per_step}（= {per_device_train_batch_size}×"
            f"{gradient_accumulation_steps}×{num_gpus}）必须能被 num_generations={num_generations} 整除"
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
    """从 pilot_report.json 读出 (每步秒数, 试跑每卡每步回答数, 试跑 max_completion_length, GPU 名)。"""
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
        f"题数={r['num_questions']} × epochs={r['epochs']} | 每题生成 {r['num_generations']} | "
        f"batch {r['per_device_train_batch_size']}×{r['gradient_accumulation_steps']}×{r['num_gpus']}卡 "
        f"→ 每步 {r['prompts_per_step']} 题/{r['completions_per_step']} 条回答 | "
        f"总步数 {r['total_steps']} | 每步≈{r['sec_per_step_estimated']:.1f}s | "
        f"墙钟≈{r['wall_hours']:.1f}h | GPU 小时≈{r['gpu_hours']:.1f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="用试跑结果估算正式 GRPO 训练的 GPU 小时")
    p.add_argument("--report", required=True, help="pilot_report.json 路径")
    p.add_argument("--num_questions", type=int, default=10178)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--max_completion_length", type=int, default=None, help="不填则与试跑相同")
    p.add_argument("--overhead_factor", type=float, default=1.15)
    p.add_argument("--grid", action="store_true", help="打印多种方案的对比表")
    args = p.parse_args()

    t, c, pilot_len, gpu = load_pilot_timing(args.report)
    print(f"[试跑] GPU={gpu}  每步 {t:.1f}s  每步 {c} 条回答  max_completion_length={pilot_len}")
    print(f"[公式] {FORMULA}\n")

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
    print("[你的方案] " + format_estimate(r))

    if args.grid:
        print("\n[方案对比]（batch 设置固定为每卡每步 per_device × grad_accum 条回答）")
        for nq in (2000, 5000, args.num_questions):
            for gens in (4, 8):
                # 保持"每步题数"不变：生成数翻倍 → grad_accum 翻倍
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
                    print(f"  题数={nq} 生成数={gens}: {e}")


if __name__ == "__main__":
    sys.exit(main())
