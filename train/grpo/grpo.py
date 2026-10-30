"""
train/grpo/grpo.py —— GRPO 试跑（pilot）主程序

【这个文件做什么（简单版）】
  1. 打印环境信息（GPU 型号、TRL / transformers / peft / bitsandbytes 版本），自动选择 bf16 或 fp16
  2. 从 MedQA **训练集**用固定种子抽 N 题（默认 64），构造 Qwen chat template 的 prompt
  3. 用 4-bit（nf4）加载 Qwen2.5-3B-Instruct，加上 LoRA adapter（QLoRA）；--quantization none 时改用 bf16 + LoRA
  4. 训练前：随机生成 2 个回答，打印内容和 reward
  5. 用 TRL 的 GRPOTrainer 训练 max_steps 步（默认 30），每一步记录：
        耗时、显存峰值、平均 reward / 标准差、组内 reward 全相同比例、回答长度、截断比例、格式正确率、KL
     → 写入 metrics.csv（每步一行，边训练边写，中途断掉也不会全丢）
  6. 训练后：再对同样 2 道题生成回答，打印对比
  7. 保存 LoRA adapter、metrics.csv、pilot_report.json、曲线图 curves.png；如果给了 --drive_dir，整个目录复制到 Google Drive

【GRPO 一步在做什么】（帮助理解耗时和显存从哪来）
  每个"优化步"（optimizer step）：
   a) 取 P 道题（P = 每步回答数 / num_generations，默认 16/4 = 4 题）
   b) 每道题生成 num_generations 个回答（默认 4 个）→ 一共 16 条回答        ← 通常最慢的部分（逐 token 生成）
   c) 用 rewards.py 给每条回答打 0/1 分
   d) 组内比较得到 advantage：advantage = (reward - 组内平均) / 组内标准差（全对/全错的组 advantage = 0）
   e) 分 gradient_accumulation_steps 次（默认 4 次，每次 4 条回答）前向+反向，累积梯度后更新一次 LoRA
      如果 beta > 0，还要额外算一次"关掉 LoRA 的原始模型"的概率，用来计算 KL 惩罚

【哪些参数影响显存 / 速度】
  显存：per_device_train_batch_size（每次前向/反向的回答条数，最主要）、max_completion_length（序列长度）、
        gradient_checkpointing（开着省显存但慢 ~20–30%）、lora_r（影响很小）、use_vllm（会额外占显存）
  速度：max_completion_length（生成要逐 token 走完，最长那条回答决定生成时间）、
        每步回答数 = per_device_train_batch_size × gradient_accumulation_steps、num_generations、
        beta > 0（多一次参考模型前向，约 +10–20%）、use_vllm（A100 上生成可快数倍）

【运行】（在 repo 根目录）
    python -m train.grpo.grpo --n_questions 64 --max_steps 30 --output_dir outputs/grpo_pilot \\
        --drive_dir /content/drive/MyDrive/medqa-calibration/grpo_pilot
  正式训练（按 epoch 算步数，每 50 步存 checkpoint 到 Drive；Colab 断开后加 --resume 重跑同一条命令即可续上）：
    python -m train.grpo.grpo --n_questions 2000 --num_train_epochs 1 --max_completion_length 768 \\
        --output_dir /content/drive/MyDrive/medqa-calibration/grpo --save_steps 50 --resume
  全部参数见 python -m train.grpo.grpo --help
"""

from __future__ import annotations

import os

# 减少显存碎片导致的"明明还有空间却 OOM"；必须在 import torch 之前设置
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

# 让 "from train.xxx import ..." 不管用 python -m train.grpo.grpo 还是 python train/grpo/grpo.py 都能工作
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from train.grpo.estimate import FORMULA, estimate_gpu_hours  # noqa: E402
from train.grpo.grpo_data import build_grpo_dataset, print_samples, prompt_token_stats  # noqa: E402
from train.grpo.rewards import (  # noqa: E402
    RewardTracker,
    extract_answer,
    make_correctness_reward,
    score_completion,
)

FULL_TRAIN_SIZE = 10178  # MedQA-USMLE-4-options 训练集题数，用于报告里的"正式训练"估算


# ---------------------------------------------------------------------------
# 命令行参数（全部可配置）
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO pilot on MedQA (QLoRA + TRL GRPOTrainer)")

    # 数据
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n_questions", type=int, default=64, help="从训练集抽多少题")
    p.add_argument("--seed", type=int, default=42, help="抽题和训练的随机种子")

    # 量化
    p.add_argument(
        "--quantization",
        choices=["4bit", "none"],
        default="4bit",
        help="4bit（默认）= QLoRA，基座 nf4；none = 基座不量化，用 bf16（A100）/ fp16（T4）+ LoRA。"
        "QLoRA + --use_vllm 时，每步把 LoRA 合并进 4-bit 权重同步给 vLLM 会累积舍入误差，"
        "A100 试跑中 importance_sampling_ratio 掉到接近 0、梯度消失；开 vLLM 时建议用 none",
    )

    # LoRA（之后 SFT / DPO 用同样的设置）
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        default="all-linear",
        help='"all-linear" 或逗号分隔的模块名，例如 "q_proj,k_proj,v_proj,o_proj"',
    )
    p.add_argument(
        "--lora_param_dtype",
        choices=["fp32", "trl_default"],
        default="fp32",
        help="LoRA 参数的存储精度。TRL 1.16 会把 QLoRA 的 LoRA 参数强制转成 bf16；"
        "fp32（默认）会转回 fp32：T4 上 fp16 混合精度必须这样，A100 上可避免小学习率的更新被 bf16 舍入掉。",
    )

    # GRPO 核心参数
    p.add_argument("--num_generations", type=int, default=4, help="每题生成几个回答（组大小 G）")
    p.add_argument("--max_completion_length", type=int, default=512, help="回答最多多少 token")
    p.add_argument("--max_steps", type=int, default=30, help="训练多少步；--num_train_epochs > 0 时忽略")
    p.add_argument(
        "--num_train_epochs",
        type=float,
        default=0,
        help="> 0 时按 epoch 训练：总步数 = ceil(题数 × epochs / 每步题数)，覆盖 --max_steps（正式训练用）",
    )
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="KL 系数。>0 时会记录 KL（用关掉 LoRA 的模型当参考，不额外占显存，但每步多一次前向）。"
        "注意 TRL 1.16 的默认值是 0.0。",
    )
    p.add_argument("--temperature", type=float, default=1.0, help="生成温度")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument(
        "--loss_type",
        default=None,
        help="不填 = TRL 默认（1.16 为 'dapo'）。可选 'grpo'（原论文按序列长度归一化）、'dr_grpo' 等",
    )
    p.add_argument(
        "--scale_rewards",
        default=None,
        help="不填 = TRL 默认（'group'：advantage 除以组内标准差）。可选 'batch' / 'none'",
    )

    # Batch 设置。注意：TRL GRPO 里 per_device_train_batch_size 的单位是"回答条数"
    #   每步回答数 = per_device_train_batch_size × gradient_accumulation_steps × GPU 数
    #   必须能被 num_generations 整除；每步题数 = 每步回答数 / num_generations
    #   默认 4 × 4 = 16 条回答 = 4 题 × 4 个回答
    p.add_argument("--per_device_train_batch_size", type=int, default=4, help="每次前向/反向几条回答（显存主要旋钮）")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--no_gradient_checkpointing", action="store_true", help="关闭梯度检查点（更快但更占显存）")

    # vLLM（可选，A100 上加速生成）
    p.add_argument("--use_vllm", action="store_true", help="用 vLLM colocate 模式生成（需要 pip install 'trl[vllm]'）")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    p.add_argument("--vllm_max_model_length", type=int, default=2048)

    # 输出
    p.add_argument("--output_dir", default="outputs/grpo_pilot")
    p.add_argument("--drive_dir", default="", help="Google Drive 目标目录；留空则不复制")
    p.add_argument(
        "--run_name",
        default="",
        help="Drive 里的子目录名；留空则自动用 output_dir 的文件夹名 + GPU + 时间（如 grpo_Tesla_20261025_213538）",
    )

    # Checkpoint / 断点续跑（正式训练用；试跑默认不存 checkpoint）
    p.add_argument(
        "--save_steps",
        type=int,
        default=0,
        help="每隔多少步存一次 checkpoint（含优化器状态，用于断点续跑）；0 = 不存（试跑默认）",
    )
    p.add_argument("--save_total_limit", type=int, default=2, help="最多保留几个 checkpoint，旧的自动删除")
    p.add_argument(
        "--checkpoint_dir",
        default="",
        help="checkpoint 存放目录；留空 = <output_dir>/checkpoints。"
        "Colab 上建议放在 Google Drive，断开连接后才不会丢",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="从 checkpoint_dir 里最新的 checkpoint 继续训练；找不到 checkpoint 则从头开始",
    )
    p.add_argument("--num_samples", type=int, default=2, help="训练前后各展示几个生成样例")
    p.add_argument("--warmup_skip", type=int, default=2, help="算平均每步耗时时跳过前几步（预热）")
    p.add_argument("--report_last_k", type=int, default=5, help="报告里'最后几步'的平均取几步")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# 环境检测
# ---------------------------------------------------------------------------
def detect_environment() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("没有检测到 GPU。请在 Colab 里选择 运行时 → 更改运行时类型 → GPU。")

    import bitsandbytes
    import peft
    import transformers
    import trl

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # 用计算能力判断：>= 8.0（A100/L4/H100 等 Ampere 及以后）原生支持 bf16；T4 是 7.5，只能用 fp16。
    # （不用 torch.cuda.is_bf16_supported()：新版 PyTorch 在 T4 上可能因"软件模拟"返回 True。）
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
    print(f"TRL 版本: {trl.__version__}   （本代码按 TRL 1.16 的参数名编写）")
    print(
        f"GPU: {name} (compute {major}.{minor}, {total_gb:.1f} GB) → 精度 {env['precision']}\n"
        f"torch {env['torch']} | CUDA {env['cuda']} | transformers {env['transformers']} | "
        f"peft {env['peft']} | bitsandbytes {env['bitsandbytes']} | python {env['python']}"
    )
    print("=" * 80)
    return env


# ---------------------------------------------------------------------------
# GRPOConfig：按当前 TRL 版本的参数名构造；不认识的参数会被丢弃并打印警告
# ---------------------------------------------------------------------------
def check_batch_settings(args) -> dict[str, int]:
    n_gpus = int(os.environ.get("WORLD_SIZE", "1"))  # Colab 单卡 = 1
    completions_per_gpu = args.per_device_train_batch_size * args.gradient_accumulation_steps
    completions_per_step = completions_per_gpu * n_gpus
    if args.num_generations < 2:
        raise SystemExit("num_generations 至少为 2（GRPO 需要组内比较）。")
    if completions_per_step % args.num_generations != 0:
        raise SystemExit(
            f"batch 设置不合法：每步回答数 = per_device_train_batch_size({args.per_device_train_batch_size}) × "
            f"gradient_accumulation_steps({args.gradient_accumulation_steps}) × GPU数({n_gpus}) = "
            f"{completions_per_step}，必须能被 num_generations({args.num_generations}) 整除。\n"
            f"建议：让 per_device_train_batch_size × gradient_accumulation_steps 是 num_generations 的倍数，"
            f"例如 num_generations=4 时用 4×4 或 2×8；num_generations=8 时用 4×8 或 8×4。"
        )
    info = {
        "num_gpus": n_gpus,
        "completions_per_step_per_gpu": completions_per_gpu,
        "completions_per_step": completions_per_step,
        "prompts_per_step": completions_per_step // args.num_generations,
    }
    print(
        f"[batch] 每步 {info['completions_per_step']} 条回答 = {info['prompts_per_step']} 题 × "
        f"{args.num_generations} 个回答；每次前向/反向 {args.per_device_train_batch_size} 条，"
        f"累积 {args.gradient_accumulation_steps} 次更新一次"
    )
    return info


def make_grpo_config(args, env):
    from trl import GRPOConfig

    dtype_str = "bfloat16" if env["use_bf16"] else "float16"
    kwargs: dict[str, Any] = dict(
        output_dir=args.checkpoint_dir,  # Trainer 把 checkpoint-<step>/ 存在这里
        # ---- 训练步数 / 学习率 ----
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        # ---- batch（显存和速度的主要旋钮）----
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # TRL 1.16：generation_batch_size 默认 = per_device × GPU数 × steps_per_generation，
        #           steps_per_generation 默认 = gradient_accumulation_steps
        #           → 每个优化步只生成一次，一次生成全部 16 条回答。这里不手动设置，保持默认。
        # ---- GRPO ----
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,  # 不做 top-k 截断（TRL 默认）。注意 Qwen 自带 generation_config 是 top_k=20，这里显式覆盖
        repetition_penalty=1.0,
        beta=args.beta,
        num_iterations=1,  # 每批生成的数据只更新一次（标准 GRPO 设置）
        # ---- 精度 / 显存 ----
        bf16=env["use_bf16"],
        fp16=not env["use_bf16"],
        gradient_checkpointing=not args.no_gradient_checkpointing,
        # TRL 用这些参数加载模型（因为我们把模型名字符串交给 GRPOTrainer）
        model_init_kwargs={"dtype": dtype_str},
        # ---- 日志 ----
        logging_steps=1,  # 每步都记录，回调里才能拿到每步的 KL 等指标
        report_to="none",
        # 试跑（save_steps=0）不存中间 checkpoint，结束时手动存 adapter；
        # 正式训练每 save_steps 步存一次（含优化器状态），只保留最近 save_total_limit 个
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=args.save_total_limit,
        log_completions=False,
        remove_unused_columns=False,  # 保留 "answer" 列，打分函数要用
        # ---- vLLM（可选）----
        use_vllm=args.use_vllm,
    )
    if args.use_vllm:
        kwargs.update(
            vllm_mode="colocate",  # 和训练共用同一张 GPU，不需要单独起服务
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_length=args.vllm_max_model_length,
        )
    if args.loss_type:
        kwargs["loss_type"] = args.loss_type
    if args.scale_rewards:
        kwargs["scale_rewards"] = args.scale_rewards

    # 版本兼容：只传当前 TRL 版本认识的参数，其余丢弃并警告（TRL API 变化很快）
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in kwargs if k not in valid)
    if dropped:
        print(f"[警告] 当前 TRL 版本的 GRPOConfig 不认识这些参数，已忽略：{dropped}。请检查新版的对应参数名。")
    config = GRPOConfig(**{k: v for k, v in kwargs.items() if k in valid})
    return config, dropped


# ---------------------------------------------------------------------------
# 每步指标记录：TrainerCallback
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "step",
    "step_time_s",  # 这一步（生成 + 打分 + 前向反向 + 更新）的耗时
    "peak_mem_allocated_gb",  # torch.cuda.max_memory_allocated（这一步内的峰值）
    "peak_mem_reserved_gb",  # PyTorch 实际向显卡申请的峰值（更接近 nvidia-smi 看到的数）
    "reward_mean",
    "reward_std",
    "frac_groups_all_same",  # 组内 reward 全相同（全对或全错）→ advantage 为 0 的组的比例
    "frac_groups_all_correct",
    "frac_groups_all_wrong",
    "format_rate",  # 能提取出 "Answer: X" 的比例
    "mean_completion_tokens",
    "max_completion_tokens",
    "truncated_frac",  # 写满 max_completion_length 还没结束的比例
    "n_completions",
    "n_groups",
    "kl",  # TRL 记录的 KL（beta > 0 时才有）
    "loss",
    "grad_norm",
    "entropy",
    "learning_rate",
    # TRL 自己算的同类指标（用于交叉核对我们自己的统计）
    "trl_reward",
    "trl_reward_std",
    "trl_frac_reward_zero_std",
    "trl_mean_length",
    "trl_clipped_ratio",
]

# TRL 日志 key → CSV 列名（TRL 1.16 的 key；新版本如果改名，对应列会是空的，不影响训练）
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
    """读回 metrics.csv（续跑时用）：数字列转回 int/float，空格子转成 None。"""
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
        """每个优化步：开始时清零显存峰值并计时；结束时汇总打分函数收集的指标；
        on_log 时合并 TRL 记录的 KL 等指标，写一行 CSV。"""

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
            # 断点续跑：保留 checkpoint 之前的行（之后的步会重跑，丢掉），再接着往后写
            if self.resume and self.csv_path.exists() and state.global_step > 0:
                self.rows = [r for r in _read_metrics_csv(self.csv_path) if r["step"] <= state.global_step]
                print(f"[续跑] 从第 {state.global_step} 步继续；metrics.csv 保留前 {len(self.rows)} 行")
            self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS, extrasaction="ignore")
            self._writer.writeheader()
            self._writer.writerows(self.rows)
            self._file.flush()
            self.tracker.pop_step_stats()  # 清掉训练开始前可能残留的记录

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
                f"±{f('reward_std', '.3f')} | 全相同组 {f('frac_groups_all_same', '.2f')} | "
                f"格式 {f('format_rate', '.2f')} | 长度 {f('mean_completion_tokens', '.0f')} | "
                f"截断 {f('truncated_frac', '.2f')} | kl {f('kl', '.4f')}",
                flush=True,
            )

    return PilotMetricsCallback


# ---------------------------------------------------------------------------
# 训练前后的生成样例
# ---------------------------------------------------------------------------
@torch.no_grad()
def show_samples(model, tokenizer, examples, args, title: str) -> list[dict[str, Any]]:
    """对给定的题目各生成 1 个回答（采样参数与训练一致），打印回答和 reward。"""
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
    print(f"\n{bar}\n{title}（{len(examples)} 题，生成耗时 {dt:.1f}s）\n{bar}")
    for i, (ex, text, ids) in enumerate(zip(examples, texts, gen)):
        n_tokens = int((ids != tokenizer.pad_token_id).sum().item())
        pred = extract_answer(text)
        reward = score_completion(text, ex["answer"])
        results.append(
            {"answer": ex["answer"], "pred": pred, "reward": reward, "num_tokens": n_tokens, "completion": text}
        )
        print(f"--- 样例 {i + 1} | 标准答案 {ex['answer']} | 提取到 {pred} | reward = {reward} | {n_tokens} tokens ---")
        print(text.strip())
        print()
    if was_training:
        model.train()
    return results


# ---------------------------------------------------------------------------
# 报告 / 图
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
        "effective_grpo_config": effective,  # TRL 实际使用的值（含默认值）
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
        print("[警告] Google Drive 似乎没有挂载（找不到 /content/drive/MyDrive），跳过复制。")
        return None
    dest = Path(drive_dir) / run_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output_dir, dest, dirs_exist_ok=True)
    print(f"[保存] 已复制到 Google Drive：{dest}")
    return dest


# ---------------------------------------------------------------------------
# 主流程
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

    # ---------------- 1. tokenizer + 数据 ----------------
    # padding_side="left"：批量生成时 padding 放在左边，模型才能从 prompt 末尾接着写
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_ds, train_ds = build_grpo_dataset(tokenizer, n=args.n_questions, seed=args.seed)
    print_samples(raw_ds, train_ds, tokenizer, index=0)
    prompt_stats = prompt_token_stats(tokenizer, train_ds)
    args.n_questions = len(train_ds)  # n_questions=0 表示全量；这里记下实际题数
    if args.num_train_epochs > 0:
        args.max_steps = math.ceil(args.n_questions * args.num_train_epochs / batch_info["prompts_per_step"])
        print(
            f"[步数] {args.n_questions} 题 × {args.num_train_epochs} epoch ÷ 每步 "
            f"{batch_info['prompts_per_step']} 题 → max_steps = {args.max_steps}"
        )

    # ---------------- 2. 量化 + LoRA 配置 ----------------
    compute_dtype = torch.bfloat16 if env["use_bf16"] else torch.float16
    if args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # QLoRA 论文用的 4-bit 格式
            bnb_4bit_use_double_quant=True,  # 对量化常数再量化一次，再省一点显存
            bnb_4bit_compute_dtype=compute_dtype,  # 实际矩阵乘法用 bf16（A100）或 fp16（T4）
        )
        print("[量化] 4-bit nf4（QLoRA）")
    else:
        # 不量化：基座按 model_init_kwargs 里的 dtype（bf16 / fp16）加载，3B 模型约 6 GB
        bnb_config = None
        print(f"[量化] 不量化，基座用 {'bf16' if env['use_bf16'] else 'fp16'} + LoRA")
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

    # ---------------- 3. 打分函数 ----------------
    grpo_config, dropped = make_grpo_config(args, env)
    tracker = RewardTracker()
    # "回答已结束"的判断与 TRL 一致：最后一个 token 是结束符或 pad → 正常结束；否则 = 被截断
    # Qwen 的结束符来自 generation_config（<|im_end|> 和 <|endoftext|>）
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

    # ---------------- 4. 构造 GRPOTrainer ----------------
    # 把模型名字符串 + quantization_config + peft_config 交给 TRL，由 TRL 负责：
    #   加载模型（4-bit 或 bf16/fp16）→ 套上 LoRA → beta>0 时用"关掉 LoRA 的同一个模型"当参考模型（不额外占显存）
    PilotMetricsCallback = _make_callback_class()
    resume_ckpt = None
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint

        if Path(args.checkpoint_dir).is_dir():
            resume_ckpt = get_last_checkpoint(args.checkpoint_dir)
        if resume_ckpt:
            print(f"[续跑] 找到 checkpoint：{resume_ckpt}")
        else:
            print(f"[续跑] {args.checkpoint_dir} 里没有 checkpoint，从头开始训练")
    callback = PilotMetricsCallback(
        tracker, output_dir / "metrics.csv", args.max_steps, resume=resume_ckpt is not None
    )
    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=[reward_fn],  # 只有一个 0/1 正确性 reward，没有格式奖励
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        quantization_config=bnb_config,  # None = 不量化
        peft_config=lora_config,
        callbacks=[callback],
    )

    model = trainer.model
    # TRL 1.16 会把 QLoRA 的 LoRA 参数转成 bf16（不量化时 PEFT 默认就是 fp32）。这里按需转回 fp32：
    #  - T4（fp16 混合精度）：必须是 fp32，否则 fp16 的梯度缩放/优化器会出问题
    #  - A100：fp32 存储可避免 1e-5 这种小学习率的更新被 bf16 精度"舍入掉"；多占的显存 < 0.5 GB
    if args.lora_param_dtype == "fp32":
        n_cast = 0
        for _, p in model.named_parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
                n_cast += 1
        print(f"[LoRA] 已把 {n_cast} 个可训练张量转为 fp32")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    packed_note = "（4-bit 权重按打包后计数）" if args.quantization == "4bit" else ""
    print(f"[LoRA] 可训练参数 {n_train / 1e6:.1f}M / 总参数 {n_total / 1e6:.0f}M{packed_note}")
    print(f"[显存] 模型加载后已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # ---------------- 5. 训练前样例 ----------------
    sample_examples = [train_ds[i] for i in range(min(args.num_samples, len(train_ds)))]
    samples = {"before": show_samples(model, tokenizer, sample_examples, args, "【训练前】生成样例")}

    # ---------------- 6. 训练 ----------------
    print(f"\n开始 GRPO 训练：{args.max_steps} 步 …")
    if args.save_steps > 0:
        print(f"[checkpoint] 每 {args.save_steps} 步存到 {args.checkpoint_dir}（保留最近 {args.save_total_limit} 个）")
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_ckpt)  # 续跑时这次的耗时只算本次会话
    train_seconds = time.perf_counter() - t0
    print(f"训练结束，用时 {train_seconds / 60:.1f} 分钟")

    # ---------------- 7. 训练后样例 ----------------
    samples["after"] = show_samples(trainer.model, tokenizer, sample_examples, args, "【训练后】生成样例")

    # ---------------- 8. 保存 ----------------
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))  # PEFT 模型只保存 LoRA adapter（几十 MB）
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
    print("试跑总结")
    t = report["timing"]
    m = report["memory"]
    print(f"  GPU: {env['gpu_name']} | 精度 {env['precision']} | TRL {env['trl']}")
    print(f"  平均每步 {t['avg_step_time_s_excl_warmup']}s（跳过前 {t['warmup_steps_skipped']} 步）")
    print(f"  显存峰值 allocated {m['peak_mem_allocated_gb_max_over_steps']} GB / reserved "
          f"{m['peak_mem_reserved_gb_max_over_steps']} GB / 总 {m['gpu_total_memory_gb']} GB")
    print(f"  reward：第一步 {report['reward']['first_step']} → 最后几步 "
          f"{list(report['reward'].values())[1]}")
    print(f"  截断比例 {report['truncated_frac_mean']} | 组内全相同比例 {report['frac_groups_all_same_mean']}")
    if "full_training_estimate_1_epoch" in report:
        e = report["full_training_estimate_1_epoch"]
        print(f"  正式训练估算（{FULL_TRAIN_SIZE} 题 × 1 epoch，同样 batch）：{e['total_steps']} 步，"
              f"≈ {e['gpu_hours']:.1f} GPU 小时")
    print(f"  输出目录：{output_dir.resolve()}")
    print("=" * 80)

    copy_to_drive(output_dir, args.drive_dir, run_name)


if __name__ == "__main__":
    main()
