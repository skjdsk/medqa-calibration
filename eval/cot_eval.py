"""
cot_eval.py —— 用 GRPO 训练时的 CoT prompt 评估模型（vLLM 生成），输出和 evaluate.py 同格式的每题明细 CSV
=====================================================================================================

【为什么需要这个文件】
evaluate.py 是"直接作答"：要求只输出字母，看第一个 token 上 A–D 的概率。
GRPO 训练时模型是"先推理，最后写 Answer: X"（train/grpo/grpo_data.py 的 prompt）。
只用直接作答评估，可能看不到 GRPO 学到的东西，所以这里用**完全相同的训练 prompt**再评估一次。

【做什么】
1. 用 vLLM 加载基座（bf16，不量化），给了 --adapter 就叠上 LoRA。
2. 贪心生成（temperature 0）：
     - 用和训练 reward 相同的规则（train/grpo/rewards.py 的 extract_answer）取最后一个 "Answer: X"
     - 在 "Answer:" 后面那个字母 token 的位置，读 vLLM 返回的 top-20 logprobs，
       取 A–D（"A" 和 " A" 两种写法合并）的概率，在 4 个字母上归一化 → prob_A..prob_D
       这就是 CoT 下的 confidence（eval_mode = "cot"）
     - 没按格式写出答案的题：4 个选项各 0.25（相当于"没回答"），format_ok = 0
3. 采样 --n_samples 次（temperature 1.0，和训练时相同）：
     - 每个选项被选中的比例 → prob_A..prob_D（eval_mode = "cot_vote{K}"，投票式 confidence）
     - sample_acc = 采样回答的平均正确率，可以直接和训练时的 reward 对比
4. 输出 preds_{模型}_medqa4_val_{mode}.csv，可直接用 eval/metrics.py 计算 ECE / Brier / NLL 等。

【运行示例】（A100；每个模型约 5–10 分钟；两个模型分两次运行，vLLM 不方便在同一个进程里换模型）
    python -m eval.cot_eval --model_name base --out_dir results/val
    python -m eval.cot_eval --adapter /path/to/adapter --model_name grpo_seed1 --out_dir results/val
    python -m eval.metrics --pred_csv "results/val/preds_*.csv" --out_dir results/val/metrics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import time
from datetime import datetime, timezone

import pandas as pd

from eval.data_loaders import LETTERS as ALL_LETTERS, load_items
from train.grpo.grpo_data import LETTERS, build_prompt_text
from train.grpo.rewards import extract_answer

BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"
PROMPT_VERSION = "grpo-cot-v1"  # = train/grpo/grpo_data.py 的 SYSTEM_PROMPT + INSTRUCTION

# 字母 token 前面紧挨着的文字要像 "Answer:" / "**Answer:** " / "Answer: (" —— 和 rewards._ANSWER_RE 的前半段一致
_ANSWER_PREFIX_RE = re.compile(r"\b(?i:answer)\s*[*_]*\s*[:：]\s*[*_]*\s*\(?\s*$")


# ---------------------------------------------------------------------------
# 纯 Python 的小工具（不依赖 vLLM，test_cot_eval.py 里有单元测试）
# ---------------------------------------------------------------------------
def find_answer_token_index(token_ids: list[int], letter_ids: dict[str, list[int]], decode) -> int | None:
    """从后往前找最后一个"紧跟在 Answer: 后面的字母 token"的位置；找不到返回 None。

    decode(ids) -> str，用 tokenizer.decode。只解码字母前面的几个 token，速度很快。
    """
    id_to_letter = {i: L for L, ids in letter_ids.items() for i in ids}
    for j in range(len(token_ids) - 1, -1, -1):
        if token_ids[j] not in id_to_letter:
            continue
        before = decode(token_ids[max(0, j - 8):j])
        # " A" 这种带空格的 token：空格属于字母 token，所以前文以 "Answer:" 结尾即可
        if _ANSWER_PREFIX_RE.search(before):
            return j
    return None


def letter_probs_from_logprobs(logprobs: dict[int, float], letter_ids: dict[str, list[int]]) -> tuple[dict, float]:
    """{token_id: logprob}（top-k）→ 4 个字母上归一化的概率，以及归一化前的字母概率总和 letter_mass。

    不在 top-k 里的字母当作概率 0（top-20 之外的概率本来就极小）。
    """
    mass = {L: sum(math.exp(logprobs[i]) for i in ids if i in logprobs) for L, ids in letter_ids.items()}
    total = sum(mass.values())
    if total <= 0:
        return {L: 1.0 / len(letter_ids) for L in letter_ids}, 0.0
    return {L: m / total for L, m in mass.items()}, total


def vote_probs(preds: list[str | None], letters=LETTERS) -> dict[str, float]:
    """K 个采样答案 → 每个字母的得票比例。没按格式回答的不计票；全部没回答时均匀分布。"""
    valid = [p for p in preds if p in letters]
    if not valid:
        return {L: 1.0 / len(letters) for L in letters}
    return {L: valid.count(L) / len(valid) for L in letters}


def make_row(item, model_name: str, eval_mode: str, probs: dict[str, float], **extra) -> dict:
    """和 evaluate.build_rows 同样的列，metrics.py 可以直接读。"""
    pred = max(probs, key=probs.get)  # 并列时取靠前的字母
    row = {
        "dataset": item.dataset, "question_id": item.question_id, "model_name": model_name,
        "eval_mode": eval_mode, "n_options": len(item.options), "gold": item.gold, "pred": pred,
        "confidence": probs[pred], "correct": int(pred == item.gold),
    }
    row.update(extra)
    for L in ALL_LETTERS[:len(item.options)]:
        row[f"prob_{L}"] = probs[L]
    return row


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CoT 评估（GRPO 训练 prompt，vLLM 生成）")
    p.add_argument("--base_model", default=BASE_MODEL_DEFAULT)
    p.add_argument("--adapter", default=None, help="LoRA adapter 目录；不给则评估原始模型")
    p.add_argument("--model_name", default=None, help="写进 CSV 的模型名；默认 base 或 adapter 文件夹名")
    p.add_argument("--medqa_subset", default="val", choices=["train", "val"],
                   help="data/medqa_split_seed42.json 里的训练集 / 验证集（默认验证集 500 题）")
    p.add_argument("--n", type=int, default=None, help="只跑前 N 题（调试用）")
    p.add_argument("--max_tokens", type=int, default=1024, help="和训练时的 max_completion_length 一致")
    p.add_argument("--n_samples", type=int, default=8, help="采样次数（0 = 只做贪心）；8 和训练时的 G 一致")
    p.add_argument("--temperature", type=float, default=1.0, help="采样温度，和训练时一致")
    p.add_argument("--max_lora_rank", type=int, default=16)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from data.medqa_split import load_split_indices
    from eval.evaluate import build_letter_token_table, sanitize

    model_name = sanitize(args.model_name or (
        os.path.basename(os.path.normpath(args.adapter)) if args.adapter else "base"))
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    items = load_items("medqa", n=args.n, num_options=4, split="train",
                       indices=load_split_indices(args.medqa_subset))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    prompts = [build_prompt_text(tokenizer, it.question, dict(zip(LETTERS, it.options))) for it in items]
    print("[prompt] ===== 第一题的完整输入（应和 GRPO 训练时一致）=====")
    print(prompts[0])
    print("[prompt] ===== 结束 =====")
    letter_ids = build_letter_token_table(tokenizer, list(LETTERS))

    llm = LLM(model=args.base_model, dtype="bfloat16", seed=args.seed, enable_lora=bool(args.adapter),
              max_lora_rank=args.max_lora_rank, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    lora = LoRARequest("adapter", 1, args.adapter) if args.adapter else None
    tag = f"medqa4_{args.medqa_subset}"
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)  # noqa: E731

    # ---- 1. 贪心生成 + "Answer:" 后字母 token 的概率 ----
    greedy = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=20,
                                                  seed=args.seed), lora_request=lora)
    rows, samples = [], []
    for it, out in zip(items, greedy):
        c = out.outputs[0]
        ids = list(c.token_ids)
        j = find_answer_token_index(ids, letter_ids, decode)
        if j is not None and c.logprobs:
            probs, mass = letter_probs_from_logprobs({k: v.logprob for k, v in c.logprobs[j].items()}, letter_ids)
        else:
            probs, mass = {L: 0.25 for L in LETTERS}, 0.0
        rows.append(make_row(it, model_name, "cot", probs, letter_mass=mass,
                             format_ok=int(extract_answer(c.text) is not None),
                             n_completion_tokens=len(ids), truncated=int(c.finish_reason == "length")))
        samples.append({"question_id": it.question_id, "gold": it.gold, "text": c.text})
    df = pd.DataFrame(rows)
    paths = [os.path.join(args.out_dir, f"preds_{model_name}_{tag}_cot.csv")]
    df.to_csv(paths[0], index=False)
    with open(os.path.join(args.out_dir, f"generations_{model_name}_{tag}_cot.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[cot] 贪心：准确率 {df['correct'].mean():.3f}，格式正确率 {df['format_ok'].mean():.3f}，"
          f"截断 {df['truncated'].mean():.3f}，平均长度 {df['n_completion_tokens'].mean():.0f} token，"
          f"平均 letter_mass {df['letter_mass'].mean():.3f}")

    # ---- 2. 采样 K 次：投票 confidence + 采样正确率（≈ 训练时的 reward）----
    if args.n_samples > 0:
        sampled = llm.generate(prompts, SamplingParams(n=args.n_samples, temperature=args.temperature,
                                                       max_tokens=args.max_tokens, seed=args.seed),
                               lora_request=lora)
        rows = []
        for it, out in zip(items, sampled):
            preds = [extract_answer(c.text) for c in out.outputs]
            rows.append(make_row(it, model_name, f"cot_vote{args.n_samples}", vote_probs(preds),
                                 sample_acc=sum(p == it.gold for p in preds) / len(preds),
                                 format_ok=sum(p is not None for p in preds) / len(preds)))
        dv = pd.DataFrame(rows)
        paths.append(os.path.join(args.out_dir, f"preds_{model_name}_{tag}_cot_vote{args.n_samples}.csv"))
        dv.to_csv(paths[-1], index=False)
        print(f"[cot] 采样 {args.n_samples} 次：采样正确率 {dv['sample_acc'].mean():.3f}（可和训练 reward 对比），"
              f"多数投票准确率 {dv['correct'].mean():.3f}")

    meta = {
        "model_name": model_name, "base_model": args.base_model, "adapter": args.adapter,
        "dataset": "medqa", "medqa_subset": args.medqa_subset, "n_items": len(items),
        "prompt_version": PROMPT_VERSION, "dtype": "bfloat16", "max_tokens": args.max_tokens,
        "n_samples": args.n_samples, "temperature": args.temperature, "seed": args.seed,
        "letter_token_ids": letter_ids, "python": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seconds": round(time.time() - t0, 1),
    }
    try:
        import torch, transformers, vllm  # noqa: E401
        meta.update(torch=torch.__version__, transformers=transformers.__version__, vllm=vllm.__version__,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        pass
    with open(paths[0].replace(".csv", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[done] 已保存：{paths}（用时 {meta['seconds']} 秒）")


if __name__ == "__main__":
    main()
