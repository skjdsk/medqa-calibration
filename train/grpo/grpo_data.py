"""
train/grpo/grpo_data.py —— 加载 MedQA 训练集，并把每道题变成模型要看的 prompt

【这个文件做什么（简单版）】
1. 从 Hugging Face 下载 GBaker/MedQA-USMLE-4-options 的 **train** split（只允许 train，代码里写死了检查）。
2. 只取 data/medqa_split_seed42.json 里的 5000 道**训练题**（另外 500 道验证题绝对不用于训练），
   再取其中前 N 题（N=0 表示全部 5000 题）。划分文件是固定的，所以不管训练 seed 是多少，用的都是同一批题。
3. 把每道题拼成 Qwen 的对话格式（chat template）：
       system: 你是医学专家
       user:   题目 + 选项 A–D + "先一步步推理，最后单独一行写 Answer: X"
   然后用 tokenizer.apply_chat_template 把它变成一整段字符串（包含 <|im_start|> 等特殊标记）。
4. 最终给 GRPOTrainer 的数据集每行有：
       prompt  —— 已经套好 chat template 的字符串（TRL 看到字符串就不会再套一次模板）
       answer  —— 标准答案字母 A/B/C/D（TRL 会把这一列原样传给打分函数）
       以及几列方便核对的原始信息（question / options_json / answer_text）

【为什么 prompt 用"已套模板的字符串"而不是消息列表】
这样我们打印出来的 prompt 和模型真正看到的输入一字不差，方便和 eval/ 里的评估 prompt 对齐。

【单独运行预览】（会下载 tokenizer，不需要 GPU）
    python -m train.grpo.grpo_data --n 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 让 "from data.xxx import ..." 不管用 python -m 还是直接运行都能工作
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.medqa_split import SPLIT_FILE, check_dataset_size, load_split  # noqa: E402

DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
ALLOWED_SPLIT = "train"  # 训练只能用 train split，绝对不能用 test
LETTERS = ("A", "B", "C", "D")

# ---------------------------------------------------------------------------
# Prompt 模板
# ⚠️ 请和 eval/ 里评估用的 prompt 对比一下：GRPO 训练时的 prompt 最好和评估时完全一致，
#    否则"训练格式"和"评估格式"不匹配，会让 SFT/DPO/GRPO 的比较不公平。
#    如果 eval/ 里已经有构造 prompt 的函数，建议直接 import 那个函数替换下面的实现。
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a medical expert. Answer the following USMLE-style multiple-choice question."
)

INSTRUCTION = (
    "Think through the problem step by step. "
    "Then, on the final line, write your answer in exactly this format:\n"
    "Answer: X\n"
    "where X is one of A, B, C, or D."
)


def normalize_options(options: Any) -> dict[str, str]:
    """把 options 统一成 {"A": "...", "B": "...", "C": "...", "D": "..."}。

    GBaker/MedQA-USMLE-4-options 里 options 本来就是字典；
    这里顺便兼容 [{"key": "A", "value": "..."}] 这种列表格式，以防数据版本变化。
    """
    if isinstance(options, dict):
        opts = {str(k).strip().upper(): str(v).strip() for k, v in options.items()}
    elif isinstance(options, list):
        opts = {}
        for item in options:
            if isinstance(item, dict) and "key" in item and "value" in item:
                opts[str(item["key"]).strip().upper()] = str(item["value"]).strip()
            else:
                raise ValueError(f"无法识别的 options 列表元素：{item!r}")
    else:
        raise ValueError(f"无法识别的 options 类型：{type(options)}")

    missing = [L for L in LETTERS if L not in opts]
    if missing:
        raise ValueError(f"options 缺少选项 {missing}，实际 keys={list(opts)}")
    return {L: opts[L] for L in LETTERS}


def format_user_message(question: str, options: dict[str, str]) -> str:
    """题目 + 选项 + 作答要求，拼成 user 消息的正文。"""
    lines = [question.strip(), ""]
    for L in LETTERS:
        lines.append(f"{L}. {options[L]}")
    lines += ["", INSTRUCTION]
    return "\n".join(lines)


def build_messages(question: str, options: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_user_message(question, options)},
    ]


def build_prompt_text(tokenizer, question: str, options: dict[str, str]) -> str:
    """用 Qwen 的 chat template 把消息变成模型输入字符串。

    add_generation_prompt=True 会在最后加上 "<|im_start|>assistant\\n"，
    模型从这里开始续写，也就是开始"答题"。
    """
    return tokenizer.apply_chat_template(
        build_messages(question, options),
        tokenize=False,
        add_generation_prompt=True,
    )


def convert_example(raw: dict[str, Any], tokenizer) -> dict[str, Any]:
    """把一条原始 MedQA 样例转换成 GRPO 训练需要的格式。"""
    options = normalize_options(raw["options"])
    answer = str(raw["answer_idx"]).strip().upper()
    if answer not in LETTERS:
        raise ValueError(f"answer_idx 不是 A–D：{raw['answer_idx']!r}")

    # 核对：answer_idx 对应的选项文字应该等于 answer 字段（如果数据里有 answer 字段）
    answer_text = str(raw.get("answer", "")).strip()
    if answer_text and options[answer] != answer_text:
        print(
            f"[警告] answer_idx={answer} 对应的选项文字与 answer 字段不一致：\n"
            f"  options[{answer}] = {options[answer]!r}\n  answer = {answer_text!r}"
        )

    return {
        "prompt": build_prompt_text(tokenizer, raw["question"], options),
        "answer": answer,
        "question": raw["question"],
        "options_json": json.dumps(options, ensure_ascii=False),
        "answer_text": options[answer],
    }


def load_medqa_train_subset(n: int | None = 64, split_file: str | Path = SPLIT_FILE, split: str = ALLOWED_SPLIT):
    """加载 MedQA train split，只保留划分文件里的训练题（按划分文件里的顺序），再取前 n 题。

    n=None 或 n<=0 或 n>=训练题数 时返回全部训练题（5000 题，正式训练时用）。
    验证集的题不会出现在这里。
    """
    from datasets import load_dataset

    if split != ALLOWED_SPLIT:
        raise ValueError(
            f"只允许使用 '{ALLOWED_SPLIT}' split 训练，收到 split={split!r}。测试集绝对不能用于训练。"
        )

    full = load_dataset(DATASET_NAME, split=split)
    split_info = load_split(split_file)
    check_dataset_size(len(full), split_info)
    train_idx = split_info["train_indices"]
    total = len(train_idx)
    if n is None or n <= 0 or n >= total:
        n = total
    subset = full.select(train_idx[:n])
    print(
        f"[数据] {DATASET_NAME} split={split} 共 {len(full)} 题；按 {Path(split_file).name} 取训练集 "
        f"{total} 题中的前 {n} 题（验证集 {len(split_info['val_indices'])} 题不参与训练）"
    )
    return subset


def build_grpo_dataset(tokenizer, n: int | None = 64, split_file: str | Path = SPLIT_FILE):
    """返回 (原始子集, 转换后的 GRPO 数据集)。"""
    raw = load_medqa_train_subset(n=n, split_file=split_file)
    converted = raw.map(
        lambda ex: convert_example(ex, tokenizer),
        remove_columns=raw.column_names,  # 只保留我们构造的列，避免多余字段传进打分函数
        desc="构造 prompt",
    )
    return raw, converted


def prompt_token_stats(tokenizer, dataset) -> dict[str, float]:
    """统计 prompt 的 token 长度（影响显存和速度：prompt 越长，每条序列越长）。"""
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in dataset["prompt"]]
    lengths.sort()
    return {
        "min": lengths[0],
        "mean": sum(lengths) / len(lengths),
        "p95": lengths[int(0.95 * (len(lengths) - 1))],
        "max": lengths[-1],
    }


def print_samples(raw, converted, tokenizer=None, index: int = 0) -> None:
    """打印一条原始样例和转换后的样例，方便核对字段。"""
    bar = "=" * 80
    print(bar)
    print(f"[原始样例 #{index}]（来自 {DATASET_NAME} train）")
    print(json.dumps(raw[index], ensure_ascii=False, indent=2, default=str))
    print(bar)
    ex = converted[index]
    print(f"[转换后样例 #{index}]")
    print(f"answer（标准答案字母）: {ex['answer']}    answer_text: {ex['answer_text']}")
    print("prompt（模型实际看到的完整输入，含 chat template 特殊标记）:")
    print("-" * 80)
    print(ex["prompt"])
    print("-" * 80)
    if tokenizer is not None:
        stats = prompt_token_stats(tokenizer, converted)
        print(
            f"prompt token 长度：min={stats['min']} mean={stats['mean']:.0f} "
            f"p95={stats['p95']} max={stats['max']}"
        )
    print(bar)


def main() -> None:
    parser = argparse.ArgumentParser(description="预览 GRPO 训练数据")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n", type=int, default=64, help="取训练集前几题；0 = 全部 5000 题")
    parser.add_argument("--index", type=int, default=0, help="打印第几条样例")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    raw, converted = build_grpo_dataset(tokenizer, n=args.n)
    print_samples(raw, converted, tokenizer, index=args.index)
    counts = {L: converted["answer"].count(L) for L in LETTERS}
    print(f"答案分布：{counts}")


if __name__ == "__main__":
    main()
