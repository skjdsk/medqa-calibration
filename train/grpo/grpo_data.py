"""
train/grpo/grpo_data.py -- load the MedQA training questions and turn each one into the prompt the model sees

[What this file does]
1. Downloads the **train** split of GBaker/MedQA-USMLE-4-options from Hugging Face (only train is allowed; the code checks).
2. Keeps only the 5000 **training** questions listed in data/medqa_split_seed42.json (the other 500 validation questions are never used for training),
      then takes the first N of them (N=0 means all 5000). The split file is fixed, so every training seed uses the same questions.
3. Puts each question into Qwen's chat format (chat template):
              system: you're a medical expert
              user:   question + options A-D + "reason step by step, then write Answer: X on its own line"
      then tokenizer.apply_chat_template turns it into one string (with the <|im_start|> markers etc.).
4. Each row of the dataset handed to GRPOTrainer has:
              prompt  -- the string with the chat template already applied (TRL won't apply it again to a string)
              answer  -- the gold letter A/B/C/D (TRL passes this column straight to the reward function)
              plus a few raw columns for checking (question / options_json / answer_text)

[Why the prompt is a pre-templated string rather than a list of messages]
That way the prompt we print is exactly what the model sees, which makes it easy to line up with the eval prompts in eval/.

[Preview on its own] (downloads the tokenizer, no GPU needed)
    python -m train.grpo.grpo_data --n 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# make "from data.xxx import ..." work with both python -m and running the file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.medqa_split import SPLIT_FILE, check_dataset_size, load_split  # noqa: E402

DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
ALLOWED_SPLIT = "train"  # training only ever uses the train split, never test
LETTERS = ("A", "B", "C", "D")

# ---------------------------------------------------------------------------
# Prompt template
# Heads up: compare this with the eval prompt in eval/. The GRPO training prompt should match the eval prompt,
#    otherwise training and eval formats differ and the SFT/DPO/GRPO comparison isn't fair.
#    If eval/ already has a prompt-building function, just import it instead of the version below.
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
    """Normalize options to {"A": "...", "B": "...", "C": "...", "D": "..."}.

    In GBaker/MedQA-USMLE-4-options, options is already a dict;
    this also accepts a list like [{"key": "A", "value": "..."}] in case the data format changes.
    """
    if isinstance(options, dict):
        opts = {str(k).strip().upper(): str(v).strip() for k, v in options.items()}
    elif isinstance(options, list):
        opts = {}
        for item in options:
            if isinstance(item, dict) and "key" in item and "value" in item:
                opts[str(item["key"]).strip().upper()] = str(item["value"]).strip()
            else:
                raise ValueError(f"Unrecognized options list item: {item!r}")
    else:
        raise ValueError(f"Unrecognized options type: {type(options)}")

    missing = [L for L in LETTERS if L not in opts]
    if missing:
        raise ValueError(f"options is missing {missing}, keys are {list(opts)}")
    return {L: opts[L] for L in LETTERS}


def format_user_message(question: str, options: dict[str, str]) -> str:
    """Question + options + instructions, as the text of the user message."""
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
    """Turn the messages into the model input string with Qwen's chat template.

    add_generation_prompt=True adds "<|im_start|>assistant\\n" at the end,
    which is where the model starts writing its answer.
    """
    return tokenizer.apply_chat_template(
        build_messages(question, options),
        tokenize=False,
        add_generation_prompt=True,
    )


def convert_example(raw: dict[str, Any], tokenizer) -> dict[str, Any]:
    """Convert one raw MedQA example into the format GRPO training needs."""
    options = normalize_options(raw["options"])
    answer = str(raw["answer_idx"]).strip().upper()
    if answer not in LETTERS:
        raise ValueError(f"answer_idx isn't A-D: {raw['answer_idx']!r}")

    # sanity check: the option text for answer_idx should match the answer field (if the data has one)
    answer_text = str(raw.get("answer", "")).strip()
    if answer_text and options[answer] != answer_text:
        print(
            f"[warning] the option text for answer_idx={answer} doesn't match the answer field:\n"
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
    """Load the MedQA train split, keep only the training questions from the split file (in split-file order), then take the first n.

    Returns all training questions (5000, used for the real run) when n is None, n<=0, or n>=the number of training questions.
    Validation questions never show up here.
    """
    from datasets import load_dataset

    if split != ALLOWED_SPLIT:
        raise ValueError(
            f"Training only allows the '{ALLOWED_SPLIT}' split, got split={split!r}. The test set must never be used for training."
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
        f"[data] {DATASET_NAME} split={split} has {len(full)} questions; taking the first {n} of the "
        f"{total} training questions in {Path(split_file).name} (the {len(split_info['val_indices'])} validation questions are left out)"
    )
    return subset


def build_grpo_dataset(tokenizer, n: int | None = 64, split_file: str | Path = SPLIT_FILE):
    """Return (raw subset, converted GRPO dataset)."""
    raw = load_medqa_train_subset(n=n, split_file=split_file)
    converted = raw.map(
        lambda ex: convert_example(ex, tokenizer),
        remove_columns=raw.column_names,  # keep only the columns we build, so nothing extra reaches the reward function
        desc="building prompts",
    )
    return raw, converted


def prompt_token_stats(tokenizer, dataset) -> dict[str, float]:
    """Prompt length stats in tokens (affects memory and speed: longer prompts mean longer sequences)."""
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in dataset["prompt"]]
    lengths.sort()
    return {
        "min": lengths[0],
        "mean": sum(lengths) / len(lengths),
        "p95": lengths[int(0.95 * (len(lengths) - 1))],
        "max": lengths[-1],
    }


def print_samples(raw, converted, tokenizer=None, index: int = 0) -> None:
    """Print one raw example and its converted version so the fields can be checked."""
    bar = "=" * 80
    print(bar)
    print(f"[raw example #{index}] (from {DATASET_NAME} train)")
    print(json.dumps(raw[index], ensure_ascii=False, indent=2, default=str))
    print(bar)
    ex = converted[index]
    print(f"[converted example #{index}]")
    print(f"answer (gold letter): {ex['answer']}    answer_text: {ex['answer_text']}")
    print("prompt (the full input the model sees, chat template markers included):")
    print("-" * 80)
    print(ex["prompt"])
    print("-" * 80)
    if tokenizer is not None:
        stats = prompt_token_stats(tokenizer, converted)
        print(
            f"prompt length in tokens: min={stats['min']} mean={stats['mean']:.0f} "
            f"p95={stats['p95']} max={stats['max']}"
        )
    print(bar)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the GRPO training data")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n", type=int, default=64, help="how many training questions to take from the start; 0 = all 5000")
    parser.add_argument("--index", type=int, default=0, help="which example to print")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    raw, converted = build_grpo_dataset(tokenizer, n=args.n)
    print_samples(raw, converted, tokenizer, index=args.index)
    counts = {L: converted["answer"].count(L) for L in LETTERS}
    print(f"answer distribution: {counts}")


if __name__ == "__main__":
    main()
