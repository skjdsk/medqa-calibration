"""Evaluate direct answers or generated reasoning with normalized option probabilities.

Direct mode preserves the original protocol. CoT mode uses the frozen DPO training
prompt, greedily generates reasoning, removes the final answer line, and rescores
all answer letters with the full vocabulary. Both modes export raw probabilities
and log probabilities for offline temperature scaling. See train/dpo/EVALUATION.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch

from eval.data_loaders import LETTERS, MCQItem, load_items

BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"
PROMPT_VERSION = "direct-v1"  # Increment the prompt version whenever the prompt changes to prevent mixing protocols.


# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Seed future sampled generation as well as the current forward-only evaluation.


# ---------------------------------------------------------------------------
# 1. Load an NF4 model and optional LoRA adapter
# ---------------------------------------------------------------------------
def pick_compute_dtype(name: str) -> torch.dtype:
    if name == "auto":
        # Use BF16 when supported; otherwise use FP16.
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_model_and_tokenizer(base_model: str, adapter: Optional[str], compute_dtype: torch.dtype,
                             double_quant: bool = True, quantization: str = "4bit"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Match the NF4, double-quantization and compute-dtype settings used for training.
    # Use quantization="none" to load directly in compute_dtype, for example for BF16 LoRA.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    ) if quantization == "4bit" else None
    # Prefer an adapter tokenizer when available; otherwise load the base tokenizer.
    tok_src = adapter if adapter and os.path.exists(os.path.join(adapter, "tokenizer_config.json")) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    tokenizer.padding_side = "left"  # Left-padding keeps the final position on a real token for decoder-only models.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Transformers 4.56 and later use dtype instead of torch_dtype.
    import transformers
    from packaging import version
    dtype_kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        **{dtype_kw: compute_dtype},
    )
    if adapter:
        from peft import PeftModel

        print(f"[model] Loading LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# 2. Prompt construction
# ---------------------------------------------------------------------------
def build_user_message(item: MCQItem) -> str:
    """Build the shared user message for comparable evaluation across models and datasets."""
    letters = item.letters
    option_lines = "\n".join(f"{L}. {opt}" for L, opt in zip(letters, item.options))
    letter_list = ", ".join(letters[:-1]) + f", or {letters[-1]}"
    parts = []
    if item.context:  # Prepend the abstract for PubMedQA.
        parts.append(f"Context:\n{item.context}")
    parts.append(f"Question: {item.question}")
    parts.append(f"Options:\n{option_lines}")
    parts.append(f"Answer with only the letter of the correct option ({letter_list}). Do not explain.")
    return "\n\n".join(parts)


def build_prompt_text(item: MCQItem, tokenizer, system_prompt: Optional[str] = None,
                      assistant_prefix: str = "") -> str:
    """Build the complete prompt with the chat template and an assistant generation prefix.
    The next token is the first answer token. assistant_prefix can append reasoning
    and an answer cue; direct evaluation uses an empty prefix. Without an explicit
    system_prompt, the tokenizer may supply its default system message. Keep system
    instructions consistent when comparing evaluations."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": build_user_message(item)})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + assistant_prefix


# ---------------------------------------------------------------------------
# 3. Map option letters and leading-space variants to single-token IDs
# ---------------------------------------------------------------------------
def build_letter_token_table(tokenizer, letters: list[str]) -> dict[str, list[int]]:
    """Collect single-token IDs for each letter and its leading-space variant.
    Ignore multi-token variants and unknown tokens, deduplicate IDs within letters,
    and reject IDs shared by different letters. Return a letter-to-token-ID mapping."""
    table: dict[str, list[int]] = {}
    for L in letters:
        ids = []
        for variant in (L, " " + L):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1 and toks[0] != tokenizer.unk_token_id:  # Exclude unknown tokens.
                ids.append(toks[0])
            else:
                print(f"[warn] {variant!r} does not form an eligible single token: {len(toks)} token(s), {toks}; skipping variant")
        ids = sorted(set(ids))
        if not ids:
            raise RuntimeError(f"Letter {L!r} has no single-token variant for scoring")
        table[L] = ids
    all_ids = [i for ids in table.values() for i in ids]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError(f"Token IDs overlap across letters; inspect the mapping: {table}")
    return table


# ---------------------------------------------------------------------------
# 4. Forward pass using only the final-position logits
# ---------------------------------------------------------------------------
def last_position_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return final-position logits with shape [batch, vocabulary].
    Derive position IDs from the attention mask so left-padding does not shift real
    token positions. Request only the final position when supported to avoid large
    sequence-by-vocabulary tensors. Older versions use num_logits_to_keep."""
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
    keep_arg = _logits_keep_arg(model)
    if keep_arg:
        kwargs[keep_arg] = 1
    out = model(**kwargs)
    return out.logits[:, -1, :]


def _logits_keep_arg(model) -> Optional[str]:
    """Inspect the underlying model forward signature for the supported logits-limit argument."""
    import inspect

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    params = inspect.signature(base.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None  # Older versions compute all positions, using more memory but the same scoring rule.


@torch.no_grad()
def score_letters(model, tokenizer, items: list[MCQItem], letter_table: dict[str, list[int]],
                  batch_size: int = 8, system_prompt: Optional[str] = None,
                  assistant_prefixes: Optional[list[str]] = None,
                  prompt_texts: Optional[list[str]] = None) -> list[dict]:
    """Score each option letter and return dictionaries in the original item order.
    Each result contains letter_probs, letter_log_probs, letter_mass and n_prompt_tokens.
    assistant_prefixes supplies optional per-question continuations. prompt_texts
    overrides the complete prompts, as used by the CoT scorer."""
    from tqdm.auto import tqdm

    device = next(model.parameters()).device
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if prompt_texts is not None and len(prompt_texts) != len(items):
        raise ValueError("Expected one scoring prompt per item")
    prompts = prompt_texts if prompt_texts is not None else [
        build_prompt_text(it, tokenizer, system_prompt, assistant_prefixes[i] if assistant_prefixes else "")
        for i, it in enumerate(items)
    ]
    # Sort longest first to reduce padding and surface memory limits early.
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(items)), key=lambda i: -lengths[i])
    results: list[Optional[dict]] = [None] * len(items)

    for start in tqdm(range(0, len(order), batch_size), desc="scoring"):
        idx = order[start:start + batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False)  # The chat template already includes special tokens.
        logits = last_position_logits(model, enc["input_ids"].to(device), enc["attention_mask"].to(device))
        logits = logits.float()  # Compute probabilities in FP32 for numerical stability.

        # Step 1: log probabilities across the full vocabulary.
        # log_probs_full[b, t] is log P(next token = t).
        log_probs_full = torch.log_softmax(logits, dim=-1)  # [B, V]

        for row, i in enumerate(idx):
            letters = items[i].letters  # Use only the options present in this question, including three-option PubMedQA.

            # Step 2: combine plain and leading-space letter variants.
            # logsumexp combines the probability mass of both token variants.
            letter_logp = torch.stack([
                torch.logsumexp(log_probs_full[row, letter_table[L]], dim=0) for L in letters
            ])  # [K], where K is the number of options.

            # Step 3: total letter probability mass before option normalization.
            # Mass near one indicates that the model intends to output an option letter.
            letter_mass = letter_logp.exp().sum().item()

            # Step 4: normalize probabilities across the available option letters.
            # The softmax divides each option probability by the sum over options.
            # This is equivalent to merging the corresponding raw logits before softmax.
            norm_probs = torch.softmax(letter_logp, dim=0).cpu().numpy()

            results[i] = {
                "letter_probs": {L: float(p) for L, p in zip(letters, norm_probs)},
                "letter_log_probs": {L: float(v) for L, v in zip(letters, torch.log_softmax(letter_logp, dim=0).cpu().tolist())},
                "letter_mass": float(letter_mass),
                "n_prompt_tokens": int(lengths[i]),
            }
    return results  # type: ignore[return-value]


def score_cot(*args, **kwargs):
    """Run the Transformers CoT scorer through the shared evaluation interface."""
    from eval.reasoning import score_cot as implementation
    return implementation(*args, **kwargs)


# ---------------------------------------------------------------------------
# 5. Build per-question prediction records
# ---------------------------------------------------------------------------
def build_rows(items: list[MCQItem], scores: list[dict], model_name: str, eval_mode: str,
               max_letters: int) -> pd.DataFrame:
    rows = []
    for it, sc in zip(items, scores):
        probs = sc["letter_probs"]
        pred = max(probs, key=probs.get)  # Select the maximum-probability letter; ties retain the first option.
        row = {
            "dataset": it.dataset,
            "question_id": it.question_id,
            "model_name": model_name,
            "eval_mode": eval_mode,
            "n_options": len(it.options),
            "gold": it.gold,
            "pred": pred,
            "confidence": probs[pred],
            "correct": int(pred == it.gold),
            "letter_mass": sc["letter_mass"],  # Total letter probability mass before option normalization.
            "n_prompt_tokens": sc["n_prompt_tokens"],
        }
        # Leave unavailable options empty, such as option D for PubMedQA.
        for L in LETTERS[:max_letters]:
            row[f"prob_{L}"] = probs.get(L, np.nan)
        if "letter_log_probs" in sc:
            for L, value in sc["letter_log_probs"].items():
                row[f"log_prob_{L}"] = value
        for key in ("reasoning", "generated_text", "generated_answer", "format_ok",
                    "generated_correct", "generation_truncated", "n_generated_tokens", "prompt_version"):
            if key in sc:
                row[key] = sc[key]
        rows.append(row)
    return pd.DataFrame(rows)


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


# ---------------------------------------------------------------------------
# 6. Command-line interface
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Multiple-choice calibration evaluation with direct or CoT letter scoring")
    p.add_argument("--dataset", required=True, nargs="+", choices=["medqa", "medmcqa", "pubmedqa"],
                   help="Accepts multiple datasets, for example --dataset medqa medmcqa pubmedqa")
    p.add_argument("--base_model", default=BASE_MODEL_DEFAULT)
    p.add_argument("--adapter", default=None, help="LoRA adapter directory; omit to evaluate the base model")
    p.add_argument("--model_name", default=None, help="Model label in CSV output; defaults to base or the adapter directory name")
    p.add_argument("--n", type=int, default=None, help="Evaluate only the first N questions for debugging")
    p.add_argument("--num_options", type=int, default=4, help="Number of MedQA options: 4 or 5")
    p.add_argument("--hf_name", default=None, help="Override the HF dataset name; valid with a single dataset only")
    p.add_argument("--hf_config", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--medqa_subset", default=None, choices=["train", "val"],
                   help="Evaluate the training/validation subset defined by data/medqa_split_seed42.json within MedQA train "
                   "(omit to evaluate the official test split)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--compute_dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                   help="4-bit compute dtype; match the training bnb_4bit_compute_dtype")
    p.add_argument("--no_double_quant", action="store_true", help="Disable double quantization if it was disabled during training")
    p.add_argument("--quantization", default="4bit", choices=["4bit", "none"],
                   help="Base-model loading: 4bit NF4 for QLoRA or none for unquantized LoRA")
    p.add_argument("--system_prompt", default=None, help="Defaults to the system message supplied by the chat template")
    p.add_argument("--mode", default="direct", choices=["direct", "cot"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--check_batch_consistency", action="store_true",
                   help="Rescore up to four questions with batch size 1 to check padding consistency")
    p.add_argument("--cot_max_new_tokens", type=int, default=512)
    p.add_argument("--cot_max_context_tokens", type=int, default=4096)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    if args.hf_name and len(args.dataset) > 1:
        raise SystemExit("--hf_name requires a single --dataset")
    if args.medqa_subset and (args.dataset != ["medqa"] or args.num_options != 4 or args.split or args.hf_name):
        raise SystemExit("--medqa_subset requires four-option MedQA without --split or --hf_name")

    model_name = args.model_name or (
        os.path.basename(os.path.normpath(args.adapter)) if args.adapter else "base")
    model_name = sanitize(model_name)
    os.makedirs(args.out_dir, exist_ok=True)

    compute_dtype = pick_compute_dtype(args.compute_dtype)
    print(f"[model] base={args.base_model} adapter={args.adapter} dtype={compute_dtype} name={model_name}")
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter, compute_dtype,
                                                double_quant=not args.no_double_quant,
                                                quantization=args.quantization)

    for ds_name in args.dataset:
        t0 = time.time()
        split, indices = args.split, None
        if args.medqa_subset:
            from data.medqa_split import load_split_indices
            split, indices = "train", load_split_indices(args.medqa_subset)
        items = load_items(ds_name, n=args.n, num_options=args.num_options if ds_name == "medqa" else None,
                           hf_name=args.hf_name, hf_config=args.hf_config, split=split, indices=indices)
        max_letters = max(len(it.options) for it in items)
        letter_table = build_letter_token_table(tokenizer, LETTERS[:max_letters])
        print(f"[tokens] Letter-to-token mapping: {letter_table}")
        print("[prompt] ===== Complete prompt for the first question; inspect formatting=====")
        print(build_prompt_text(items[0], tokenizer, args.system_prompt))
        print("[prompt] ===== End =====")

        if args.mode == "cot":
            scores = score_cot(model, tokenizer, items, letter_table, args.batch_size, args.system_prompt,
                               max_new_tokens=args.cot_max_new_tokens,
                               max_context_tokens=args.cot_max_context_tokens)
        else:
            scores = score_letters(model, tokenizer, items, letter_table, args.batch_size, args.system_prompt)

        if args.check_batch_consistency and args.mode == "direct":
            k = min(4, len(items))
            single = score_letters(model, tokenizer, items[:k], letter_table, 1, args.system_prompt)
            diff = max(abs(single[i]["letter_probs"][L] - scores[i]["letter_probs"][L])
                       for i in range(k) for L in single[i]["letter_probs"])
            print(f"[check] batch_size={args.batch_size} vs 1 maximum probability difference: {diff:.2e}"
                  f" (differences around 1e-3 or smaller may occur with 4-bit FP16)")

        df = build_rows(items, scores, model_name, args.mode, max_letters)
        tag = f"{ds_name}{args.num_options}" if ds_name == "medqa" else ds_name
        if args.medqa_subset:
            tag += f"_{args.medqa_subset}"
        csv_path = os.path.join(args.out_dir, f"preds_{model_name}_{tag}_{args.mode}.csv")
        df.to_csv(csv_path, index=False)

        # Record configuration and provenance for reproducible model comparisons.
        meta = {
            "model_name": model_name, "base_model": args.base_model, "adapter": args.adapter,
            "dataset": ds_name, "n_items": len(items), "num_options": args.num_options if ds_name == "medqa" else None,
            "hf_source": dict(hf_name=args.hf_name, hf_config=args.hf_config, split=split),
            "medqa_subset": args.medqa_subset,
            "mode": args.mode, "prompt_version": PROMPT_VERSION if args.mode == "direct" else "dpo-cot-v1",
            "cot_max_new_tokens": args.cot_max_new_tokens, "cot_max_context_tokens": args.cot_max_context_tokens, "system_prompt": args.system_prompt,
            "compute_dtype": str(compute_dtype), "quantization": args.quantization,
            "double_quant": not args.no_double_quant if args.quantization == "4bit" else None,
            "batch_size": args.batch_size, "seed": args.seed, "letter_token_ids": letter_table,
            "torch": torch.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seconds": round(time.time() - t0, 1),
        }
        try:
            import transformers, peft, bitsandbytes  # noqa: E401
            meta.update(transformers=transformers.__version__, peft=peft.__version__,
                        bitsandbytes=bitsandbytes.__version__)
        except Exception:
            pass
        with open(csv_path.replace(".csv", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[done] {ds_name}: {len(df)} questions, accuracy {df['correct'].mean():.3f} , "
              f"mean letter_mass {df['letter_mass'].mean():.3f}(minimum {df['letter_mass'].min():.3f})")
        if df["letter_mass"].mean() < 0.5:
            print("[warn] mean letter_mass < 0.5: the model often prefers non-letter tokens; "
                  "inspect the prompt/chat template before interpreting letter probabilities")
        print(f"[done] Predictions saved: {csv_path} (elapsed {meta['seconds']} seconds)")


if __name__ == "__main__":
    main()
