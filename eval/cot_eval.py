"""
cot_eval.py -- evaluate a model with the CoT prompt used in GRPO training (vLLM generation); writes the same per-question CSV as evaluate.py
=====================================================================================================

[Why this file exists]
evaluate.py is "direct answer": it asks for just the letter and looks at the A-D probabilities on the first token.
In GRPO training the model "reasons first, then writes Answer: X" (the prompt in train/grpo/grpo_data.py).
Direct answer alone might miss what GRPO learned, so this evaluates again with **exactly the same training prompt**.

[What it does]
1. Loads the base model in vLLM (bf16, no quantization), plus the LoRA if --adapter is given.
2. Greedy generation (temperature 0):
          - takes the last "Answer: X" using the same rule as the training reward (extract_answer in train/grpo/rewards.py)
          - at the letter token right after "Answer:", reads the top-20 logprobs from vLLM,
              takes the probabilities of A-D ("A" and " A" merged), normalized over the 4 letters -> prob_A..prob_D
              that's the confidence under CoT (eval_mode = "cot")
          - questions where no answer in the right format comes out: 0.25 for each of the 4 options (basically "no answer"), format_ok = 0
3. Samples --n_samples times (temperature 1.0, same as training):
          - share of samples that picked each option -> prob_A..prob_D (eval_mode = "cot_vote{K}", vote-based confidence)
          - sample_acc = average accuracy of the sampled answers, directly comparable to the training reward
4. Writes preds_{model}_{tag}_{mode}.csv, which eval/metrics.py can use directly for ECE / Brier / NLL etc.
      tag: MedQA validation = medqa4_val, MedQA test = medqa4, MedMCQA = medmcqa, PubMedQA = pubmedqa
      (same naming as evaluate.py, so direct and cot results line up by question_id)

[Datasets]
    --dataset medqa --medqa_subset val    the 500 validation questions split off the MedQA train split (default)
    --dataset medqa --medqa_subset test   the official MedQA test split (1273 questions, for the final report)
    --dataset medmcqa                     MedMCQA validation (about 4183 questions, transfer eval)
    --dataset pubmedqa                    PubMedQA expert-labeled subset (1000 questions, yes/no/maybe -> A/B/C, transfer eval)
    MedQA / MedMCQA (4 options, no context) use a prompt that's **word for word** the same as GRPO training;
    PubMedQA has only 3 options plus an abstract: "Context:" + the abstract goes before the question and the last sentence becomes "X is one of A, B, or C";
    the system message and everything else stay the same (see build_cot_prompt).

[Example] (A100; about 5-10 minutes per model; run the two models separately, since swapping models in one vLLM process is a pain)
    python -m eval.cot_eval --model_name base --out_dir results/val
    python -m eval.cot_eval --adapter /path/to/adapter --model_name grpo_seed1 --out_dir results/val
    python -m eval.metrics --pred_csv "results/val/preds_*.csv" --out_dir results/val/metrics
    python -m eval.cot_eval --dataset medqa --medqa_subset test --model_name base --out_dir results/test
    python -m eval.cot_eval --dataset pubmedqa --model_name base --out_dir results/test
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
from train.grpo.grpo_data import INSTRUCTION, LETTERS, SYSTEM_PROMPT, build_prompt_text
from train.grpo.rewards import extract_answer

BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"
PROMPT_VERSION = "grpo-cot-v1"  # = SYSTEM_PROMPT + INSTRUCTION from train/grpo/grpo_data.py
DATASETS = ("medqa", "medmcqa", "pubmedqa")
_TRAIN_LETTER_LIST = "A, B, C, or D"  # the letter list in the last sentence of INSTRUCTION

# the text right before the letter token should look like "Answer:" / "**Answer:** " / "Answer: (", matching the first half of rewards._ANSWER_RE
_ANSWER_PREFIX_RE = re.compile(r"\b(?i:answer)\s*[*_]*\s*[:：]\s*[*_]*\s*\(?\s*$")


# ---------------------------------------------------------------------------
# small pure-Python helpers (no vLLM needed; unit tests are in test_cot_eval.py)
# ---------------------------------------------------------------------------
def find_answer_token_index(token_ids: list[int], letter_ids: dict[str, list[int]], decode) -> int | None:
    """Search backwards for the last letter token that comes right after Answer:; return None if there isn't one.

    decode(ids) -> str, i.e. tokenizer.decode. Only decodes the few tokens before the letter, so it's fast.
    """
    id_to_letter = {i: L for L, ids in letter_ids.items() for i in ids}
    for j in range(len(token_ids) - 1, -1, -1):
        if token_ids[j] not in id_to_letter:
            continue
        before = decode(token_ids[max(0, j - 8):j])
        # a token with a leading space like " A": the space is part of the letter token, so the text before just has to end with "Answer:"
        if _ANSWER_PREFIX_RE.search(before):
            return j
    return None


def letter_probs_from_logprobs(logprobs: dict[int, float], letter_ids: dict[str, list[int]]) -> tuple[dict, float]:
    """{token_id: logprob} (top-k) -> probabilities normalized over the 4 letters, plus letter_mass (total letter probability before normalizing).

    Letters outside the top-k count as probability 0 (anything outside the top 20 is tiny anyway).
    """
    mass = {L: sum(math.exp(logprobs[i]) for i in ids if i in logprobs) for L, ids in letter_ids.items()}
    total = sum(mass.values())
    if total <= 0:
        return {L: 1.0 / len(letter_ids) for L in letter_ids}, 0.0
    return {L: m / total for L, m in mass.items()}, total


def vote_probs(preds: list[str | None], letters=LETTERS) -> dict[str, float]:
    """K sampled answers -> vote share for each letter. Answers in the wrong format don't get a vote; if none answered, it's uniform."""
    valid = [p for p in preds if p in letters]
    if not valid:
        return {L: 1.0 / len(letters) for L in letters}
    return {L: valid.count(L) / len(valid) for L in letters}


def item_letters(item) -> tuple[str, ...]:
    return tuple(ALL_LETTERS[:len(item.options)])


def format_cot_user_message(item) -> str:
    """Same format as grpo_data.format_user_message, generalized to 3 options / questions with context.

    With 4 options and no context it's exactly the same as training (there's a test for that in test_cot_eval.py).
    """
    letters = item_letters(item)
    if not 2 <= len(letters) <= len(LETTERS):
        raise ValueError(f"{item.question_id} has {len(letters)} options; extract_answer only handles A-D")
    lines = []
    if item.context:
        lines += ["Context:", item.context.strip(), ""]
    lines += [item.question.strip(), ""]
    lines += [f"{L}. {opt}" for L, opt in zip(letters, item.options)]
    letter_list = ", ".join(letters[:-1]) + f", or {letters[-1]}"
    lines += ["", INSTRUCTION.replace(_TRAIN_LETTER_LIST, letter_list)]
    return "\n".join(lines)


def build_cot_prompt(tokenizer, item) -> str:
    """4 options, no context: call the training build_prompt_text directly (word for word the same); otherwise use the generalized format."""
    if item_letters(item) == LETTERS and not item.context:
        return build_prompt_text(tokenizer, item.question, dict(zip(LETTERS, item.options)))
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_cot_user_message(item)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def valid_answer(text: str, letters) -> str | None:
    """The training extract_answer, minus letters this question doesn't have (e.g. PubMedQA writing Answer: D)."""
    a = extract_answer(text)
    return a if a in letters else None


def result_tag(dataset: str, medqa_subset: str) -> str:
    """Dataset tag for the output file name, same as evaluate.py."""
    if dataset != "medqa":
        return dataset
    return "medqa4" if medqa_subset == "test" else f"medqa4_{medqa_subset}"


def make_row(item, model_name: str, eval_mode: str, probs: dict[str, float], **extra) -> dict:
    """Same columns as evaluate.build_rows, so metrics.py can read it directly."""
    pred = max(probs, key=probs.get)  # ties go to the earlier letter
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
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CoT eval (GRPO training prompt, vLLM generation)")
    p.add_argument("--base_model", default=BASE_MODEL_DEFAULT)
    p.add_argument("--adapter", default=None, help="LoRA adapter folder; leave out to evaluate the original model")
    p.add_argument("--model_name", default=None, help="model name written to the CSV; defaults to base or the adapter folder name")
    p.add_argument("--dataset", default="medqa", choices=list(DATASETS))
    p.add_argument("--medqa_subset", default="val", choices=["train", "val", "test"],
                   help="MedQA only: train / val = the training / validation sets from data/medqa_split_seed42.json "
                   "(default: the 500 validation questions); test = the official test split (1273 questions)")
    p.add_argument("--n", type=int, default=None, help="only run the first N questions (for debugging)")
    p.add_argument("--max_tokens", type=int, default=1024, help="same as max_completion_length in training")
    p.add_argument("--n_samples", type=int, default=8, help="number of samples (0 = greedy only); 8 matches G in training")
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature, same as training")
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

    if args.dataset == "medqa" and args.medqa_subset in ("train", "val"):
        items = load_items("medqa", n=args.n, num_options=4, split="train",
                           indices=load_split_indices(args.medqa_subset))
    elif args.dataset == "medqa":
        items = load_items("medqa", n=args.n, num_options=4, split="test")
    else:
        if args.medqa_subset != "val":
            print(f"[note] --medqa_subset only applies to MedQA; ignored for {args.dataset}")
        items = load_items(args.dataset, n=args.n)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    prompts = [build_cot_prompt(tokenizer, it) for it in items]
    print("[prompt] ===== full input for the first question (MedQA/MedMCQA should match GRPO training) =====")
    print(prompts[0])
    print("[prompt] ===== end =====")
    # one letter -> token id table per option count (PubMedQA only has A-C)
    letter_tables = {lets: build_letter_token_table(tokenizer, list(lets)) for lets in {item_letters(it) for it in items}}

    # prompt + generation length can't exceed max_model_len (PubMedQA prompts are longer because of the abstract); bump it automatically if needed, it only affects memory allocation
    longest = max(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts)
    max_model_len = max(args.max_model_len, longest + args.max_tokens)
    print(f"[prompt] longest prompt {longest} tokens; max_model_len = {max_model_len}")

    llm = LLM(model=args.base_model, dtype="bfloat16", seed=args.seed, enable_lora=bool(args.adapter),
              max_lora_rank=args.max_lora_rank, max_model_len=max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    lora = LoRARequest("adapter", 1, args.adapter) if args.adapter else None
    tag = result_tag(args.dataset, args.medqa_subset)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)  # noqa: E731

    # ---- 1. greedy generation + probability of the letter token after "Answer:" ----
    greedy = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=20,
                                                  seed=args.seed), lora_request=lora)
    rows, samples = [], []
    for it, out in zip(items, greedy):
        c = out.outputs[0]
        ids = list(c.token_ids)
        lets = item_letters(it)
        letter_ids = letter_tables[lets]
        j = find_answer_token_index(ids, letter_ids, decode)
        if j is not None and c.logprobs:
            probs, mass = letter_probs_from_logprobs({k: v.logprob for k, v in c.logprobs[j].items()}, letter_ids)
        else:
            probs, mass = {L: 1.0 / len(lets) for L in lets}, 0.0
        rows.append(make_row(it, model_name, "cot", probs, letter_mass=mass,
                             format_ok=int(valid_answer(c.text, lets) is not None),
                             n_completion_tokens=len(ids), truncated=int(c.finish_reason == "length")))
        samples.append({"question_id": it.question_id, "gold": it.gold, "text": c.text})
    df = pd.DataFrame(rows)
    paths = [os.path.join(args.out_dir, f"preds_{model_name}_{tag}_cot.csv")]
    df.to_csv(paths[0], index=False)
    with open(os.path.join(args.out_dir, f"generations_{model_name}_{tag}_cot.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[cot] greedy: accuracy {df['correct'].mean():.3f}, format rate {df['format_ok'].mean():.3f}, "
          f"truncated {df['truncated'].mean():.3f}, mean length {df['n_completion_tokens'].mean():.0f} tokens, "
          f"mean letter_mass {df['letter_mass'].mean():.3f}")

    # ---- 2. sample K times: vote confidence + sample accuracy (~ the training reward) ----
    if args.n_samples > 0:
        sampled = llm.generate(prompts, SamplingParams(n=args.n_samples, temperature=args.temperature,
                                                       max_tokens=args.max_tokens, seed=args.seed),
                               lora_request=lora)
        rows = []
        for it, out in zip(items, sampled):
            lets = item_letters(it)
            preds = [valid_answer(c.text, lets) for c in out.outputs]
            rows.append(make_row(it, model_name, f"cot_vote{args.n_samples}", vote_probs(preds, lets),
                                 sample_acc=sum(p == it.gold for p in preds) / len(preds),
                                 format_ok=sum(p is not None for p in preds) / len(preds)))
        dv = pd.DataFrame(rows)
        paths.append(os.path.join(args.out_dir, f"preds_{model_name}_{tag}_cot_vote{args.n_samples}.csv"))
        dv.to_csv(paths[-1], index=False)
        print(f"[cot] {args.n_samples} samples: sample accuracy {dv['sample_acc'].mean():.3f} (compare with the training reward), "
              f"majority-vote accuracy {dv['correct'].mean():.3f}")

    meta = {
        "model_name": model_name, "base_model": args.base_model, "adapter": args.adapter,
        "dataset": args.dataset, "medqa_subset": args.medqa_subset if args.dataset == "medqa" else None,
        "tag": tag, "n_items": len(items), "max_model_len": max_model_len,
        "prompt_version": PROMPT_VERSION, "dtype": "bfloat16", "max_tokens": args.max_tokens,
        "n_samples": args.n_samples, "temperature": args.temperature, "seed": args.seed,
        "letter_token_ids": {"".join(k): v for k, v in letter_tables.items()}, "python": platform.python_version(),
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
    print(f"[done] saved: {paths} (took {meta['seconds']} s)")


if __name__ == "__main__":
    main()
