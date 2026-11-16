"""
data_loaders.py -- load the three eval datasets and put them in the same format
=========================================================

[What this file does]
MedQA / MedMCQA / PubMedQA on Hugging Face all use different field names, option formats and answer formats.
This file turns all of them into the same structure, MCQItem:
        - dataset      dataset name, e.g. "medqa"
        - question_id  a unique id per question
        - question     the question text
        - options      list of option texts, e.g. ["Aspirin", "Heparin", ...] (index 0 is A)
        - gold         the correct letter, e.g. "C"
        - context      extra material (PubMedQA only: the paper abstract)
That way evaluate.py doesn't have to care how the datasets differ.

[Important: double-check the dataset names and field names]
The Hugging Face names / field names in DATASET_DEFAULTS below are defaults taken from the dataset cards,
but other mirrors or versions can differ. Every load prints one "raw example" and one "converted example";
compare them the first time you run it.
To use a different version, override with --hf_name / --hf_config / --split on the command line; no code changes needed.

Run this file on its own to just look at examples without loading a model:
    python -m eval.data_loaders --dataset medqa --n 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

LETTERS = list("ABCDEFGHIJ")  # up to 10 options

PUBMEDQA_LABELS = ["yes", "no", "maybe"]  # A / B / C


@dataclass
class MCQItem:
    """One multiple-choice question in the shared format."""

    dataset: str
    question_id: str
    question: str
    options: list[str]
    gold: str  # correct letter
    context: Optional[str] = None  # PubMedQA abstract; None for the other datasets
    meta: dict = field(default_factory=dict)  # other info (subject etc.), not used in eval

    @property
    def letters(self) -> list[str]:
        return LETTERS[: len(self.options)]


# ---------------------------------------------------------------------------
# default dataset sources (double-check these, see the note at the top)
# ---------------------------------------------------------------------------
DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    # MedQA (USMLE, English). The 4-option and 5-option versions live in different HF repos; picked by num_options.
    "medqa": {
        4: dict(hf_name="GBaker/MedQA-USMLE-4-options", hf_config=None, split="test"),
        5: dict(hf_name="lavita/medical-qa-datasets", hf_config="med-qa-en-5options-source", split="test"),
    },
    # MedMCQA: the test set has no public answers, so we use validation (about 4183 questions)
    "medmcqa": dict(hf_name="openlifescienceai/medmcqa", hf_config=None, split="validation"),
    # PubMedQA expert-labeled subset: the pqa_labeled config only has a train split, 1000 questions
    "pubmedqa": dict(hf_name="qiaojin/PubMedQA", hf_config="pqa_labeled", split="train"),
}


def resolve_source(dataset: str, num_options: Optional[int] = None,
                   hf_name: Optional[str] = None, hf_config: Optional[str] = None,
                   split: Optional[str] = None) -> dict[str, Any]:
    """Work out where to actually load from: defaults + command-line overrides."""
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown dataset {dataset!r}, options: {list(DATASET_DEFAULTS)}")
    default = DATASET_DEFAULTS[dataset]
    if dataset == "medqa":
        n = num_options or 4
        if n not in default and hf_name is None:
            raise ValueError(f"No default MedQA source with {n} options; pass one with --hf_name")
        default = default.get(n, dict(hf_name=None, hf_config=None, split="test"))
    src = dict(default)
    if hf_name is not None:
        src["hf_name"] = hf_name
        src["hf_config"] = hf_config  # config follows when you switch repos (can be None)
    elif hf_config is not None:
        src["hf_config"] = hf_config
    if split is not None:
        src["split"] = split
    return src


# ---------------------------------------------------------------------------
# helper: turn all the odd option formats into [(letter, text), ...]
# ---------------------------------------------------------------------------
def parse_options(raw: Any) -> list[tuple[str, str]]:
    """
    Handles a few common formats:
      1) dict:   {"A": "xxx", "B": "yyy"}                  （GBaker/MedQA）
      2) list of dict: [{"key": "A", "value": "xxx"}, ...] (lavita / bigbio style)
      3) list of str:  ["xxx", "yyy"]                       (numbered A, B, C... in order)
    """
    if isinstance(raw, dict):
        # some HF versions store the dict "column-wise" as {"key": [...], "value": [...]}
        if set(raw.keys()) >= {"key", "value"} and isinstance(raw["key"], list):
            pairs = list(zip(raw["key"], raw["value"]))
        else:
            pairs = list(raw.items())
        pairs = [(str(k).strip().upper(), str(v)) for k, v in pairs if v is not None]
        pairs.sort(key=lambda kv: kv[0])
        return pairs
    if isinstance(raw, list):
        if all(isinstance(x, dict) for x in raw):
            return [(str(x["key"]).strip().upper(), str(x["value"])) for x in raw]
        return [(LETTERS[i], str(x)) for i, x in enumerate(raw)]
    raise ValueError(f"Unrecognized option format: {type(raw)} -> {str(raw)[:200]}")


def _truncate(v: Any, n: int = 300) -> Any:
    """Cut long strings short when printing examples so they don't flood the output."""
    if isinstance(v, str) and len(v) > n:
        return v[:n] + f"...<{len(v)} chars total>"
    if isinstance(v, dict):
        return {k: _truncate(x, n) for k, x in v.items()}
    if isinstance(v, list):
        return [_truncate(x, n) for x in v[:10]] + (["..."] if len(v) > 10 else [])
    return v


# ---------------------------------------------------------------------------
# per-dataset conversion: raw row -> MCQItem
# ---------------------------------------------------------------------------
def convert_medqa(row: dict, idx: int, num_options: int) -> MCQItem:
    pairs = parse_options(row["options"])
    if len(pairs) != num_options:
        # don't silently truncate: that could drop the right answer, so raise and make you switch to the matching dataset version
        raise ValueError(
            f"MedQA question {idx} has {len(pairs)} options, but --num_options={num_options}. "
            f"Switch to the dataset version with that many options (--hf_name), or change --num_options."
        )
    keys = [k for k, _ in pairs]
    texts = [v for _, v in pairs]

    # gold answer: use answer_idx (a letter) if it's there, otherwise match the answer text against the options
    gold_pos = None
    if row.get("answer_idx") not in (None, ""):
        key = str(row["answer_idx"]).strip().upper()
        if key in keys:
            gold_pos = keys.index(key)
    if gold_pos is None and row.get("answer") not in (None, ""):
        ans = str(row["answer"]).strip()
        matches = [i for i, t in enumerate(texts) if t.strip() == ans]
        if len(matches) == 1:
            gold_pos = matches[0]
    if gold_pos is None:
        raise ValueError(f"Can't find the gold answer for MedQA question {idx}: answer_idx={row.get('answer_idx')!r}, "
                         f"answer={row.get('answer')!r}, option keys={keys}")

    # MedQA has no id field, so use the row number in the split as the id (stable for a given data version, so it lines up)
    return MCQItem(
        dataset="medqa",
        question_id=f"medqa{num_options}-{idx}",
        question=str(row["question"]).strip(),
        options=texts,
        gold=LETTERS[gold_pos],  # renumber by position as A, B, C... so the letters are contiguous
        meta={"meta_info": row.get("meta_info")},
    )


def convert_medmcqa(row: dict, idx: int, **_) -> MCQItem:
    options = [str(row[k]) for k in ("opa", "opb", "opc", "opd")]
    cop = row["cop"]
    if isinstance(cop, str):
        cop = cop.strip().lower()
        cop = "abcd".index(cop) if cop in "abcd" and len(cop) == 1 else int(cop)
    cop = int(cop)
    if not 0 <= cop <= 3:
        raise ValueError(f"MedMCQA question {idx} has cop={cop}, which isn't between 0 and 3. "
                         f"If your version numbers them 1-4 (like the raw JSON), subtract 1.")
    return MCQItem(
        dataset="medmcqa",
        question_id=str(row.get("id", f"medmcqa-{idx}")),
        question=str(row["question"]).strip(),
        options=options,
        gold=LETTERS[cop],
        meta={"subject_name": row.get("subject_name"), "choice_type": row.get("choice_type")},
    )


def format_pubmedqa_context(ctx: Any) -> str:
    """
    PubMedQA's context is usually:
        {"contexts": [paragraph 1, paragraph 2, ...], "labels": ["BACKGROUND", "METHODS", ...], ...}
    Also handles versions where it's a list of dicts, or just a string.
    Output: one paragraph per line, prefixed with its label, e.g. "BACKGROUND: ...".
    """
    if isinstance(ctx, str):
        return ctx.strip()
    if isinstance(ctx, dict):
        paras = ctx.get("contexts") or []
        labels = ctx.get("labels") or []
    elif isinstance(ctx, list):
        paras = [c.get("contexts", "") for c in ctx]
        labels = [c.get("labels", "") for c in ctx]
    else:
        raise ValueError(f"Unrecognized PubMedQA context format: {type(ctx)}")
    lines = []
    for i, p in enumerate(paras):
        lab = labels[i] if i < len(labels) and labels[i] else ""
        lines.append(f"{lab}: {p.strip()}" if lab else p.strip())
    return "\n".join(lines)


def convert_pubmedqa(row: dict, idx: int, **_) -> MCQItem:
    decision = str(row["final_decision"]).strip().lower()
    if decision not in PUBMEDQA_LABELS:
        raise ValueError(f"PubMedQA question {idx} has final_decision={decision!r}, not yes/no/maybe")
    return MCQItem(
        dataset="pubmedqa",
        question_id=str(row.get("pubid", f"pubmedqa-{idx}")),
        question=str(row["question"]).strip(),
        options=list(PUBMEDQA_LABELS),  # A. yes / B. no / C. maybe
        gold=LETTERS[PUBMEDQA_LABELS.index(decision)],
        context=format_pubmedqa_context(row["context"]),
    )


CONVERTERS = {"medqa": convert_medqa, "medmcqa": convert_medmcqa, "pubmedqa": convert_pubmedqa}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def load_items(dataset: str, n: Optional[int] = None, num_options: Optional[int] = None,
               hf_name: Optional[str] = None, hf_config: Optional[str] = None,
               split: Optional[str] = None, show_sample: bool = True,
               indices: Optional[list[int]] = None) -> list[MCQItem]:
    """
    Load a dataset and convert it to a list of MCQItem.
    n: only take the first n questions (for debugging); None means all.
    num_options: only matters for MedQA (4 or 5).
    indices: only take these rows (e.g. the validation row numbers in data/medqa_split_seed42.json); question_id still uses the original row number.
    """
    from datasets import load_dataset  # imported late so the unit tests don't need datasets installed

    if dataset == "medqa":
        num_options = num_options or 4
    src = resolve_source(dataset, num_options, hf_name, hf_config, split)
    print(f"[data] loading {dataset}: hf_name={src['hf_name']}, config={src['hf_config']}, split={src['split']}")
    ds = load_dataset(src["hf_name"], src["hf_config"], split=src["split"])
    print(f"[data] {len(ds)} rows; fields: {ds.column_names}")
    row_ids = list(range(len(ds)))
    if indices is not None:
        ds = ds.select(indices)
        row_ids = list(indices)
        print(f"[data] took {len(ds)} rows from the row-number list")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
        row_ids = row_ids[: len(ds)]

    if show_sample and len(ds) > 0:
        print("[data] ===== raw example (row 0, check the field names) =====")
        print(json.dumps(_truncate(dict(ds[0])), ensure_ascii=False, indent=2, default=str))

    convert = CONVERTERS[dataset]
    items = [convert(dict(row), i, num_options=num_options) for i, row in zip(row_ids, ds)]

    if show_sample and items:
        print("[data] ===== converted example (check the options and gold answer line up) =====")
        print(json.dumps(_truncate(asdict(items[0])), ensure_ascii=False, indent=2, default=str))
        # print the answer distribution: if a letter never shows up, or it's all one letter, parsing probably went wrong
        dist: dict[str, int] = {}
        for it in items:
            dist[it.gold] = dist.get(it.gold, 0) + 1
        print(f"[data] gold answer distribution: {dict(sorted(dist.items()))}")

    ids = [it.question_id for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{dataset} has duplicate question_ids, so it can't be lined up across models")
    return items


def main():
    p = argparse.ArgumentParser(description="Just load and print dataset examples, no model")
    p.add_argument("--dataset", required=True, choices=list(DATASET_DEFAULTS))
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--num_options", type=int, default=None)
    p.add_argument("--hf_name", default=None)
    p.add_argument("--hf_config", default=None)
    p.add_argument("--split", default=None)
    a = p.parse_args()
    load_items(a.dataset, a.n, a.num_options, a.hf_name, a.hf_config, a.split, show_sample=True)


if __name__ == "__main__":
    main()
