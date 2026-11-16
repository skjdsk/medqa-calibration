"""
data/medqa_split.py -- the fixed split of the MedQA train split: 5000 training + 500 validation questions

[What this file does]
The proposal says "5000 training + 500 validation questions". The MedQA (GBaker/MedQA-USMLE-4-options) train split has 10178 questions;
we shuffle it once with a fixed seed, then:
        shuffled 0-4999    -> training set (SFT / DPO / GRPO all train on just these 5000)
        shuffled 5000-5499 -> validation set (tuning, picking checkpoints, checking calibration; never used for training)
The test split (1273 questions) isn't here; it's only used for the final eval.

The split is saved as data/medqa_split_seed42.json (each question's row number in the original train split) and committed to git.
**Every method (SFT / DPO / GRPO) and every training seed reads this same file**, so everyone uses the same questions.
Note: seed 42 here only decides which questions go into train/val. It has nothing to do with --seed at training time (sampling, LoRA init, data order).

MedQA question_ids in eval/ are "medqa4-<row number>", matching the row numbers here one to one.

[Usage]
        python -m data.medqa_split --check     # regenerate and compare with the committed file (no download needed)
        python -m data.medqa_split --write     # rewrite the file (you usually don't want this; changing the split makes all earlier results incomparable)

    from data.medqa_split import load_split_indices
    train_idx = load_split_indices("train")   # 5000 row numbers
    val_idx   = load_split_indices("val")     # 500 row numbers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
SOURCE_SPLIT = "train"
TOTAL_ROWS = 10178  # number of questions in the GBaker/MedQA-USMLE-4-options train split
SPLIT_SEED = 42
N_TRAIN = 5000
N_VAL = 500
SPLIT_FILE = Path(__file__).resolve().parent / "medqa_split_seed42.json"
SUBSETS = ("train", "val")


def make_split(total: int = TOTAL_ROWS, seed: int = SPLIT_SEED, n_train: int = N_TRAIN, n_val: int = N_VAL) -> dict:
    """Shuffle 0..total-1 with a fixed seed; the first n_train are training, the next n_val are validation."""
    if n_train + n_val > total:
        raise ValueError(f"n_train + n_val = {n_train + n_val} is more than the {total} questions available")
    # same shuffle as datasets' Dataset.shuffle(seed) (numpy default_rng permutation)
    perm = np.random.default_rng(seed).permutation(total).tolist()
    return {
        "dataset": DATASET_NAME,
        "source_split": SOURCE_SPLIT,
        "total_rows": total,
        "split_seed": seed,
        "method": "np.random.default_rng(split_seed).permutation(total_rows); first n_train are train, next n_val are val",
        "n_train": n_train,
        "n_val": n_val,
        "train_indices": perm[:n_train],
        "val_indices": perm[n_train : n_train + n_val],
    }


def load_split(path: str | Path = SPLIT_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    train, val = split["train_indices"], split["val_indices"]
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError(f"{path} has duplicate row numbers")
    if set(train) & set(val):
        raise ValueError(f"{path} has overlap between train and val")
    return split


def load_split_indices(subset: str, path: str | Path = SPLIT_FILE) -> list[int]:
    """Return the row numbers for 'train' or 'val' (rows in the original train split, in shuffled order)."""
    if subset not in SUBSETS:
        raise ValueError(f"subset has to be one of {SUBSETS}, got {subset!r}")
    return load_split(path)[f"{subset}_indices"]


def check_dataset_size(n_rows: int, split: dict) -> None:
    """Call after loading the data: checks the dataset has the same number of rows as when the split was made, otherwise row numbers won't line up."""
    if n_rows != split["total_rows"]:
        raise ValueError(
            f"{split['dataset']} {split['source_split']} now has {n_rows} rows, but the split file was made with "
            f"{split['total_rows']} rows. The dataset version changed and row numbers won't line up; check the dataset version first."
        )


def main() -> None:
    p = argparse.ArgumentParser(description="MedQA 5000/500 train/val split")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="regenerate and compare with the committed file")
    g.add_argument("--write", action="store_true", help="rewrite the split file")
    args = p.parse_args()

    fresh = make_split()
    if args.write:
        with open(SPLIT_FILE, "w", encoding="utf-8") as f:
            json.dump(fresh, f, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {SPLIT_FILE}: train {len(fresh['train_indices'])}, val {len(fresh['val_indices'])}")
    else:
        saved = load_split()
        same = saved["train_indices"] == fresh["train_indices"] and saved["val_indices"] == fresh["val_indices"]
        print(f"{SPLIT_FILE.name}: train {len(saved['train_indices'])}, val {len(saved['val_indices'])}, "
              f"{'matches the regenerated split' if same else 'Heads up: does NOT match the regenerated split'}")
        if not same:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
