"""
data/medqa_split.py —— MedQA train split 的固定划分：5000 题训练 + 500 题验证

【这个文件做什么】
proposal 里写的是"5000 题训练 + 500 题验证"。MedQA（GBaker/MedQA-USMLE-4-options）的 train split 共 10178 题，
这里用固定种子把它打乱一次，然后：
    打乱后的第 0–4999 个   → 训练集（SFT / DPO / GRPO 都只用这 5000 题训练）
    打乱后的第 5000–5499 个 → 验证集（调参、选 checkpoint、看校准；绝对不能拿来训练）
test split（1273 题）不在这里，只用于最终评估。

划分结果存成 data/medqa_split_seed42.json（每题在原 train split 里的行号），提交到 git。
**所有方法（SFT / DPO / GRPO）、所有训练 seed 都读这同一个文件**，保证用的是同一批题。
注意：这里的 seed 42 只决定"哪些题进训练集/验证集"，和训练时的 --seed（采样、LoRA 初始化、数据顺序）是两回事。

eval/ 里 MedQA 的 question_id 是 "medqa4-<行号>"，和这里的行号一一对应。

【用法】
    python -m data.medqa_split --check     # 重新生成一遍，和已提交的文件对比（不需要下载数据）
    python -m data.medqa_split --write     # 重新写文件（一般不需要；改了划分会让之前的结果全部不可比）

    from data.medqa_split import load_split_indices
    train_idx = load_split_indices("train")   # 5000 个行号
    val_idx   = load_split_indices("val")     # 500 个行号
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
SOURCE_SPLIT = "train"
TOTAL_ROWS = 10178  # GBaker/MedQA-USMLE-4-options train split 的题数
SPLIT_SEED = 42
N_TRAIN = 5000
N_VAL = 500
SPLIT_FILE = Path(__file__).resolve().parent / "medqa_split_seed42.json"
SUBSETS = ("train", "val")


def make_split(total: int = TOTAL_ROWS, seed: int = SPLIT_SEED, n_train: int = N_TRAIN, n_val: int = N_VAL) -> dict:
    """用固定种子打乱 0..total-1，前 n_train 个做训练集，接下来 n_val 个做验证集。"""
    if n_train + n_val > total:
        raise ValueError(f"n_train + n_val = {n_train + n_val} 超过总题数 {total}")
    # 和 datasets 的 Dataset.shuffle(seed) 用的是同一种打乱方式（numpy default_rng 的 permutation）
    perm = np.random.default_rng(seed).permutation(total).tolist()
    return {
        "dataset": DATASET_NAME,
        "source_split": SOURCE_SPLIT,
        "total_rows": total,
        "split_seed": seed,
        "method": "np.random.default_rng(split_seed).permutation(total_rows)；前 n_train 个为 train，接下来 n_val 个为 val",
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
        raise ValueError(f"{path} 里有重复的行号")
    if set(train) & set(val):
        raise ValueError(f"{path} 里训练集和验证集有重叠")
    return split


def load_split_indices(subset: str, path: str | Path = SPLIT_FILE) -> list[int]:
    """返回 'train' 或 'val' 的行号列表（原 train split 里的行号，按打乱后的顺序）。"""
    if subset not in SUBSETS:
        raise ValueError(f"subset 只能是 {SUBSETS}，收到 {subset!r}")
    return load_split(path)[f"{subset}_indices"]


def check_dataset_size(n_rows: int, split: dict) -> None:
    """加载数据后调用：确认数据集行数和划分时一致，否则行号会对不上。"""
    if n_rows != split["total_rows"]:
        raise ValueError(
            f"{split['dataset']} {split['source_split']} 现在有 {n_rows} 行，但划分文件是按 "
            f"{split['total_rows']} 行做的。数据集版本变了，行号对不上，请先确认数据集版本。"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="MedQA 5000/500 训练/验证划分")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="重新生成并和已提交的文件对比")
    g.add_argument("--write", action="store_true", help="重新写划分文件")
    args = p.parse_args()

    fresh = make_split()
    if args.write:
        with open(SPLIT_FILE, "w", encoding="utf-8") as f:
            json.dump(fresh, f, ensure_ascii=False)
            f.write("\n")
        print(f"已写入 {SPLIT_FILE}：train {len(fresh['train_indices'])} 题，val {len(fresh['val_indices'])} 题")
    else:
        saved = load_split()
        same = saved["train_indices"] == fresh["train_indices"] and saved["val_indices"] == fresh["val_indices"]
        print(f"{SPLIT_FILE.name}：train {len(saved['train_indices'])} 题，val {len(saved['val_indices'])} 题，"
              f"{'和重新生成的一致' if same else '⚠️ 和重新生成的不一致'}")
        if not same:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
