"""data/medqa_split.py 的单元测试（不需要下载数据）：python -m pytest -q data/test_medqa_split.py"""

from data.medqa_split import N_TRAIN, N_VAL, TOTAL_ROWS, load_split, load_split_indices, make_split


def test_saved_file_matches_regeneration():
    saved, fresh = load_split(), make_split()
    assert saved["train_indices"] == fresh["train_indices"]
    assert saved["val_indices"] == fresh["val_indices"]


def test_sizes_and_disjoint():
    train, val = load_split_indices("train"), load_split_indices("val")
    assert len(train) == N_TRAIN and len(val) == N_VAL
    assert len(set(train)) == N_TRAIN and len(set(val)) == N_VAL
    assert not set(train) & set(val)
    assert all(0 <= i < TOTAL_ROWS for i in train + val)


def test_same_order_as_datasets_shuffle():
    # datasets 的 Dataset.shuffle(seed=42) 用的也是 np.random.default_rng(42).permutation，
    # 所以试跑时 "shuffle(seed=42) 取前 64 题" 正好是训练集的前 64 题
    import numpy as np

    assert load_split_indices("train")[:64] == np.random.default_rng(42).permutation(TOTAL_ROWS)[:64].tolist()
