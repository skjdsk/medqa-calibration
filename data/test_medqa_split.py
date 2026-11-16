"""Unit tests for data/medqa_split.py (no download needed): python -m pytest -q data/test_medqa_split.py"""

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
    # datasets' Dataset.shuffle(seed=42) also uses np.random.default_rng(42).permutation,
    # so "shuffle(seed=42), take the first 64" in the pilot is exactly the first 64 training questions
    import numpy as np

    assert load_split_indices("train")[:64] == np.random.default_rng(42).permutation(TOTAL_ROWS)[:64].tolist()
