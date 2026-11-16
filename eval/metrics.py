"""
metrics.py -- compute summary metrics and draw reliability diagrams from the per-question CSVs only
=============================================================

[What this file does]
Reads one or more preds_*.csv files from evaluate.py, groups them by (model, dataset, eval mode), and computes:
        accuracy               accuracy
        ece                    expected calibration error, 15 equal-width bins
        brier                  multi-class Brier: per question, sum of (p - one_hot)^2 over all options, averaged over questions
        nll                    mean of -log p(correct option)
        hc_wrong_rate          questions with confidence >= 0.9 that are wrong / all questions ("very sure but wrong")
        mean_conf_wrong        mean confidence on the questions it got wrong
        hc_accuracy / hc_count accuracy on questions with confidence >= 0.9, and how many there are
        mean_confidence        mean confidence (compare with accuracy to see if it's over- or underconfident overall)
        mean_letter_mass       mean total letter probability (before normalizing), a sanity check on the prompt
and saves reliability diagrams as PNGs.

Needs no model and no GPU, so if a metric definition changes later, just rerun this file.

[Example]
    python -m eval.metrics --pred_csv results/preds_*.csv --out_dir results/metrics
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

LETTERS = list("ABCDEFGHIJ")
GROUP_KEYS = ["model_name", "dataset", "eval_mode"]


# ---------------------------------------------------------------------------
# get the probability matrix from the per-question table
# ---------------------------------------------------------------------------
def prob_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (probs, gold_idx):
        probs    shape [questions, max options]; options a question doesn't have (empty in the CSV) are 0
        gold_idx column index of each question's correct option
    """
    cols = [f"prob_{L}" for L in LETTERS if f"prob_{L}" in df.columns]
    probs = df[cols].to_numpy(dtype=float)
    probs = np.nan_to_num(probs, nan=0.0)
    letter_to_idx = {L: i for i, L in enumerate(LETTERS)}
    gold_idx = df["gold"].map(letter_to_idx).to_numpy()
    if np.any(pd.isna(gold_idx)) or np.any(gold_idx >= probs.shape[1]):
        raise ValueError("The gold column has letters that can't be parsed or are beyond the prob_* columns")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(f"Some questions' option probabilities don't sum to 1 (range {row_sums.min():.4f}-{row_sums.max():.4f})")
    return probs, gold_idx.astype(int)


# ---------------------------------------------------------------------------
# Individual metrics (all take numpy arrays, so they're easy to unit test)
# ---------------------------------------------------------------------------
def bin_index(conf: np.ndarray, n_bins: int = 15) -> np.ndarray:
    """Equal-width bin index: [0,1/15) -> 0, ..., [14/15, 1] -> 14 (1.0 goes in the last bin)."""
    return np.clip(np.floor(conf * n_bins).astype(int), 0, n_bins - 1)


def reliability_table(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Per bin: number of questions, mean confidence, accuracy. Empty bins get NaN for conf/acc."""
    idx = bin_index(conf, n_bins)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin": b,
            "bin_lower": b / n_bins,
            "bin_upper": (b + 1) / n_bins,
            "count": int(m.sum()),
            "mean_confidence": float(conf[m].mean()) if m.any() else np.nan,
            "accuracy": float(correct[m].mean()) if m.any() else np.nan,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """ECE = sum over bins of (questions in bin / total) x |bin accuracy - bin mean confidence|"""
    tab = reliability_table(conf, correct, n_bins)
    tab = tab[tab["count"] > 0]
    n = len(conf)
    return float(((tab["count"] / n) * (tab["accuracy"] - tab["mean_confidence"]).abs()).sum())


def brier_score(probs: np.ndarray, gold_idx: np.ndarray) -> float:
    """Multi-class Brier: per question sum over options of (p - one_hot)^2, then averaged. Ranges from 0 (perfect) to 2 (fully confident and wrong)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(gold_idx)), gold_idx] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(probs: np.ndarray, gold_idx: np.ndarray, eps: float = 1e-12) -> float:
    """NLL = mean of -log p(correct option). p gets a floor of eps so log(0) doesn't blow up."""
    p_gold = probs[np.arange(len(gold_idx)), gold_idx]
    return float(-np.log(np.clip(p_gold, eps, 1.0)).mean())


def compute_metrics(df: pd.DataFrame, n_bins: int = 15, hc_threshold: float = 0.9) -> dict:
    """Compute every summary metric for one group of questions (one model x one dataset)."""
    probs, gold_idx = prob_matrix(df)

    # recompute the answer and confidence from the probabilities and check them against the CSV columns (in case the CSV got edited by hand)
    pred_idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred_idx == gold_idx).astype(float)
    if "correct" in df.columns and not np.array_equal(correct, df["correct"].to_numpy(dtype=float)):
        warnings.warn("The correct column in the CSV doesn't match what the probabilities give; using the recomputed values")

    n = len(df)
    wrong = correct == 0
    hc = conf >= hc_threshold
    return {
        "n": n,
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "ece": expected_calibration_error(conf, correct, n_bins),
        "brier": brier_score(probs, gold_idx),
        "nll": negative_log_likelihood(probs, gold_idx),
        "hc_wrong_rate": float((hc & wrong).sum() / n),
        "hc_wrong_count": int((hc & wrong).sum()),
        "mean_conf_wrong": float(conf[wrong].mean()) if wrong.any() else float("nan"),
        "hc_count": int(hc.sum()),
        "hc_accuracy": float(correct[hc].mean()) if hc.any() else float("nan"),
        "mean_letter_mass": float(df["letter_mass"].mean()) if "letter_mass" in df.columns else float("nan"),
        "min_letter_mass": float(df["letter_mass"].min()) if "letter_mass" in df.columns else float("nan"),
    }


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------
def plot_reliability(df: pd.DataFrame, title: str, out_path: str, n_bins: int = 15) -> None:
    """
    Top: accuracy per bin (bars) vs mean confidence; the diagonal = perfect calibration.
                        Bars below the diagonal = overconfident (sounds sure, isn't that accurate).
    Bottom: number of questions per bin (shows where the confidence piles up).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probs, gold_idx = prob_matrix(df)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == gold_idx).astype(float)
    tab = reliability_table(conf, correct, n_bins)
    ece = expected_calibration_error(conf, correct, n_bins)
    width = 1.0 / n_bins
    nonempty = tab[tab["count"] > 0]

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    ax.bar(nonempty["bin_lower"], nonempty["accuracy"], width=width, align="edge",
           color="#4C72B0", edgecolor="white", label="Accuracy")
    # translucent red: the gap between mean confidence and accuracy in each bin
    ax.bar(nonempty["bin_lower"], nonempty["mean_confidence"] - nonempty["accuracy"],
           bottom=nonempty["accuracy"], width=width, align="edge",
           color="#C44E52", alpha=0.35, edgecolor="#C44E52", label="Gap (conf − acc)")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=10)
    ax.text(0.03, 0.95, f"n = {len(df)}\nAcc = {correct.mean():.3f}\nECE = {ece:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(loc="lower right", fontsize=8)

    axh.bar(tab["bin_lower"], tab["count"], width=width, align="edge", color="gray", edgecolor="white")
    axh.set_xlabel("Confidence")
    axh.set_ylabel("Count")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Read files + summarize
# ---------------------------------------------------------------------------
def load_predictions(paths: list[str]) -> pd.DataFrame:
    files = []
    for p in paths:
        matched = sorted(glob.glob(p))
        files.extend(matched if matched else [p])
    files = [f for f in dict.fromkeys(files) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSVs found: {paths}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "eval_mode" not in df.columns:
        df["eval_mode"] = "direct"
    dup = df.duplicated(subset=GROUP_KEYS + ["question_id"])
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicate questions for the same model/dataset; the same result file may have been read twice")
    print(f"[metrics] read {len(files)} files, {len(df)} rows")
    return df


def summarize(df: pd.DataFrame, n_bins: int = 15, hc_threshold: float = 0.9,
              out_dir: str | None = None, plots: bool = True) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(GROUP_KEYS, sort=True):
        m = dict(zip(GROUP_KEYS, keys))
        m.update(compute_metrics(g, n_bins, hc_threshold))
        rows.append(m)
        if out_dir:
            tag = "_".join(str(k) for k in keys)
            probs, gold_idx = prob_matrix(g)
            conf = probs.max(axis=1)
            correct = (probs.argmax(axis=1) == gold_idx).astype(float)
            reliability_table(conf, correct, n_bins).to_csv(
                os.path.join(out_dir, f"bins_{tag}.csv"), index=False)
            if plots:
                plot_reliability(g, f"{keys[0]} | {keys[1]} | {keys[2]}",
                                 os.path.join(out_dir, f"reliability_{tag}.png"), n_bins)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="Compute calibration metrics from per-question CSVs")
    p.add_argument("--pred_csv", nargs="+", required=True, help="one or more CSVs, wildcards OK")
    p.add_argument("--out_dir", default="results/metrics")
    p.add_argument("--n_bins", type=int, default=15)
    p.add_argument("--hc_threshold", type=float, default=0.9)
    p.add_argument("--no_plots", action="store_true")
    a = p.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    df = load_predictions(a.pred_csv)
    summary = summarize(df, a.n_bins, a.hc_threshold, a.out_dir, plots=not a.no_plots)
    summary.to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.round(4).to_string(index=False))
    print(f"[metrics] saved to {a.out_dir}/ (summary.csv, summary.json, bins_*.csv, reliability_*.png)")


if __name__ == "__main__":
    main()
