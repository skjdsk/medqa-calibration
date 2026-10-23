"""
metrics.py —— 只根据每题明细 CSV 计算汇总指标、画可靠性图
=============================================================

【这个文件做什么】
读入 evaluate.py 输出的一个或多个 preds_*.csv，按 (模型, 数据集, 评估模式) 分组，算出：
    accuracy               准确率
    ece                    期望校准误差，15 个等宽区间
    brier                  多分类 Brier：每题对所有选项求 (p − one_hot)² 之和，再对题目取平均
    nll                    −log p(正确选项) 的平均
    hc_wrong_rate          confidence ≥ 0.9 且答错的题数 ÷ 总题数（“很自信却答错”）
    mean_conf_wrong        答错题目的平均 confidence
    hc_accuracy / hc_count confidence ≥ 0.9 的题目的准确率，以及这类题有多少道
    mean_confidence        平均 confidence（和 accuracy 对比可看出整体是过度自信还是不够自信）
    mean_letter_mass       平均“字母概率总和”（归一化前），检查 prompt 是否合理
并画可靠性图（reliability diagram）存成 PNG。

完全不需要模型和 GPU，所以以后改指标定义时直接重跑这个文件即可。

【运行示例】
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
# 从明细表取出概率矩阵
# ---------------------------------------------------------------------------
def prob_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 (probs, gold_idx)：
      probs    形状 [题数, 最大选项数]，这道题不存在的选项（CSV 里是空）记为 0
      gold_idx 每题正确选项的列号
    """
    cols = [f"prob_{L}" for L in LETTERS if f"prob_{L}" in df.columns]
    probs = df[cols].to_numpy(dtype=float)
    probs = np.nan_to_num(probs, nan=0.0)
    letter_to_idx = {L: i for i, L in enumerate(LETTERS)}
    gold_idx = df["gold"].map(letter_to_idx).to_numpy()
    if np.any(pd.isna(gold_idx)) or np.any(gold_idx >= probs.shape[1]):
        raise ValueError("gold 列里有无法识别的字母，或超出了 prob_* 列的范围")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(f"有题目的选项概率之和不等于 1（范围 {row_sums.min():.4f}–{row_sums.max():.4f}）")
    return probs, gold_idx.astype(int)


# ---------------------------------------------------------------------------
# 单项指标（输入都是 numpy 数组，方便单元测试）
# ---------------------------------------------------------------------------
def bin_index(conf: np.ndarray, n_bins: int = 15) -> np.ndarray:
    """等宽区间编号：[0,1/15) → 0，…，[14/15, 1] → 14（1.0 归入最后一个区间）。"""
    return np.clip(np.floor(conf * n_bins).astype(int), 0, n_bins - 1)


def reliability_table(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """每个区间的题数、平均 confidence、准确率。空区间的 conf/acc 为 NaN。"""
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
    """ECE = Σ_区间 (该区间题数 / 总题数) × |该区间准确率 − 该区间平均 confidence|"""
    tab = reliability_table(conf, correct, n_bins)
    tab = tab[tab["count"] > 0]
    n = len(conf)
    return float(((tab["count"] / n) * (tab["accuracy"] - tab["mean_confidence"]).abs()).sum())


def brier_score(probs: np.ndarray, gold_idx: np.ndarray) -> float:
    """多分类 Brier：每题 Σ_选项 (p − one_hot)²，再取平均。范围 0（完美）到 2（满分自信地答错）。"""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(gold_idx)), gold_idx] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(probs: np.ndarray, gold_idx: np.ndarray, eps: float = 1e-12) -> float:
    """NLL = 平均 −log p(正确选项)。p 加下限 eps，避免 log(0) 变成无穷大。"""
    p_gold = probs[np.arange(len(gold_idx)), gold_idx]
    return float(-np.log(np.clip(p_gold, eps, 1.0)).mean())


def compute_metrics(df: pd.DataFrame, n_bins: int = 15, hc_threshold: float = 0.9) -> dict:
    """对一组题目（同一模型 × 同一数据集）计算全部汇总指标。"""
    probs, gold_idx = prob_matrix(df)

    # 从概率重新算出答案和 confidence，并和 CSV 里的列核对（防止 CSV 被手动改坏）
    pred_idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred_idx == gold_idx).astype(float)
    if "correct" in df.columns and not np.array_equal(correct, df["correct"].to_numpy(dtype=float)):
        warnings.warn("CSV 里的 correct 列和由概率重算的结果不一致，以重算结果为准")

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
# 可靠性图
# ---------------------------------------------------------------------------
def plot_reliability(df: pd.DataFrame, title: str, out_path: str, n_bins: int = 15) -> None:
    """
    上半部分：每个区间的准确率（柱子）vs 平均 confidence；对角线 = 完美校准。
              柱子低于对角线 = 过度自信（说得很有把握，实际没那么准）。
    下半部分：每个区间的题目数量（看 confidence 都集中在哪里）。
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
    # 红色半透明：每个区间“平均 confidence 与准确率之间的差距”
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
# 读文件 + 汇总
# ---------------------------------------------------------------------------
def load_predictions(paths: list[str]) -> pd.DataFrame:
    files = []
    for p in paths:
        matched = sorted(glob.glob(p))
        files.extend(matched if matched else [p])
    files = [f for f in dict.fromkeys(files) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"没有找到 CSV：{paths}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "eval_mode" not in df.columns:
        df["eval_mode"] = "direct"
    dup = df.duplicated(subset=GROUP_KEYS + ["question_id"])
    if dup.any():
        raise ValueError(f"同一模型/数据集下有 {dup.sum()} 道重复题目，可能同一个结果文件读了两次")
    print(f"[metrics] 读入 {len(files)} 个文件，共 {len(df)} 行")
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
    p = argparse.ArgumentParser(description="从每题明细 CSV 计算校准指标")
    p.add_argument("--pred_csv", nargs="+", required=True, help="一个或多个 CSV，可以用通配符")
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
    print(f"[metrics] 已保存到 {a.out_dir}/（summary.csv、summary.json、bins_*.csv、reliability_*.png）")


if __name__ == "__main__":
    main()
