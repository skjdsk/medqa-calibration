"""
test_metrics.py —— 用手工编的小例子检查 ECE、Brier、NLL 等是否算对
===================================================================

【这个文件做什么】
每个测试都是几道“假题目”，答案是我们用纸笔就能算出来的。如果 metrics.py 的公式写错了，
这里的测试就会失败。运行方法（在 calibration 文件夹里）：
    python -m pytest eval/test_metrics.py -v

注意：例子里的 confidence 故意避开区间边界（如 0.8 = 12/15 正好在边界上），
防止浮点误差导致题目落进相邻区间，让测试结果依赖机器。
"""

import math

import numpy as np
import pandas as pd
import pytest

from eval.metrics import (
    bin_index,
    brier_score,
    compute_metrics,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_table,
)


def make_df(prob_rows, golds, letters="ABCDE"):
    """把 [[pA, pB, ...], ...] 和 ["A", "B", ...] 变成和 evaluate.py 输出相同格式的 DataFrame。"""
    rows = []
    for i, (p, g) in enumerate(zip(prob_rows, golds)):
        row = {"dataset": "toy", "question_id": f"q{i}", "model_name": "m", "eval_mode": "direct",
               "gold": g, "letter_mass": 0.95}
        for j, L in enumerate(letters):
            row[f"prob_{L}"] = p[j] if j < len(p) else np.nan
        pred = "ABCDE"[int(np.argmax(p))]
        row.update(pred=pred, confidence=max(p), correct=int(pred == g))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 例 1：完美的模型（100% 把握且全对）→ 所有误差指标都是 0
# ---------------------------------------------------------------------------
def test_perfect_model():
    df = make_df([[1, 0, 0, 0], [0, 0, 1, 0]], ["A", "C"])
    m = compute_metrics(df)
    assert m["accuracy"] == 1.0
    assert m["ece"] == pytest.approx(0.0, abs=1e-12)
    assert m["brier"] == pytest.approx(0.0, abs=1e-12)
    assert m["nll"] == pytest.approx(0.0, abs=1e-9)
    assert m["hc_wrong_rate"] == 0.0
    assert math.isnan(m["mean_conf_wrong"])  # 没有答错的题
    assert m["hc_count"] == 2 and m["hc_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# 例 2：手算 ECE / Brier / NLL（2 个选项，4 道题）
#   q1 [0.85, 0.15] 标准 A → 答 A 对，conf 0.85，区间 floor(0.85*15)=12
#   q2 [0.85, 0.15] 标准 B → 答 A 错，conf 0.85，区间 12
#   q3 [0.62, 0.38] 标准 A → 对，     conf 0.62，区间 floor(9.3)=9
#   q4 [0.30, 0.70] 标准 B → 对，     conf 0.70，区间 floor(10.5)=10
# ECE：区间12 两题，准确率 0.5，平均conf 0.85 → 2/4 × 0.35 = 0.175
#      区间9  一题，准确率 1，  conf 0.62      → 1/4 × 0.38 = 0.095
#      区间10 一题，准确率 1，  conf 0.70      → 1/4 × 0.30 = 0.075
#      合计 0.345
# Brier：q1 0.15²+0.15²=0.045；q2 0.85²+0.85²=1.445；q3 2×0.38²=0.2888；q4 2×0.3²=0.18
#        平均 = 1.9588 / 4 = 0.4897
# NLL：平均[−ln0.85, −ln0.15, −ln0.62, −ln0.70]
# ---------------------------------------------------------------------------
PROBS2 = [[0.85, 0.15], [0.85, 0.15], [0.62, 0.38], [0.30, 0.70]]
GOLDS2 = ["A", "B", "A", "B"]


def test_hand_computed_example():
    df = make_df(PROBS2, GOLDS2)
    m = compute_metrics(df)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["ece"] == pytest.approx(0.345, abs=1e-9)
    assert m["brier"] == pytest.approx(0.4897, abs=1e-9)
    expected_nll = -(math.log(0.85) + math.log(0.15) + math.log(0.62) + math.log(0.70)) / 4
    assert m["nll"] == pytest.approx(expected_nll, abs=1e-9)
    assert m["nll"] == pytest.approx(0.723587, abs=1e-5)  # 手算数值
    assert m["mean_confidence"] == pytest.approx((0.85 + 0.85 + 0.62 + 0.70) / 4)


def test_low_level_functions_match():
    probs = np.array(PROBS2)
    gold = np.array([0, 1, 0, 1])
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == gold).astype(float)
    assert expected_calibration_error(conf, correct, 15) == pytest.approx(0.345)
    assert brier_score(probs, gold) == pytest.approx(0.4897)
    assert negative_log_likelihood(probs, gold) == pytest.approx(0.723587, abs=1e-5)


# ---------------------------------------------------------------------------
# 例 3：高置信错误相关指标（3 个选项，5 道题）
#   q1 [0.95,0.03,0.02] 标准 B → 错，conf 0.95（高置信错误）
#   q2 [0.92,0.05,0.03] 标准 A → 对，conf 0.92
#   q3 [0.90,0.06,0.04] 标准 C → 错，conf 正好 0.90（≥0.9 要算进去）
#   q4 [0.50,0.30,0.20] 标准 A → 对
#   q5 [0.40,0.35,0.25] 标准 C → 错
# 高置信错误比例 = 2/5；高置信题数 3，其中对 1 道 → 准确率 1/3
# 答错题平均 conf = (0.95 + 0.90 + 0.40) / 3 = 0.75
# ---------------------------------------------------------------------------
def test_high_confidence_metrics():
    df = make_df([[0.95, 0.03, 0.02], [0.92, 0.05, 0.03], [0.90, 0.06, 0.04],
                  [0.50, 0.30, 0.20], [0.40, 0.35, 0.25]], ["B", "A", "C", "A", "C"])
    m = compute_metrics(df)
    assert m["accuracy"] == pytest.approx(2 / 5)
    assert m["hc_wrong_rate"] == pytest.approx(2 / 5)
    assert m["hc_wrong_count"] == 2
    assert m["hc_count"] == 3
    assert m["hc_accuracy"] == pytest.approx(1 / 3)
    assert m["mean_conf_wrong"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 例 4：选项不存在的列（如 PubMedQA 没有 D/E）是空值，不能影响 Brier
#   单题 [0.6, 0.3, 0.1] 标准 A：Brier = 0.4² + 0.3² + 0.1² = 0.26
# ---------------------------------------------------------------------------
def test_missing_option_columns_ignored():
    df = make_df([[0.6, 0.3, 0.1]], ["A"])  # prob_D、prob_E 是 NaN
    assert df["prob_D"].isna().all()
    m = compute_metrics(df)
    assert m["brier"] == pytest.approx(0.26)
    assert m["nll"] == pytest.approx(-math.log(0.6))


def test_brier_worst_case_is_two():
    # 100% 自信地选错：(1−0)² + (0−1)² = 2
    df = make_df([[1.0, 0.0, 0.0, 0.0]], ["B"])
    m = compute_metrics(df)
    assert m["brier"] == pytest.approx(2.0)
    assert m["nll"] == pytest.approx(-math.log(1e-12))  # p=0 时被截到 1e-12，不会变成无穷大


# ---------------------------------------------------------------------------
# 区间划分：15 个等宽区间，1.0 归入最后一个区间
# ---------------------------------------------------------------------------
def test_bins():
    conf = np.array([0.0, 0.05, 0.07, 0.5, 0.99, 1.0])
    assert list(bin_index(conf, 15)) == [0, 0, 1, 7, 14, 14]
    tab = reliability_table(np.array([0.25, 0.26, 0.95]), np.array([1.0, 0.0, 1.0]), 15)
    assert len(tab) == 15 and tab["count"].sum() == 3
    row3 = tab.loc[tab["bin"] == 3].iloc[0]  # 0.25 和 0.26 都在 [0.2, 0.2667)
    assert row3["count"] == 2 and row3["accuracy"] == 0.5


def test_probabilities_must_sum_to_one():
    df = make_df([[0.5, 0.2, 0.1, 0.1]], ["A"])  # 和为 0.9
    with pytest.raises(ValueError):
        compute_metrics(df)
