"""
test_metrics.py -- check that ECE, Brier, NLL etc. come out right on small hand-made examples
===================================================================

[What this file does]
Each test is a few fake questions whose answers we can work out with pen and paper. If a formula in metrics.py is wrong,
these tests fail. To run (inside the calibration folder):
    python -m pytest eval/test_metrics.py -v

Note: the confidences in the examples deliberately avoid bin edges (e.g. 0.8 = 12/15 sits right on one),
so floating-point error can't push a question into the next bin and make the result machine-dependent.
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
    """Turn [[pA, pB, ...], ...] and ["A", "B", ...] into a DataFrame in the same format evaluate.py writes."""
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
# Example 1: a perfect model (100% sure and always right) -> every error metric is 0
# ---------------------------------------------------------------------------
def test_perfect_model():
    df = make_df([[1, 0, 0, 0], [0, 0, 1, 0]], ["A", "C"])
    m = compute_metrics(df)
    assert m["accuracy"] == 1.0
    assert m["ece"] == pytest.approx(0.0, abs=1e-12)
    assert m["brier"] == pytest.approx(0.0, abs=1e-12)
    assert m["nll"] == pytest.approx(0.0, abs=1e-9)
    assert m["hc_wrong_rate"] == 0.0
    assert math.isnan(m["mean_conf_wrong"])  # nothing answered wrong
    assert m["hc_count"] == 2 and m["hc_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Example 2: ECE / Brier / NLL by hand (2 options, 4 questions)
#   q1 [0.85, 0.15] gold A -> answers A, right, conf 0.85, bin floor(0.85*15)=12
#   q2 [0.85, 0.15] gold B -> answers A, wrong, conf 0.85, bin 12
#   q3 [0.62, 0.38] gold A -> right,       conf 0.62, bin floor(9.3)=9
#   q4 [0.30, 0.70] gold B -> right,       conf 0.70, bin floor(10.5)=10
# ECE: bin 12 has two, accuracy 0.5, mean conf 0.85 -> 2/4 x 0.35 = 0.175
#      bin 9  has one, accuracy 1,   conf 0.62      -> 1/4 x 0.38 = 0.095
#      bin 10 has one, accuracy 1,   conf 0.70      -> 1/4 x 0.30 = 0.075
#      total 0.345
# Brier：q1 0.15²+0.15²=0.045；q2 0.85²+0.85²=1.445；q3 2×0.38²=0.2888；q4 2×0.3²=0.18
#        mean = 1.9588 / 4 = 0.4897
# NLL: mean of [-ln0.85, -ln0.15, -ln0.62, -ln0.70]
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
    assert m["nll"] == pytest.approx(0.723587, abs=1e-5)  # worked out by hand
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
# Example 3: high-confidence-wrong metrics (3 options, 5 questions)
#   q1 [0.95,0.03,0.02] gold B -> wrong, conf 0.95 (confident and wrong)
#   q2 [0.92,0.05,0.03] gold A -> right, conf 0.92
#   q3 [0.90,0.06,0.04] gold C -> wrong, conf exactly 0.90 (>= 0.9 counts)
#   q4 [0.50,0.30,0.20] gold A -> right
#   q5 [0.40,0.35,0.25] gold C -> wrong
# high-confidence-wrong rate = 2/5; 3 high-confidence questions, 1 right -> accuracy 1/3
# mean conf on wrong answers = (0.95 + 0.90 + 0.40) / 3 = 0.75
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
# Example 4: columns for options that don't exist (e.g. PubMedQA has no D/E) are empty and mustn't affect Brier
#   one question [0.6, 0.3, 0.1] gold A: Brier = 0.4^2 + 0.3^2 + 0.1^2 = 0.26
# ---------------------------------------------------------------------------
def test_missing_option_columns_ignored():
    df = make_df([[0.6, 0.3, 0.1]], ["A"])  # prob_D, prob_E are NaN
    assert df["prob_D"].isna().all()
    m = compute_metrics(df)
    assert m["brier"] == pytest.approx(0.26)
    assert m["nll"] == pytest.approx(-math.log(0.6))


def test_brier_worst_case_is_two():
    # 100% sure and wrong: (1-0)^2 + (0-1)^2 = 2
    df = make_df([[1.0, 0.0, 0.0, 0.0]], ["B"])
    m = compute_metrics(df)
    assert m["brier"] == pytest.approx(2.0)
    assert m["nll"] == pytest.approx(-math.log(1e-12))  # p=0 gets clipped to 1e-12 so it doesn't go to infinity


# ---------------------------------------------------------------------------
# Binning: 15 equal-width bins, 1.0 goes in the last one
# ---------------------------------------------------------------------------
def test_bins():
    conf = np.array([0.0, 0.05, 0.07, 0.5, 0.99, 1.0])
    assert list(bin_index(conf, 15)) == [0, 0, 1, 7, 14, 14]
    tab = reliability_table(np.array([0.25, 0.26, 0.95]), np.array([1.0, 0.0, 1.0]), 15)
    assert len(tab) == 15 and tab["count"].sum() == 3
    row3 = tab.loc[tab["bin"] == 3].iloc[0]  # 0.25 and 0.26 are both in [0.2, 0.2667)
    assert row3["count"] == 2 and row3["accuracy"] == 0.5


def test_probabilities_must_sum_to_one():
    df = make_df([[0.5, 0.2, 0.1, 0.1]], ["A"])  # sums to 0.9
    with pytest.raises(ValueError):
        compute_metrics(df)
