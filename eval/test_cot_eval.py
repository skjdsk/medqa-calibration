"""eval/cot_eval.py 里纯 Python 工具的单元测试（不需要 GPU / vLLM）：python -m pytest -q eval/test_cot_eval.py"""

import math

from eval.cot_eval import find_answer_token_index, letter_probs_from_logprobs, vote_probs

# 假的词表：每个 token id 对应一段文字
VOCAB = {1: "The", 2: " answer", 3: " is", 4: " A", 5: ".\n", 6: "Answer", 7: ":", 8: " C", 9: "B", 10: "**",
         11: " (", 12: " D"}
LETTER_IDS = {"A": [4], "B": [9], "C": [8], "D": [12]}


def decode(ids):
    return "".join(VOCAB[i] for i in ids)


def test_finds_last_answer_letter():
    ids = [6, 7, 4, 5, 6, 7, 8]  # "Answer: A.\nAnswer: C" → 取最后一个
    assert find_answer_token_index(ids, LETTER_IDS, decode) == 6


def test_ignores_letters_not_after_answer():
    ids = [1, 2, 3, 4, 5]  # "The answer is A.\n" 没有冒号，不算按格式回答
    assert find_answer_token_index(ids, LETTER_IDS, decode) is None


def test_bold_and_paren_variants():
    assert find_answer_token_index([10, 6, 7, 10, 9], LETTER_IDS, decode) == 4  # "**Answer:**B"
    assert find_answer_token_index([6, 7, 11, 12], LETTER_IDS, decode) == 3  # "Answer: (D"


def test_letter_probs_normalized():
    lp = {4: math.log(0.6), 8: math.log(0.2), 99: math.log(0.1)}
    probs, mass = letter_probs_from_logprobs(lp, LETTER_IDS)
    assert math.isclose(mass, 0.8)
    assert math.isclose(probs["A"], 0.75) and math.isclose(probs["C"], 0.25)
    assert probs["B"] == 0 and probs["D"] == 0


def test_vote_probs():
    assert vote_probs(["A", "A", "B", None]) == {"A": 2 / 3, "B": 1 / 3, "C": 0, "D": 0}
    assert vote_probs([None, None]) == {L: 0.25 for L in "ABCD"}
