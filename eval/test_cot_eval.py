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


# ---- 测试集 / 迁移数据集用的 prompt 和解析 ----
import json  # noqa: E402

from eval.cot_eval import (  # noqa: E402
    _TRAIN_LETTER_LIST, build_cot_prompt, format_cot_user_message, result_tag, valid_answer,
)
from eval.data_loaders import MCQItem  # noqa: E402
from train.grpo.grpo_data import INSTRUCTION, build_prompt_text, format_user_message  # noqa: E402


class FakeTokenizer:
    """apply_chat_template 直接把消息列表变成 JSON，便于逐字比较。"""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return json.dumps(messages, ensure_ascii=False)


MEDQA = MCQItem("medqa", "medqa4-1", "Which drug?", ["Aspirin", "Heparin", "Warfarin", "Insulin"], "B")
PUBMED = MCQItem("pubmedqa", "123", "Does X cause Y?", ["yes", "no", "maybe"], "A",
                 context="BACKGROUND: something.\nRESULTS: more.")


def test_instruction_has_letter_list():
    assert INSTRUCTION.count(_TRAIN_LETTER_LIST) == 1


def test_four_option_prompt_identical_to_training():
    opts = dict(zip("ABCD", MEDQA.options))
    assert format_cot_user_message(MEDQA) == format_user_message(MEDQA.question, opts)
    tok = FakeTokenizer()
    assert build_cot_prompt(tok, MEDQA) == build_prompt_text(tok, MEDQA.question, opts)


def test_pubmedqa_prompt():
    msg = format_cot_user_message(PUBMED)
    assert msg.startswith("Context:\nBACKGROUND: something.\nRESULTS: more.\n\nDoes X cause Y?\n\nA. yes\nB. no\nC. maybe\n")
    assert msg.endswith("where X is one of A, B, or C.")
    assert "D." not in msg
    system = json.loads(build_cot_prompt(FakeTokenizer(), PUBMED))[0]["content"]
    assert system == json.loads(build_cot_prompt(FakeTokenizer(), MEDQA))[0]["content"]


def test_valid_answer_drops_missing_letters():
    assert valid_answer("reasoning\nAnswer: D", ("A", "B", "C")) is None
    assert valid_answer("reasoning\nAnswer: B", ("A", "B", "C")) == "B"
    assert valid_answer("Answer: yes", ("A", "B", "C")) is None


def test_vote_probs_three_options():
    assert vote_probs(["A", "C", None], ("A", "B", "C")) == {"A": 0.5, "B": 0, "C": 0.5}


def test_result_tags_match_evaluate_py():
    assert result_tag("medqa", "val") == "medqa4_val"
    assert result_tag("medqa", "test") == "medqa4"
    assert result_tag("medmcqa", "val") == "medmcqa"
    assert result_tag("pubmedqa", "test") == "pubmedqa"
