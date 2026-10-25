"""
train/grpo/test_rewards.py —— 打分函数的单元测试

【这个文件做什么】
用一堆"模拟的模型回答"检查答案提取和打分是否正确，例如：
  - 标准格式、markdown 加粗、括号、全角冒号、小写 answer 等变体能否正确提取
  - 回答中出现多个 "Answer:" 时是否取最后一个
  - 没按格式回答 / 写了 E / "Answer: Doxycycline" 这种情况是否判为提取失败（reward = 0）
  - 监控指标（格式正确率、组内全相同比例、截断比例）是否算对

运行（在 repo 根目录）：
    python -m pytest -q train/grpo/test_rewards.py
不需要 GPU，也不需要下载模型。
"""

import sys
from pathlib import Path

import pytest

# 让 "from train.grpo.rewards import ..." 在任何运行方式下都能找到模块
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.grpo.rewards import (  # noqa: E402
    RewardTracker,
    completion_to_text,
    extract_answer,
    make_correctness_reward,
    score_completion,
)


# ---------------------------------------------------------------------------
# 1. 答案提取：各种回答格式
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        # 标准格式
        ("The patient has hypertension...\nAnswer: C", "C"),
        ("Answer: A", "A"),
        ("推理……\nAnswer:B", "B"),  # 冒号后无空格
        ("Step 1 ...\nAnswer: D\n", "D"),  # 结尾多一个换行
        # 大小写 / 前缀变体
        ("answer: B", "B"),
        ("ANSWER: C", "C"),
        ("Final Answer: A", "A"),
        # markdown / 括号 / 标点
        ("**Answer:** D", "D"),
        ("**Answer: B**", "B"),
        ("Answer: **C**", "C"),
        ("Answer: (A)", "A"),
        ("Answer: (C) Hypertension", "C"),
        ("Answer: B. Metformin", "B"),
        ("Answer: D)", "D"),
        ("Answer：C", "C"),  # 全角冒号
        # 多个 Answer：取最后一个
        ("At first I thought Answer: A, but on reflection...\nAnswer: C", "C"),
        ("I will end with 'Answer: X'.\nReasoning...\nAnswer: B", "B"),
        # 选项后接逗号：取第一个字母（说明模型犹豫，按惯例取它写的第一个）
        ("Answer: A, B", "A"),
    ],
)
def test_extract_valid(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",  # 空回答
        "The answer is C.",  # 没有冒号：不算按格式回答
        "I think C is correct.",  # 完全没有 Answer
        "Answer: E",  # 不是 A–D
        "Answer: a",  # 小写字母不算（可能是英文冠词 a）
        "Answer: Doxycycline",  # 以 D 开头的单词，不能当成 D
        "Answer: Atrial fibrillation",  # 以 A 开头的单词
        "Answer: X",  # 照抄了格式说明里的占位符
        "Answer:",  # 被截断，只写了一半
        "The patient presents with ... and we must consider",  # 被截断的长推理
    ],
)
def test_extract_invalid(text):
    assert extract_answer(text) is None


def test_extract_handles_chat_format():
    # 对话格式的 completion：[{"role": "assistant", "content": ...}]
    completion = [{"role": "assistant", "content": "Reasoning...\nAnswer: B"}]
    assert completion_to_text(completion) == "Reasoning...\nAnswer: B"
    assert extract_answer(completion) == "B"


# ---------------------------------------------------------------------------
# 2. 单条打分：只有 0/1，格式错误 = 0
# ---------------------------------------------------------------------------
def test_score_correct_wrong_and_unformatted():
    assert score_completion("...\nAnswer: C", "C") == 1.0
    assert score_completion("...\nAnswer: C", "c") == 1.0  # 标准答案大小写不敏感
    assert score_completion("...\nAnswer: B", "C") == 0.0
    assert score_completion("The answer is C", "C") == 0.0  # 没按格式 → 0，不给部分分


# ---------------------------------------------------------------------------
# 3. GRPOTrainer 调用的打分函数 + 监控指标
# ---------------------------------------------------------------------------
EOS = 151645  # Qwen 的 <|im_end|> token id（这里只用于模拟）


def test_reward_function_batch_and_tracker():
    tracker = RewardTracker()
    logged = {}
    reward_fn = make_correctness_reward(tracker=tracker, max_completion_length=8, eos_token_ids=[EOS])

    # 模拟 2 道题 × 每题 4 个回答（num_generations = 4），TRL 会把同一道题的回答排在一起
    prompts = ["Q1"] * 4 + ["Q2"] * 4
    completions = [
        "...Answer: A",  # Q1 对
        "...Answer: B",  # Q1 错
        "...no answer",  # Q1 格式错误 → 0
        "...Answer: A",  # Q1 对
        "...Answer: C",  # Q2 错
        "...Answer: C",  # Q2 错
        "...Answer: C",  # Q2 错
        "...truncated",  # Q2 被截断（格式错误）→ 0
    ]
    answer = ["A"] * 4 + ["D"] * 4
    completion_ids = [
        [1, 2, 3, EOS],
        [1, 2, EOS],
        [1, 2, 3, 4, EOS],
        [1, EOS],
        [1, 2, EOS],
        [1, 2, 3, EOS],
        [1, EOS],
        [1, 2, 3, 4, 5, 6, 7, 8],  # 长度 = 8 且最后不是 EOS → 截断
    ]

    rewards = reward_fn(
        prompts=prompts,
        completions=completions,
        answer=answer,
        completion_ids=completion_ids,
        log_metric=lambda name, value: logged.__setitem__(name, value),
    )
    assert rewards == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert reward_fn.__name__ == "correctness_reward"
    assert logged["format_rate"] == pytest.approx(6 / 8)

    stats = tracker.pop_step_stats()
    assert stats["n_completions"] == 8
    assert stats["n_groups"] == 2
    assert stats["reward_mean"] == pytest.approx(2 / 8)
    assert stats["format_rate"] == pytest.approx(6 / 8)
    # Q1 有对有错（有学习信号），Q2 全错（advantage 全为 0）→ 全相同比例 = 1/2
    assert stats["frac_groups_all_same"] == pytest.approx(0.5)
    assert stats["frac_groups_all_wrong"] == pytest.approx(0.5)
    assert stats["frac_groups_all_correct"] == pytest.approx(0.0)
    assert stats["truncated_frac"] == pytest.approx(1 / 8)
    assert stats["mean_completion_tokens"] == pytest.approx(
        sum(len(x) for x in completion_ids) / 8
    )
    # pop 之后清空，下一步重新统计
    assert len(tracker) == 0


def test_reward_function_without_tracker_or_ids():
    # 不传 tracker / completion_ids 也能正常打分（例如训练前后的样例展示）
    reward_fn = make_correctness_reward()
    assert reward_fn(prompts=["Q"] * 2, completions=["Answer: A", "Answer: B"], answer=["B", "B"]) == [0.0, 1.0]
