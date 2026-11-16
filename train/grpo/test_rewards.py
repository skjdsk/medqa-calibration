"""
train/grpo/test_rewards.py -- unit tests for the reward function

[What this file does]
Feeds a bunch of fake model answers through answer extraction and scoring, e.g.:
    - standard format, markdown bold, parentheses, full-width colon, lowercase 'answer', etc.
    - when an answer has several "Answer:" lines, we take the last one
    - no format / an E / "Answer: Doxycycline" all count as extraction failures (reward = 0)
    - the tracked metrics (format rate, all-same group share, truncation rate) come out right

Run from the repo root:
    python -m pytest -q train/grpo/test_rewards.py
No GPU and no model download needed.
"""

import sys
from pathlib import Path

import pytest

# make "from train.grpo.rewards import ..." work however this gets run
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.grpo.rewards import (  # noqa: E402
    RewardTracker,
    completion_to_text,
    extract_answer,
    make_correctness_reward,
    score_completion,
)


# ---------------------------------------------------------------------------
# 1. Answer extraction: all kinds of formats
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        # standard format
        ("The patient has hypertension...\nAnswer: C", "C"),
        ("Answer: A", "A"),
        ("Reasoning...\nAnswer:B", "B"),  # no space after the colon
        ("Step 1 ...\nAnswer: D\n", "D"),  # extra newline at the end
        # case / prefix variants
        ("answer: B", "B"),
        ("ANSWER: C", "C"),
        ("Final Answer: A", "A"),
        # markdown / parentheses / punctuation
        ("**Answer:** D", "D"),
        ("**Answer: B**", "B"),
        ("Answer: **C**", "C"),
        ("Answer: (A)", "A"),
        ("Answer: (C) Hypertension", "C"),
        ("Answer: B. Metformin", "B"),
        ("Answer: D)", "D"),
        ("Answer：C", "C"),  # full-width colon
        # several Answers: take the last one
        ("At first I thought Answer: A, but on reflection...\nAnswer: C", "C"),
        ("I will end with 'Answer: X'.\nReasoning...\nAnswer: B", "B"),
        # option followed by a comma: take the first letter (the model is hedging; by convention we take what it wrote first)
        ("Answer: A, B", "A"),
    ],
)
def test_extract_valid(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",  # empty answer
        "The answer is C.",  # no colon: doesn't follow the format
        "I think C is correct.",  # no Answer at all
        "Answer: E",  # not A-D
        "Answer: a",  # lowercase doesn't count (could be the article "a")
        "Answer: Doxycycline",  # a word starting with D is not D
        "Answer: Atrial fibrillation",  # a word starting with A
        "Answer: X",  # copied the placeholder from the instructions
        "Answer:",  # cut off halfway
        "The patient presents with ... and we must consider",  # long reasoning that got cut off
    ],
)
def test_extract_invalid(text):
    assert extract_answer(text) is None


def test_extract_handles_chat_format():
    # chat-format completion: [{"role": "assistant", "content": ...}]
    completion = [{"role": "assistant", "content": "Reasoning...\nAnswer: B"}]
    assert completion_to_text(completion) == "Reasoning...\nAnswer: B"
    assert extract_answer(completion) == "B"


# ---------------------------------------------------------------------------
# 2. Scoring one answer: only 0/1, format error = 0
# ---------------------------------------------------------------------------
def test_score_correct_wrong_and_unformatted():
    assert score_completion("...\nAnswer: C", "C") == 1.0
    assert score_completion("...\nAnswer: C", "c") == 1.0  # gold letter is case-insensitive
    assert score_completion("...\nAnswer: B", "C") == 0.0
    assert score_completion("The answer is C", "C") == 0.0  # wrong format -> 0, no partial credit


# ---------------------------------------------------------------------------
# 3. The reward function GRPOTrainer calls + tracked metrics
# ---------------------------------------------------------------------------
EOS = 151645  # Qwen's <|im_end|> token id (only used to fake things here)


def test_reward_function_batch_and_tracker():
    tracker = RewardTracker()
    logged = {}
    reward_fn = make_correctness_reward(tracker=tracker, max_completion_length=8, eos_token_ids=[EOS])

    # fake 2 questions x 4 answers each (num_generations = 4); TRL keeps answers to the same question together
    prompts = ["Q1"] * 4 + ["Q2"] * 4
    completions = [
        "...Answer: A",  # Q1 right
        "...Answer: B",  # Q1 wrong
        "...no answer",  # Q1 format error -> 0
        "...Answer: A",  # Q1 right
        "...Answer: C",  # Q2 wrong
        "...Answer: C",  # Q2 wrong
        "...Answer: C",  # Q2 wrong
        "...truncated",  # Q2 cut off (format error) -> 0
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
        [1, 2, 3, 4, 5, 6, 7, 8],  # length = 8 and last token isn't EOS -> truncated
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
    # Q1 is mixed (there's signal), Q2 is all wrong (advantages all 0) -> all-same share = 1/2
    assert stats["frac_groups_all_same"] == pytest.approx(0.5)
    assert stats["frac_groups_all_wrong"] == pytest.approx(0.5)
    assert stats["frac_groups_all_correct"] == pytest.approx(0.0)
    assert stats["truncated_frac"] == pytest.approx(1 / 8)
    assert stats["mean_completion_tokens"] == pytest.approx(
        sum(len(x) for x in completion_ids) / 8
    )
    # pop clears it so the next step starts fresh
    assert len(tracker) == 0


def test_reward_function_without_tracker_or_ids():
    # scoring still works without a tracker / completion_ids (e.g. showing samples before and after training)
    reward_fn = make_correctness_reward()
    assert reward_fn(prompts=["Q"] * 2, completions=["Answer: A", "Answer: B"], answer=["B", "B"]) == [0.0, 1.0]
