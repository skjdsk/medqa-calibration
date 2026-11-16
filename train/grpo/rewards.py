"""
train/grpo/rewards.py -- pull the answer out of a model response and score it 0/1

[What this file does]
1. extract_answer(text): finds every "Answer: X" (X is A/B/C/D) in the response and keeps the **last** one.
      Returns None if there isn't one (the model didn't follow the format).
2. correctness_reward: the GRPO reward. Right answer -> 1.0, anything else (format errors included) -> 0.0.
      That's the only reward. No format bonus (same as the proposal).
3. RewardTracker: logs a few things to watch while scoring (doesn't affect the reward):
          - format rate (how often we can pull out an answer)
          - whether every answer in a group (the num_generations answers to one question) got the same reward
          - answer length in tokens, and whether it got cut off at max_completion_length

[How reward turns into advantage (the core of GRPO)]
For each question the model writes G answers (G = num_generations, default 4), each with reward r_i in {0, 1}.
GRPO doesn't train a value network. It compares answers within the group instead:
                advantage_i = (r_i - group mean reward) / group reward std
e.g. 4 answers with reward [1, 0, 0, 1] -> mean 0.5 -> advantage ~ [+1, -1, -1, +1]
        -> the model gets pushed toward answers 1 and 4 and away from 2 and 3.
If all 4 are right [1,1,1,1] or all wrong [0,0,0,0], every r_i equals the mean -> all advantages are 0
        -> that question adds nothing to the gradient this step. That's why we track the share of all-same groups:
              if it's too high (say > 0.8) the signal is too sparse; try more num_generations or a higher temperature.
(TRL 1.16 defaults to scale_rewards="group", i.e. divide by the group std as above; --scale_rewards none turns that off.)

Only uses the Python standard library, so the unit tests run on a machine without a GPU.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import Any, Iterable, Sequence

VALID_LETTERS = ("A", "B", "C", "D")

# Matches "Answer: X" and tolerates common variants:
#   Answer: C / answer: C / ANSWER: C / Final Answer: C
#   **Answer:** C / **Answer: C** / Answer: (C) / Answer: C. xxx / Answer：C (full-width colon)
# What it's strict about:
#   - the letter has to be uppercase A-D (the "a" in "Answer: a" is probably just the article)
#   - the letter can't be followed by a letter or digit ("Answer: Doxycycline" is not D)
#   - there has to be a colon ("The answer is C" doesn't count as following the format)
# (?i:answer) makes only the word "answer" case-insensitive; the option letter still has to be uppercase.
_ANSWER_RE = re.compile(
    r"\b(?i:answer)\s*[*_]*\s*[:：]\s*[*_]*\s*\(?\s*([A-D])\s*\)?(?![A-Za-z0-9])"
)


def completion_to_text(completion: Any) -> str:
    """TRL may hand the reward function a completion as a string or as a list of messages; turn it into a string.

    - string prompt (what we do) -> completion is a string
    - chat-format prompt -> completion is [{"role": "assistant", "content": "..."}]
    """
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, (list, tuple)):
        parts = []
        for msg in completion:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content", "")))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    return str(completion)


def extract_answer(text: Any) -> str | None:
    """Return X (A-D) from the last "Answer: X" in the response, or None if there isn't one."""
    text = completion_to_text(text)
    matches = _ANSWER_RE.findall(text)
    return matches[-1] if matches else None


def score_completion(completion: Any, gold: str) -> float:
    """Score one response: extracted answer == gold -> 1.0, otherwise 0.0 (format errors included)."""
    pred = extract_answer(completion)
    gold = str(gold).strip().upper()
    return 1.0 if (pred is not None and pred == gold) else 0.0


# ---------------------------------------------------------------------------
# Metric tracker
# ---------------------------------------------------------------------------
class RewardTracker:
    """Collects per-answer info inside the reward function; a training callback pulls and summarizes it after each step.

    Each record: group_key (same for answers to the same question), reward, format_ok, num_tokens, truncated
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def add(
        self,
        group_key: str,
        reward: float,
        format_ok: bool,
        num_tokens: int | None,
        truncated: bool | None,
    ) -> None:
        self._records.append(
            {
                "group_key": group_key,
                "reward": float(reward),
                "format_ok": bool(format_ok),
                "num_tokens": num_tokens,
                "truncated": truncated,
            }
        )

    def __len__(self) -> int:
        return len(self._records)

    def pop_step_stats(self) -> dict[str, Any]:
        """Summarize every record since the last call, then clear them."""
        records, self._records = self._records, []
        return summarize_records(records)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _pstd(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    if not records:
        return {"n_completions": 0}

    rewards = [r["reward"] for r in records]

    # Group by question (the num_generations answers to one question share a group_key)
    groups: "OrderedDict[str, list[float]]" = OrderedDict()
    for r in records:
        groups.setdefault(r["group_key"], []).append(r["reward"])
    n_groups = len(groups)
    all_same = sum(1 for g in groups.values() if len(set(g)) == 1)
    all_correct = sum(1 for g in groups.values() if all(x == 1.0 for x in g))
    all_wrong = sum(1 for g in groups.values() if all(x == 0.0 for x in g))

    lengths = [r["num_tokens"] for r in records if r["num_tokens"] is not None]
    truncs = [r["truncated"] for r in records if r["truncated"] is not None]

    return {
        "n_completions": len(records),
        "n_groups": n_groups,
        "reward_mean": _mean(rewards),
        "reward_std": _pstd(rewards),
        # Share of groups where every reward is the same (all right or all wrong): advantage = 0, no gradient this step
        "frac_groups_all_same": all_same / n_groups,
        "frac_groups_all_correct": all_correct / n_groups,
        "frac_groups_all_wrong": all_wrong / n_groups,
        "format_rate": _mean([1.0 if r["format_ok"] else 0.0 for r in records]),
        "mean_completion_tokens": _mean(lengths) if lengths else float("nan"),
        "max_completion_tokens": max(lengths) if lengths else float("nan"),
        "truncated_frac": _mean([1.0 if t else 0.0 for t in truncs]) if truncs else float("nan"),
    }


# ---------------------------------------------------------------------------
# Reward function for GRPOTrainer
# ---------------------------------------------------------------------------
def make_correctness_reward(
    tracker: RewardTracker | None = None,
    max_completion_length: int | None = None,
    eos_token_ids: Iterable[int] | None = None,
    tokenizer=None,
):
    """Build the reward function TRL's GRPOTrainer expects.

    TRL 1.16 calls it as reward_func(prompts=..., completions=..., completion_ids=..., **other dataset columns, ...)
        - prompts / completions: one entry per answer in the batch (each question repeated num_generations times, so answers to the same question share a prompt)
        - answer: the dataset's "answer" column, already expanded by TRL to line up with completions
        - completion_ids: token ids for each answer (used for length and truncation)
        - log_metric: TRL's function for logging a scalar (we use it to put the format rate in TRL's logs too)
    Returns list[float], one reward per answer.
    """
    eos_set = set(eos_token_ids) if eos_token_ids is not None else None

    def correctness_reward(prompts, completions, answer, completion_ids=None, **kwargs) -> list[float]:
        rewards: list[float] = []
        n_format_ok = 0
        for i, (prompt, completion, gold) in enumerate(zip(prompts, completions, answer)):
            text = completion_to_text(completion)
            pred = extract_answer(text)
            r = 1.0 if (pred is not None and pred == str(gold).strip().upper()) else 0.0
            rewards.append(r)
            n_format_ok += pred is not None

            if tracker is not None:
                # answer length in tokens and whether it got cut off
                num_tokens, truncated = None, None
                if completion_ids is not None:
                    ids = list(completion_ids[i])
                    num_tokens = len(ids)
                    if max_completion_length is not None:
                        # same check TRL uses: last token isn't EOS -> it got cut off before finishing
                        if eos_set is not None and ids:
                            truncated = ids[-1] not in eos_set
                        else:
                            truncated = num_tokens >= max_completion_length
                elif tokenizer is not None:
                    num_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                    if max_completion_length is not None:
                        truncated = num_tokens >= max_completion_length
                group_key = completion_to_text(prompt)
                tracker.add(group_key, r, pred is not None, num_tokens, truncated)

        # put the format rate in TRL's own logs (TRL >= 1.x has log_metric; skip it on older versions)
        log_metric = kwargs.get("log_metric")
        if callable(log_metric) and rewards:
            log_metric("format_rate", n_format_ok / len(rewards))
        return rewards

    # TRL uses the function name as the metric name in its logs, e.g. "rewards/correctness_reward/mean"
    correctness_reward.__name__ = "correctness_reward"
    return correctness_reward
