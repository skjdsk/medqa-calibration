"""
train/grpo/rewards.py —— 从模型回答里提取答案，并打 0/1 分

【这个文件做什么（简单版）】
1. extract_answer(text)：在回答里找所有 "Answer: X"（X 是 A/B/C/D），取**最后一个**。
   找不到（没按格式回答）就返回 None。
2. correctness_reward：GRPO 用的打分函数。答案 == 标准答案 → 1.0，否则（包括格式错误）→ 0.0。
   reward 只有这一个正确性分数，不加任何格式奖励（与 proposal 一致）。
3. RewardTracker：打分时顺手记录监控指标（不影响 reward 本身）：
     - 格式正确率（能提取出答案的比例）
     - 每组（同一道题的 num_generations 个回答）是不是 reward 全相同
     - 回答长度（token 数）、是否被 max_completion_length 截断

【reward → advantage 是怎么算的（GRPO 的核心）】
同一道题让模型生成 G 个回答（G = num_generations，默认 4），每个回答得到 reward r_i ∈ {0, 1}。
GRPO 不训练额外的价值网络，而是用"组内比较"得到 advantage：
        advantage_i = (r_i - 组内平均 reward) / 组内 reward 标准差
例：4 个回答 reward = [1, 0, 0, 1] → 平均 0.5 → advantage ≈ [+1, -1, -1, +1]
    → 模型被推向更像第 1、4 个回答的方向，远离第 2、3 个。
如果 4 个回答全对 [1,1,1,1] 或全错 [0,0,0,0]，每个 r_i 都等于平均值 → advantage 全是 0
    → 这道题这一步对梯度没有贡献（"学不到东西"）。所以我们要监控"组内全相同比例"：
       比例太高（比如 > 0.8）说明信号太稀疏，可以考虑增大 num_generations 或调高 temperature。
（TRL 1.16 默认 scale_rewards="group"，即上式除以组内标准差；可用 --scale_rewards none 关掉除法。）

这个文件只用到 Python 标准库，可以在没有 GPU 的电脑上跑单元测试。
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import Any, Iterable, Sequence

VALID_LETTERS = ("A", "B", "C", "D")

# 匹配 "Answer: X"，宽松处理常见变体：
#   Answer: C / answer: C / ANSWER: C / Final Answer: C
#   **Answer:** C / **Answer: C** / Answer: (C) / Answer: C. xxx / Answer：C（全角冒号）
# 严格之处：
#   - 字母必须是大写 A–D（"Answer: a" 里的 a 很可能是英文冠词，不算）
#   - 字母后面不能紧跟字母或数字（"Answer: Doxycycline" 不会被当成 D）
#   - 必须有冒号（"The answer is C" 不算按格式回答）
# (?i:answer) 只让 "answer" 这个词不区分大小写，选项字母仍然必须大写。
_ANSWER_RE = re.compile(
    r"\b(?i:answer)\s*[*_]*\s*[:：]\s*[*_]*\s*\(?\s*([A-D])\s*\)?(?![A-Za-z0-9])"
)


def completion_to_text(completion: Any) -> str:
    """TRL 传给打分函数的 completion 可能是字符串，也可能是消息列表，这里统一转成字符串。

    - 字符串 prompt（我们的做法）→ completion 是字符串
    - 对话格式 prompt → completion 是 [{"role": "assistant", "content": "..."}]
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
    """返回回答中最后一个 "Answer: X" 的 X（A–D），找不到返回 None。"""
    text = completion_to_text(text)
    matches = _ANSWER_RE.findall(text)
    return matches[-1] if matches else None


def score_completion(completion: Any, gold: str) -> float:
    """单条回答打分：提取出的答案 == 标准答案 → 1.0，否则 0.0（含格式错误）。"""
    pred = extract_answer(completion)
    gold = str(gold).strip().upper()
    return 1.0 if (pred is not None and pred == gold) else 0.0


# ---------------------------------------------------------------------------
# 监控指标收集器
# ---------------------------------------------------------------------------
class RewardTracker:
    """在打分函数里收集每条回答的信息；训练回调每一步结束时取出并汇总。

    每条记录：group_key（同一道题的回答 key 相同）、reward、format_ok、num_tokens、truncated
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
        """汇总自上次调用以来的所有记录，然后清空。"""
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

    # 按题目分组（同一道题的 num_generations 个回答 group_key 相同）
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
        # 组内 reward 全相同（全对或全错）的比例：这些组 advantage = 0，对本步梯度无贡献
        "frac_groups_all_same": all_same / n_groups,
        "frac_groups_all_correct": all_correct / n_groups,
        "frac_groups_all_wrong": all_wrong / n_groups,
        "format_rate": _mean([1.0 if r["format_ok"] else 0.0 for r in records]),
        "mean_completion_tokens": _mean(lengths) if lengths else float("nan"),
        "max_completion_tokens": max(lengths) if lengths else float("nan"),
        "truncated_frac": _mean([1.0 if t else 0.0 for t in truncs]) if truncs else float("nan"),
    }


# ---------------------------------------------------------------------------
# 给 GRPOTrainer 用的打分函数
# ---------------------------------------------------------------------------
def make_correctness_reward(
    tracker: RewardTracker | None = None,
    max_completion_length: int | None = None,
    eos_token_ids: Iterable[int] | None = None,
    tokenizer=None,
):
    """构造 TRL GRPOTrainer 需要的 reward 函数。

    TRL 1.16 调用方式：reward_func(prompts=..., completions=..., completion_ids=..., **数据集其他列, ...)
      - prompts / completions：长度 = 本批回答数（每道题重复 num_generations 次，所以同题回答 prompt 相同）
      - answer：来自数据集的 "answer" 列，TRL 已按回答展开，和 completions 一一对应
      - completion_ids：每条回答的 token id 列表（用来算长度、判断截断）
      - log_metric：TRL 提供的记录标量指标的函数（我们用它把格式正确率也写进 TRL 日志）
    返回：list[float]，每条回答一个 reward。
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
                # 回答长度（token 数）和是否被截断
                num_tokens, truncated = None, None
                if completion_ids is not None:
                    ids = list(completion_ids[i])
                    num_tokens = len(ids)
                    if max_completion_length is not None:
                        # 与 TRL 的判断一致：最后一个 token 不是结束符 → 没写完就被截断了
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

        # 把格式正确率写进 TRL 自己的日志（TRL ≥ 1.x 提供 log_metric；老版本没有就跳过）
        log_metric = kwargs.get("log_metric")
        if callable(log_metric) and rewards:
            log_metric("format_rate", n_format_ok / len(rewards))
        return rewards

    # TRL 用函数名作为日志里的指标名，例如 "rewards/correctness_reward/mean"
    correctness_reward.__name__ = "correctness_reward"
    return correctness_reward
