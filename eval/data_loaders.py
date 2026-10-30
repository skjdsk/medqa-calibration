"""
data_loaders.py —— 加载三个评估数据集，统一成相同格式
=========================================================

【这个文件做什么】
MedQA / MedMCQA / PubMedQA 在 Hugging Face 上的字段名、选项格式、答案格式都不一样。
这个文件把它们全部转换成同一种结构 MCQItem：
    - dataset      数据集名，例如 "medqa"
    - question_id  每题唯一的 id
    - question     题干
    - options      选项文字列表，例如 ["Aspirin", "Heparin", ...]（第 0 个对应 A）
    - gold         标准答案字母，例如 "C"
    - context      额外材料（只有 PubMedQA 有：论文摘要）
这样 evaluate.py 不用关心数据集之间的差异。

【重要：数据集名称和字段名需要你确认】
下面 DATASET_DEFAULTS 里的 Hugging Face 名称/字段名是根据数据集卡片写的默认值，
但不同镜像版本可能不同。每次加载时代码都会打印一条“原始样例”和“转换后样例”，
请第一次运行时对照检查（详见 README / 回答末尾的确认清单）。
如需换成别的版本，用命令行参数 --hf_name / --hf_config / --split 覆盖即可，不用改代码。

单独运行本文件可以只看样例、不加载模型：
    python -m eval.data_loaders --dataset medqa --n 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

LETTERS = list("ABCDEFGHIJ")  # 最多支持 10 个选项

PUBMEDQA_LABELS = ["yes", "no", "maybe"]  # 对应 A / B / C


@dataclass
class MCQItem:
    """统一后的一道选择题。"""

    dataset: str
    question_id: str
    question: str
    options: list[str]
    gold: str  # 标准答案字母
    context: Optional[str] = None  # PubMedQA 的摘要；其他数据集为 None
    meta: dict = field(default_factory=dict)  # 其他信息（科目等），不参与评估

    @property
    def letters(self) -> list[str]:
        return LETTERS[: len(self.options)]


# ---------------------------------------------------------------------------
# 默认的数据集来源（需要你确认，见文件开头说明）
# ---------------------------------------------------------------------------
DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    # MedQA (USMLE, 英文)。4 选项和 5 选项来自不同的 HF 仓库，按 num_options 自动选择。
    "medqa": {
        4: dict(hf_name="GBaker/MedQA-USMLE-4-options", hf_config=None, split="test"),
        5: dict(hf_name="lavita/medical-qa-datasets", hf_config="med-qa-en-5options-source", split="test"),
    },
    # MedMCQA：测试集没有公开答案，所以用 validation（约 4183 题）
    "medmcqa": dict(hf_name="openlifescienceai/medmcqa", hf_config=None, split="validation"),
    # PubMedQA 专家标注子集：pqa_labeled 配置只有一个 train split，共 1000 题
    "pubmedqa": dict(hf_name="qiaojin/PubMedQA", hf_config="pqa_labeled", split="train"),
}


def resolve_source(dataset: str, num_options: Optional[int] = None,
                   hf_name: Optional[str] = None, hf_config: Optional[str] = None,
                   split: Optional[str] = None) -> dict[str, Any]:
    """决定实际从哪里加载：默认值 + 命令行覆盖。"""
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"未知数据集 {dataset!r}，可选：{list(DATASET_DEFAULTS)}")
    default = DATASET_DEFAULTS[dataset]
    if dataset == "medqa":
        n = num_options or 4
        if n not in default and hf_name is None:
            raise ValueError(f"MedQA 没有 {n} 选项的默认来源，请用 --hf_name 指定")
        default = default.get(n, dict(hf_name=None, hf_config=None, split="test"))
    src = dict(default)
    if hf_name is not None:
        src["hf_name"] = hf_name
        src["hf_config"] = hf_config  # 换仓库时 config 跟着换（可以是 None）
    elif hf_config is not None:
        src["hf_config"] = hf_config
    if split is not None:
        src["split"] = split
    return src


# ---------------------------------------------------------------------------
# 工具函数：把各种奇怪的选项格式变成 [(字母, 文字), ...]
# ---------------------------------------------------------------------------
def parse_options(raw: Any) -> list[tuple[str, str]]:
    """
    兼容几种常见格式：
      1) dict:   {"A": "xxx", "B": "yyy"}                  （GBaker/MedQA）
      2) list of dict: [{"key": "A", "value": "xxx"}, ...] （lavita / bigbio 风格）
      3) list of str:  ["xxx", "yyy"]                       （按顺序编号 A、B、C…）
    """
    if isinstance(raw, dict):
        # 有些 HF 版本会把 dict 存成 {"key": [...], "value": [...]} 这种“列式”格式
        if set(raw.keys()) >= {"key", "value"} and isinstance(raw["key"], list):
            pairs = list(zip(raw["key"], raw["value"]))
        else:
            pairs = list(raw.items())
        pairs = [(str(k).strip().upper(), str(v)) for k, v in pairs if v is not None]
        pairs.sort(key=lambda kv: kv[0])
        return pairs
    if isinstance(raw, list):
        if all(isinstance(x, dict) for x in raw):
            return [(str(x["key"]).strip().upper(), str(x["value"])) for x in raw]
        return [(LETTERS[i], str(x)) for i, x in enumerate(raw)]
    raise ValueError(f"无法识别的选项格式：{type(raw)} -> {str(raw)[:200]}")


def _truncate(v: Any, n: int = 300) -> Any:
    """打印样例时把很长的字符串截断，避免刷屏。"""
    if isinstance(v, str) and len(v) > n:
        return v[:n] + f"...<共 {len(v)} 字符>"
    if isinstance(v, dict):
        return {k: _truncate(x, n) for k, x in v.items()}
    if isinstance(v, list):
        return [_truncate(x, n) for x in v[:10]] + (["..."] if len(v) > 10 else [])
    return v


# ---------------------------------------------------------------------------
# 各数据集的“原始一行 → MCQItem”转换
# ---------------------------------------------------------------------------
def convert_medqa(row: dict, idx: int, num_options: int) -> MCQItem:
    pairs = parse_options(row["options"])
    if len(pairs) != num_options:
        # 不静默截断：截断可能把正确答案删掉，必须报错让你去换对应版本的数据集
        raise ValueError(
            f"MedQA 第 {idx} 题有 {len(pairs)} 个选项，但 --num_options={num_options}。"
            f"请换成对应选项数的数据集版本（--hf_name），或修改 --num_options。"
        )
    keys = [k for k, _ in pairs]
    texts = [v for _, v in pairs]

    # 标准答案：优先用 answer_idx（字母），没有的话用答案文字去匹配选项
    gold_pos = None
    if row.get("answer_idx") not in (None, ""):
        key = str(row["answer_idx"]).strip().upper()
        if key in keys:
            gold_pos = keys.index(key)
    if gold_pos is None and row.get("answer") not in (None, ""):
        ans = str(row["answer"]).strip()
        matches = [i for i, t in enumerate(texts) if t.strip() == ans]
        if len(matches) == 1:
            gold_pos = matches[0]
    if gold_pos is None:
        raise ValueError(f"MedQA 第 {idx} 题找不到标准答案：answer_idx={row.get('answer_idx')!r}, "
                         f"answer={row.get('answer')!r}, 选项 keys={keys}")

    # MedQA 没有 id 字段，用“在该 split 中的行号”作为 id（同一数据版本下稳定、可对齐）
    return MCQItem(
        dataset="medqa",
        question_id=f"medqa{num_options}-{idx}",
        question=str(row["question"]).strip(),
        options=texts,
        gold=LETTERS[gold_pos],  # 重新按位置编号成 A、B、C…，保证字母连续
        meta={"meta_info": row.get("meta_info")},
    )


def convert_medmcqa(row: dict, idx: int, **_) -> MCQItem:
    options = [str(row[k]) for k in ("opa", "opb", "opc", "opd")]
    cop = row["cop"]
    if isinstance(cop, str):
        cop = cop.strip().lower()
        cop = "abcd".index(cop) if cop in "abcd" and len(cop) == 1 else int(cop)
    cop = int(cop)
    if not 0 <= cop <= 3:
        raise ValueError(f"MedMCQA 第 {idx} 题 cop={cop} 不在 0–3 之间。"
                         f"如果你用的版本是 1–4 编号（原始 JSON 是这样），需要减 1。")
    return MCQItem(
        dataset="medmcqa",
        question_id=str(row.get("id", f"medmcqa-{idx}")),
        question=str(row["question"]).strip(),
        options=options,
        gold=LETTERS[cop],
        meta={"subject_name": row.get("subject_name"), "choice_type": row.get("choice_type")},
    )


def format_pubmedqa_context(ctx: Any) -> str:
    """
    PubMedQA 的 context 通常是：
        {"contexts": [段落1, 段落2, ...], "labels": ["BACKGROUND", "METHODS", ...], ...}
    也兼容“列表里每个元素是一个 dict”的版本，以及直接是字符串的版本。
    输出：每段一行，前面加上段落标签，例如 "BACKGROUND: ..."。
    """
    if isinstance(ctx, str):
        return ctx.strip()
    if isinstance(ctx, dict):
        paras = ctx.get("contexts") or []
        labels = ctx.get("labels") or []
    elif isinstance(ctx, list):
        paras = [c.get("contexts", "") for c in ctx]
        labels = [c.get("labels", "") for c in ctx]
    else:
        raise ValueError(f"无法识别的 PubMedQA context 格式：{type(ctx)}")
    lines = []
    for i, p in enumerate(paras):
        lab = labels[i] if i < len(labels) and labels[i] else ""
        lines.append(f"{lab}: {p.strip()}" if lab else p.strip())
    return "\n".join(lines)


def convert_pubmedqa(row: dict, idx: int, **_) -> MCQItem:
    decision = str(row["final_decision"]).strip().lower()
    if decision not in PUBMEDQA_LABELS:
        raise ValueError(f"PubMedQA 第 {idx} 题 final_decision={decision!r} 不是 yes/no/maybe")
    return MCQItem(
        dataset="pubmedqa",
        question_id=str(row.get("pubid", f"pubmedqa-{idx}")),
        question=str(row["question"]).strip(),
        options=list(PUBMEDQA_LABELS),  # A. yes / B. no / C. maybe
        gold=LETTERS[PUBMEDQA_LABELS.index(decision)],
        context=format_pubmedqa_context(row["context"]),
    )


CONVERTERS = {"medqa": convert_medqa, "medmcqa": convert_medmcqa, "pubmedqa": convert_pubmedqa}


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def load_items(dataset: str, n: Optional[int] = None, num_options: Optional[int] = None,
               hf_name: Optional[str] = None, hf_config: Optional[str] = None,
               split: Optional[str] = None, show_sample: bool = True,
               indices: Optional[list[int]] = None) -> list[MCQItem]:
    """
    加载一个数据集并转换为 MCQItem 列表。
    n：只取前 n 题（调试用）；None 表示全部。
    num_options：只对 MedQA 有意义（4 或 5）。
    indices：只取这些行（例如 data/medqa_split_seed42.json 里的验证集行号）；question_id 仍用原来的行号。
    """
    from datasets import load_dataset  # 延迟导入：跑单元测试时不需要安装 datasets

    if dataset == "medqa":
        num_options = num_options or 4
    src = resolve_source(dataset, num_options, hf_name, hf_config, split)
    print(f"[data] 加载 {dataset}: hf_name={src['hf_name']}, config={src['hf_config']}, split={src['split']}")
    ds = load_dataset(src["hf_name"], src["hf_config"], split=src["split"])
    print(f"[data] 共 {len(ds)} 行；字段：{ds.column_names}")
    row_ids = list(range(len(ds)))
    if indices is not None:
        ds = ds.select(indices)
        row_ids = list(indices)
        print(f"[data] 按行号列表取 {len(ds)} 行")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
        row_ids = row_ids[: len(ds)]

    if show_sample and len(ds) > 0:
        print("[data] ===== 原始样例（第 0 行，请核对字段名）=====")
        print(json.dumps(_truncate(dict(ds[0])), ensure_ascii=False, indent=2, default=str))

    convert = CONVERTERS[dataset]
    items = [convert(dict(row), i, num_options=num_options) for i, row in zip(row_ids, ds)]

    if show_sample and items:
        print("[data] ===== 转换后样例（请核对选项和标准答案是否对得上）=====")
        print(json.dumps(_truncate(asdict(items[0])), ensure_ascii=False, indent=2, default=str))
        # 打印答案分布：如果某个字母从来不出现，或者全都是同一个字母，说明可能解析错了
        dist: dict[str, int] = {}
        for it in items:
            dist[it.gold] = dist.get(it.gold, 0) + 1
        print(f"[data] 标准答案分布：{dict(sorted(dist.items()))}")

    ids = [it.question_id for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{dataset} 中存在重复的 question_id，无法用于跨模型对齐")
    return items


def main():
    p = argparse.ArgumentParser(description="只加载并打印数据集样例，不跑模型")
    p.add_argument("--dataset", required=True, choices=list(DATASET_DEFAULTS))
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--num_options", type=int, default=None)
    p.add_argument("--hf_name", default=None)
    p.add_argument("--hf_config", default=None)
    p.add_argument("--split", default=None)
    a = p.parse_args()
    load_items(a.dataset, a.n, a.num_options, a.hf_name, a.hf_config, a.split, show_sample=True)


if __name__ == "__main__":
    main()
