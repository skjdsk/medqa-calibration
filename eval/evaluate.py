"""
evaluate.py —— 加载模型、构造 prompt、计算每个选项的概率、输出每题明细 CSV
===========================================================================

【这个文件做什么】
1. 用 4-bit（nf4）加载 Qwen2.5-3B-Instruct；如果给了 --adapter，再用 PEFT 把 LoRA 叠上去。
2. 把每道题放进 Qwen 的 chat template（user 消息里写题目和选项，要求只回答字母），
   并加上 assistant 的开头（add_generation_prompt=True）。
3. 不生成文字，只做一次前向计算，拿“下一个 token”的打分（logits）。
4. 只看选项字母（A/B/C…）对应的 token：
     - 每个字母同时考虑 "A" 和 " A"（带空格）两种 token，用 logsumexp 合并（= 两者概率相加）
     - 记录这些字母在整个词表上的概率总和 letter_mass（归一化前），用来检查 prompt 是否合理：
       如果 letter_mass 很小（比如 < 0.5），说明模型其实想输出别的东西（比如 "The"、"**"），
       这时字母概率就不太可信，需要改 prompt。
     - 在这几个字母上做 softmax，得到和为 1 的选项概率
5. 概率最大的选项 = 模型答案，它的概率 = confidence。每题一行写进 CSV。

【运行示例】
    # 原始模型，MedQA 前 50 题
    python -m eval.evaluate --dataset medqa --n 50 --out_dir results/
    # 带 LoRA adapter
    python -m eval.evaluate --dataset medqa --adapter /path/to/adapter --model_name sft_seed1 --out_dir results/

【CoT 扩展空间】
--mode 目前只实现 "direct"。之后加 CoT 的思路（已在代码里留好位置）：
    先 generate 出推理过程 → 在推理后面接上 "Final answer:" 之类的前缀 →
    再调用同一个 score_letters()（它支持 assistant_prefixes 参数）给字母打分。
这样 CoT 和直接作答共享完全相同的概率计算和指标代码，结果可直接比较。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch

from eval.data_loaders import LETTERS, MCQItem, load_items

BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"
PROMPT_VERSION = "direct-v1"  # 改动 prompt 时把版本号也改掉，CSV 里会记录，防止混用不同 prompt 的结果


# ---------------------------------------------------------------------------
# 0. 固定随机种子
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 只做前向计算本身没有随机性；这里主要是让以后加采样（CoT）时可复现


# ---------------------------------------------------------------------------
# 1. 加载模型（4-bit nf4 + 可选 LoRA）
# ---------------------------------------------------------------------------
def pick_compute_dtype(name: str) -> torch.dtype:
    if name == "auto":
        # A100/L4 支持 bf16；Colab 免费的 T4 不支持 bf16，用 fp16
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_model_and_tokenizer(base_model: str, adapter: Optional[str], compute_dtype: torch.dtype,
                             double_quant: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # 量化设置要和训练时 QLoRA 的 BitsAndBytesConfig 一致（nf4、double quant、compute dtype）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    # tokenizer 优先从 adapter 目录读取（训练时如果改过 tokenizer 会保存在那里），否则用基础模型的
    tok_src = adapter if adapter and os.path.exists(os.path.join(adapter, "tokenizer_config.json")) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    tokenizer.padding_side = "left"  # decoder-only 模型批处理必须左侧 padding，最后一个位置才是真实 token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 新版 transformers（≥4.56，包括 5.x）把参数名 torch_dtype 改成了 dtype
    import transformers
    from packaging import version
    dtype_kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        **{dtype_kw: compute_dtype},
    )
    if adapter:
        from peft import PeftModel

        print(f"[model] 叠加 LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# 2. 构造 prompt
# ---------------------------------------------------------------------------
def build_user_message(item: MCQItem) -> str:
    """把一道题变成 user 消息的文字。所有模型、所有数据集都用这一个函数，保证可比。"""
    letters = item.letters
    option_lines = "\n".join(f"{L}. {opt}" for L, opt in zip(letters, item.options))
    letter_list = ", ".join(letters[:-1]) + f", or {letters[-1]}"
    parts = []
    if item.context:  # PubMedQA：先给摘要
        parts.append(f"Context:\n{item.context}")
    parts.append(f"Question: {item.question}")
    parts.append(f"Options:\n{option_lines}")
    parts.append(f"Answer with only the letter of the correct option ({letter_list}). Do not explain.")
    return "\n\n".join(parts)


def build_prompt_text(item: MCQItem, tokenizer, system_prompt: Optional[str] = None,
                      assistant_prefix: str = "") -> str:
    """
    用 Qwen 的 chat template 生成完整输入文字。
    add_generation_prompt=True 会在末尾加上 "<|im_start|>assistant\\n"，
    所以模型下一个要输出的 token 就是它的回答开头 —— 我们正是要看这个位置字母的概率。
    assistant_prefix：留给 CoT 用（把生成的推理 + "Final answer:" 接在后面）；直接作答时为空。
    注意：不传 system_prompt 时，Qwen2.5 的模板会自动加默认 system 消息
    （"You are Qwen, created by Alibaba Cloud..."）。训练数据如果用了别的 system 消息，评估时请保持一致。
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": build_user_message(item)})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + assistant_prefix


# ---------------------------------------------------------------------------
# 3. 选项字母 → token id（检查是否单 token，处理 "A" 和 " A"）
# ---------------------------------------------------------------------------
def build_letter_token_table(tokenizer, letters: list[str]) -> dict[str, list[int]]:
    """
    对每个字母，找出 "A" 和 " A" 各自的 token id。
    - 只有“编码后正好是 1 个 token”的写法才算数（多 token 的写法无法用一步 logits 表示）
    - 两种写法如果是同一个 id，只保留一个，避免重复计算
    - 检查不同字母之间没有共用 id
    返回：{"A": [id1, id2], "B": [...], ...}
    """
    table: dict[str, list[int]] = {}
    for L in letters:
        ids = []
        for variant in (L, " " + L):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1 and toks[0] != tokenizer.unk_token_id:  # 排除“未知词” token
                ids.append(toks[0])
            else:
                print(f"[warn] {variant!r} 被切成 {len(toks)} 个 token {toks}，这个写法不计入")
        ids = sorted(set(ids))
        if not ids:
            raise RuntimeError(f"字母 {L!r} 没有任何单 token 写法，无法用 logits 打分")
        table[L] = ids
    all_ids = [i for ids in table.values() for i in ids]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError(f"不同字母共用了 token id，请检查：{table}")
    return table


# ---------------------------------------------------------------------------
# 4. 前向计算，只取最后一个位置的 logits
# ---------------------------------------------------------------------------
def last_position_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    返回形状 [batch, vocab] 的最后位置 logits。
    - position_ids 根据 attention_mask 计算：左侧 padding 时，真实 token 的位置编号仍从 0 开始，
      这样同一道题无论和谁一起成批、padding 多少，结果都一样（和 model.generate 的做法一致）。
    - logits_to_keep=1：只计算最后一个位置的 logits。否则 [batch × 序列长度 × 15 万词表]
      会占用好几 GB 显存（PubMedQA 摘要较长时尤其明显）。旧版 transformers 叫 num_logits_to_keep。
    """
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
    keep_arg = _logits_keep_arg(model)
    if keep_arg:
        kwargs[keep_arg] = 1
    out = model(**kwargs)
    return out.logits[:, -1, :]


def _logits_keep_arg(model) -> Optional[str]:
    """看当前 transformers 版本的 forward 支持哪个参数名（PEFT 包装时看里面的基础模型）。"""
    import inspect

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    params = inspect.signature(base.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None  # 很旧的版本：计算全部位置的 logits，结果一样，只是更占显存


@torch.no_grad()
def score_letters(model, tokenizer, items: list[MCQItem], letter_table: dict[str, list[int]],
                  batch_size: int = 8, system_prompt: Optional[str] = None,
                  assistant_prefixes: Optional[list[str]] = None) -> list[dict]:
    """
    对每道题算出每个选项字母的概率。返回与 items 同顺序的字典列表：
        {"letter_probs": {"A": 0.7, ...}, "letter_mass": 0.98, "n_prompt_tokens": 312}
    assistant_prefixes：CoT 预留，每题一个前缀字符串；直接作答时为 None。
    """
    from tqdm.auto import tqdm

    device = next(model.parameters()).device
    prompts = [
        build_prompt_text(it, tokenizer, system_prompt, assistant_prefixes[i] if assistant_prefixes else "")
        for i, it in enumerate(items)
    ]
    # 按长度从长到短排序再分批：减少 padding 浪费；最长的先跑，如果会爆显存第一批就会发现
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(items)), key=lambda i: -lengths[i])
    results: list[Optional[dict]] = [None] * len(items)

    for start in tqdm(range(0, len(order), batch_size), desc="scoring"):
        idx = order[start:start + batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False)  # chat template 已经包含所有特殊 token
        logits = last_position_logits(model, enc["input_ids"].to(device), enc["attention_mask"].to(device))
        logits = logits.float()  # 转成 fp32 再做 softmax，避免 fp16 精度问题

        # ---- 关键步骤 1：整个词表上的 log 概率 ----
        # log_softmax 后，log_probs_full[b, t] = log P(下一个 token = t)
        log_probs_full = torch.log_softmax(logits, dim=-1)  # [B, V]

        for row, i in enumerate(idx):
            letters = items[i].letters  # 这道题实际有几个选项就取几个字母（PubMedQA 只有 A/B/C）

            # ---- 关键步骤 2：每个字母合并 "A" 和 " A" ----
            # logsumexp(log p1, log p2) = log(p1 + p2)，即两种写法的概率相加
            letter_logp = torch.stack([
                torch.logsumexp(log_probs_full[row, letter_table[L]], dim=0) for L in letters
            ])  # [K]，K = 选项数

            # ---- 关键步骤 3：归一化前的“字母概率总和” ----
            # 这些字母在整个词表上一共占多少概率；越接近 1 说明模型确实在“选字母”
            letter_mass = letter_logp.exp().sum().item()

            # ---- 关键步骤 4：只在这 K 个字母上做 softmax ----
            # softmax(log p_A, log p_B, ...) = p_A / (p_A + p_B + ...)，即把字母概率重新归一化成和为 1
            # （与“直接对这些字母的原始 logits 做 logsumexp 后 softmax”数学上完全等价）
            norm_probs = torch.softmax(letter_logp, dim=0).cpu().numpy()

            results[i] = {
                "letter_probs": {L: float(p) for L, p in zip(letters, norm_probs)},
                "letter_mass": float(letter_mass),
                "n_prompt_tokens": int(lengths[i]),
            }
    return results  # type: ignore[return-value]


def score_cot(*args, **kwargs):
    """
    CoT 评估预留位置（暂不实现）。计划：
      1. 用 model.generate 生成推理（固定种子、记录解码参数）；
      2. 截取推理文本，接上统一的答案前缀，如 "\\nFinal answer:"；
      3. 调用 score_letters(..., assistant_prefixes=[推理 + 前缀, ...]) 得到字母概率；
      4. 结果写到 eval_mode="cot" 的 CSV，metrics.py 无需修改即可计算指标。
    """
    raise NotImplementedError("CoT 评估还没实现，目前只支持 --mode direct")


# ---------------------------------------------------------------------------
# 5. 组装每题明细表
# ---------------------------------------------------------------------------
def build_rows(items: list[MCQItem], scores: list[dict], model_name: str, eval_mode: str,
               max_letters: int) -> pd.DataFrame:
    rows = []
    for it, sc in zip(items, scores):
        probs = sc["letter_probs"]
        pred = max(probs, key=probs.get)  # 概率最大的字母（并列时取靠前的）
        row = {
            "dataset": it.dataset,
            "question_id": it.question_id,
            "model_name": model_name,
            "eval_mode": eval_mode,
            "n_options": len(it.options),
            "gold": it.gold,
            "pred": pred,
            "confidence": probs[pred],
            "correct": int(pred == it.gold),
            "letter_mass": sc["letter_mass"],  # 归一化前字母概率总和
            "n_prompt_tokens": sc["n_prompt_tokens"],
        }
        # 每个选项的归一化概率；这道题没有的选项（如 PubMedQA 的 D）留空
        for L in LETTERS[:max_letters]:
            row[f"prob_{L}"] = probs.get(L, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


# ---------------------------------------------------------------------------
# 6. 命令行入口
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="多选题校准评估（直接作答，letter-logit 打分）")
    p.add_argument("--dataset", required=True, nargs="+", choices=["medqa", "medmcqa", "pubmedqa"],
                   help="可以一次给多个，例如 --dataset medqa medmcqa pubmedqa")
    p.add_argument("--base_model", default=BASE_MODEL_DEFAULT)
    p.add_argument("--adapter", default=None, help="LoRA adapter 目录；不给则评估原始模型")
    p.add_argument("--model_name", default=None, help="写进 CSV 的模型名；默认 base 或 adapter 文件夹名")
    p.add_argument("--n", type=int, default=None, help="只跑前 N 题（调试用）")
    p.add_argument("--num_options", type=int, default=4, help="MedQA 的选项数（4 或 5）")
    p.add_argument("--hf_name", default=None, help="覆盖数据集的 HF 名称（只在单个 --dataset 时使用）")
    p.add_argument("--hf_config", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--compute_dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                   help="4-bit 计算精度；请和训练时 bnb_4bit_compute_dtype 一致")
    p.add_argument("--no_double_quant", action="store_true", help="训练时没开 double quant 就加上这个")
    p.add_argument("--system_prompt", default=None, help="默认用 Qwen 模板自带的 system 消息")
    p.add_argument("--mode", default="direct", choices=["direct", "cot"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--check_batch_consistency", action="store_true",
                   help="额外把前 4 题用 batch_size=1 重算，确认批处理/padding 不影响结果")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    if args.mode == "cot":
        score_cot()
    if args.hf_name and len(args.dataset) > 1:
        raise SystemExit("--hf_name 只能在单个 --dataset 时使用")

    model_name = args.model_name or (
        os.path.basename(os.path.normpath(args.adapter)) if args.adapter else "base")
    model_name = sanitize(model_name)
    os.makedirs(args.out_dir, exist_ok=True)

    compute_dtype = pick_compute_dtype(args.compute_dtype)
    print(f"[model] base={args.base_model} adapter={args.adapter} dtype={compute_dtype} name={model_name}")
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter, compute_dtype,
                                                double_quant=not args.no_double_quant)

    for ds_name in args.dataset:
        t0 = time.time()
        items = load_items(ds_name, n=args.n, num_options=args.num_options if ds_name == "medqa" else None,
                           hf_name=args.hf_name, hf_config=args.hf_config, split=args.split)
        max_letters = max(len(it.options) for it in items)
        letter_table = build_letter_token_table(tokenizer, LETTERS[:max_letters])
        print(f"[tokens] 字母 → token id：{letter_table}")
        print("[prompt] ===== 第一题的完整输入（请检查格式）=====")
        print(build_prompt_text(items[0], tokenizer, args.system_prompt))
        print("[prompt] ===== 结束 =====")

        scores = score_letters(model, tokenizer, items, letter_table, args.batch_size, args.system_prompt)

        if args.check_batch_consistency:
            k = min(4, len(items))
            single = score_letters(model, tokenizer, items[:k], letter_table, 1, args.system_prompt)
            diff = max(abs(single[i]["letter_probs"][L] - scores[i]["letter_probs"][L])
                       for i in range(k) for L in single[i]["letter_probs"])
            print(f"[check] batch_size={args.batch_size} vs 1 的最大概率差：{diff:.2e}"
                  f"（4-bit + fp16 下 1e-3 量级以内属正常）")

        df = build_rows(items, scores, model_name, args.mode, max_letters)
        tag = f"{ds_name}{args.num_options}" if ds_name == "medqa" else ds_name
        csv_path = os.path.join(args.out_dir, f"preds_{model_name}_{tag}_{args.mode}.csv")
        df.to_csv(csv_path, index=False)

        # 记录本次运行的设置，方便以后核对 7 个模型是否用了完全相同的设置
        meta = {
            "model_name": model_name, "base_model": args.base_model, "adapter": args.adapter,
            "dataset": ds_name, "n_items": len(items), "num_options": args.num_options if ds_name == "medqa" else None,
            "hf_source": dict(hf_name=args.hf_name, hf_config=args.hf_config, split=args.split),
            "mode": args.mode, "prompt_version": PROMPT_VERSION, "system_prompt": args.system_prompt,
            "compute_dtype": str(compute_dtype), "double_quant": not args.no_double_quant,
            "batch_size": args.batch_size, "seed": args.seed, "letter_token_ids": letter_table,
            "torch": torch.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seconds": round(time.time() - t0, 1),
        }
        try:
            import transformers, peft, bitsandbytes  # noqa: E401
            meta.update(transformers=transformers.__version__, peft=peft.__version__,
                        bitsandbytes=bitsandbytes.__version__)
        except Exception:
            pass
        with open(csv_path.replace(".csv", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[done] {ds_name}: {len(df)} 题，准确率 {df['correct'].mean():.3f}，"
              f"平均 letter_mass {df['letter_mass'].mean():.3f}（最低 {df['letter_mass'].min():.3f}）")
        if df["letter_mass"].mean() < 0.5:
            print("[warn] 平均 letter_mass < 0.5：模型多数时候并不想直接输出字母，"
                  "字母概率可能不可信，请检查 prompt / chat template")
        print(f"[done] 明细已保存：{csv_path}（用时 {meta['seconds']} 秒）")


if __name__ == "__main__":
    main()
