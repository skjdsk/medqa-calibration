# calibration（作业仓库中的评估部分）

比较 SFT / DPO / GRPO（QLoRA）对 Qwen2.5-3B-Instruct 在医学选择题上**校准**的影响。本仓库是评估部分。

## 文件
| 文件 | 作用 |
|---|---|
| `eval/data_loaders.py` | 加载 MedQA / MedMCQA / PubMedQA，统一成 (题目, 选项, 标准答案字母, id) |
| `eval/evaluate.py` | 4-bit 加载模型（可叠加 LoRA），用选项字母的 logits 计算概率，输出每题明细 CSV |
| `eval/metrics.py` | 只读明细 CSV，计算准确率 / ECE / Brier / NLL / 高置信错误等，画可靠性图 |
| `eval/test_metrics.py` | 手算小例子验证指标公式 |
| `notebooks/eval_base.ipynb` | Colab 一键运行 |

## 常用命令（先 `cd calibration`，在这个文件夹里运行）
```bash
python -m pytest eval/test_metrics.py -q                      # 验证指标
python -m eval.data_loaders --dataset medqa --n 3             # 只看数据样例
python -m eval.evaluate --dataset medqa medmcqa pubmedqa \
    --adapter /path/to/adapter --model_name sft_seed1 --out_dir results/
python -m eval.metrics --pred_csv "results/preds_*.csv" --out_dir results/metrics
```

## 明细 CSV 字段
`dataset, question_id, model_name, eval_mode, n_options, gold, pred, confidence, correct, letter_mass, n_prompt_tokens, prob_A … prob_E`
- `prob_*`：在选项字母上归一化后的概率（该题不存在的选项为空）
- `letter_mass`：归一化**前**所有选项字母（含 "A" 与 " A" 两种写法）在整个词表上的概率之和，越接近 1 越说明 prompt 有效
- 每个 CSV 旁边有 `_meta.json`，记录 prompt 版本、量化设置、库版本等，用来确认 7 个模型的设置完全一致
