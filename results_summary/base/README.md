# base 模型基线（Qwen2.5-3B-Instruct，未训练）

评估设置：直接作答（direct），在选项字母上取 logits 归一化得到概率；prompt 版本 `direct-v1`；4-bit nf4 + double quant，compute dtype = bfloat16；seed 42；Colab T4。
完整设置见各 `preds_base_*_meta.json`。**训练后的模型必须用同样的设置评估，结果才可比。**

## 结果

| 数据集 | 评估集 | n | 准确率 | 平均自信 | ECE | Brier | NLL | 自信≥0.9 且答错 | 自信≥0.9 的题的准确率 |
|---|---|---|---|---|---|---|---|---|---|
| MedQA（4 选项） | test | 1273 | 0.465 | 0.956 | 0.491 | 0.999 | 5.618 | 565 题（44.4%） | 0.493（1115 题） |
| MedMCQA | validation | 4183 | 0.448 | 0.957 | 0.509 | 1.044 | 6.831 | 1919 题（45.9%） | 0.471（3627 题） |
| PubMedQA | pqa_labeled（全部 1000 题） | 1000 | 0.649 | 0.925 | 0.280 | 0.615 | 2.408 | 219 题（21.9%） | 0.717（775 题） |

`mean_letter_mass` 在三个数据集上都 ≥ 0.989（平均约 1.0），说明模型确实在直接选字母，概率可信。

## 结论

- 三个数据集上都**严重过度自信**：平均自信约 0.93–0.96，准确率只有 0.45–0.65。
- MedQA、MedMCQA 上模型约一半的题答错，但几乎每题都给出很高的自信（绝大多数题自信 ≥ 0.9），自信≥0.9 的题实际准确率不到 0.5。
- PubMedQA 只有 yes/no/maybe 三个选项，答案分布不均衡，0.649 的准确率不能直接说明模型"更强"，与"总猜多数类"的基线比较时要注意。
- 训练后希望看到：ECE、Brier、NLL、"自信≥0.9 且答错"的比例下降，可靠性图的柱子向对角线靠近。

## 文件

| 文件 | 内容 |
|---|---|
| `summary.csv` / `summary.json` | 上表的全部指标（含未列出的 `mean_conf_wrong`、`min_letter_mass` 等） |
| `bins_base_*_direct.csv` | 15 个等宽置信度区间：每档题数、平均自信、准确率（画可靠性图用） |
| `reliability_base_*_direct.png` | 可靠性图 |
| `preds_base_*_meta.json` | 每次评估的完整设置（模型、prompt 版本、量化、seed、库版本、GPU） |

每题明细（`preds_base_*_direct.csv`，MedMCQA 约 717 KB）没有放进仓库，保存在 Google Drive：`medqa-calibration-results/base/preds/`。
