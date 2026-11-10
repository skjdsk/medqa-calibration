"""Evaluate a DPO adapter on held-out MedQA using shared scoring and metrics."""
from __future__ import annotations

import json
import math
from pathlib import Path


def write_json(path, value):
    """Write valid JSON atomically; unavailable metrics are represented by null."""
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=True, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validation_items(source_rows, indices):
    from eval.data_loaders import MCQItem
    return [MCQItem(dataset='medqa', question_id=f'medqa4-{i}', question=source_rows[i]['question'],
                    options=[source_rows[i]['options'][k] for k in 'ABCD'], gold=source_rows[i]['answer_idx'])
            for i in indices]


def letter_table(tokenizer):
    """Match the shared evaluator's letter variants without localized messages."""
    table = {}
    for letter in 'ABCD':
        ids = [tokenizer.encode(s, add_special_tokens=False) for s in (letter, ' ' + letter)]
        table[letter] = sorted({x[0] for x in ids if len(x) == 1 and x[0] != tokenizer.unk_token_id})
        if not table[letter]:
            raise ValueError(f'No single-token representation for {letter}')
    flat = [i for ids in table.values() for i in ids]
    if len(flat) != len(set(flat)):
        raise ValueError('Letter token IDs overlap')
    return table


def evaluate_validation(model, tokenizer, items, batch_size, dtype, output, tag,
                        mode="direct", cot_max_new_tokens=512, cot_max_context_tokens=4096):
    import torch
    from eval.evaluate import build_rows, score_letters, score_cot
    from eval.metrics import compute_metrics, reliability_table, prob_matrix
    was_training, padding = model.training, tokenizer.padding_side
    model.eval()
    tokenizer.padding_side = 'left'
    try:
        device_type = next(model.parameters()).device.type
        if mode not in {'direct', 'cot'}:
            raise ValueError('Unknown evaluation mode')
        with torch.inference_mode(), torch.autocast(device_type, dtype=dtype, enabled=device_type == 'cuda'):
            if mode == 'cot':
                scores = score_cot(model, tokenizer, items, letter_table(tokenizer), batch_size=batch_size,
                                   max_new_tokens=cot_max_new_tokens, max_context_tokens=cot_max_context_tokens)
            else:
                scores = score_letters(model, tokenizer, items, letter_table(tokenizer), batch_size=batch_size)
        frame = build_rows(items, scores, tag, mode, 4)
        metrics = compute_metrics(frame, n_bins=15, hc_threshold=0.9)
        if mode == 'cot':
            metrics.update(generated_accuracy=float(frame.generated_correct.mean()),
                           format_success_rate=float(frame.format_ok.mean()),
                           truncation_rate=float(frame.generation_truncated.mean()))
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output / f'{tag}_predictions.csv', index=False)
        probs, gold = prob_matrix(frame)
        reliability_table(probs.max(1), (probs.argmax(1) == gold).astype(float), 15).to_csv(
            output / f'{tag}_reliability.csv', index=False)
        write_json(output / f'{tag}_metrics.json', metrics)
        return metrics
    finally:
        tokenizer.padding_side = padding
        model.train(was_training)


def better_validation(metrics, best):
    """Select on validation accuracy, then NLL; never use transfer/test scores."""
    return best is None or (metrics['accuracy'], -metrics['nll']) > (best['accuracy'], -best['nll'])
