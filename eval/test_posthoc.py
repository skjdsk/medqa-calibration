"""Regression tests for CoT scoring and leakage-safe temperature scaling."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from eval.data_loaders import MCQItem
from eval.reasoning import reasoning_prompt, split_reasoning
from eval.temperature import fit_artifact, apply_artifact, transform, report, logits_from_frame


def frame():
    return pd.DataFrame({
        'dataset': ['medqa'] * 8, 'question_id': [f'medqa4-{i}' for i in range(8)],
        'model_name': ['model'] * 8, 'eval_mode': ['direct'] * 8,
        'n_options': [2] * 8, 'gold': ['A', 'B'] * 4,
        'prob_A': [.99] * 8, 'prob_B': [.01] * 8,
    })


def test_temperature_reduces_nll_and_preserves_answers():
    data = frame()
    artifact = fit_artifact(data.iloc[:4])
    calibrated = apply_artifact(data.iloc[4:], artifact)
    assert artifact['temperature'] > 1
    assert artifact['fit_nll_after'] < artifact['fit_nll_before']
    assert (calibrated.pred == 'A').all()
    assert (calibrated.confidence < .99).all()
    with tempfile.TemporaryDirectory() as tmp:
        metrics = report(data.iloc[4:], calibrated, Path(tmp) / 'report')
        assert metrics[0]['accuracy'] == metrics[1]['accuracy']
        assert metrics[1]['nll'] < metrics[0]['nll']
        assert (Path(tmp) / 'report/metrics.csv').exists()


def test_leakage_and_identity_checks():
    data = frame()
    artifact = fit_artifact(data.iloc[:4])
    with pytest.raises(ValueError, match='overlap'):
        apply_artifact(data.iloc[:4], artifact)
    with pytest.raises(ValueError, match='protocol'):
        apply_artifact(data.iloc[4:].assign(eval_mode='cot'), artifact)
    with pytest.raises(ValueError, match='protocol'):
        apply_artifact(data.iloc[4:].assign(model_name='other'), artifact)
    with pytest.raises(ValueError, match='Test'):
        fit_artifact(data.assign(question_id=[f'test-{i}' for i in range(8)]))
    with pytest.raises(ValueError, match='Only validation'):
        fit_artifact(data.assign(split='test'))
    with pytest.raises(ValueError, match='Duplicate'):
        fit_artifact(pd.concat([data, data]))
    with pytest.raises(ValueError, match='nonnegative'):
        fit_artifact(data.assign(prob_A=1.1, prob_B=-.1))


def test_missing_options_and_saved_log_probabilities():
    data = frame().assign(prob_C=np.nan)
    scaled = transform(data, 2)
    assert scaled.prob_C.isna().all()
    assert np.allclose(scaled.prob_A + scaled.prob_B, 1)
    data = data.assign(log_prob_A=np.log(.99), log_prob_B=np.log(.01), log_prob_C=np.nan)
    assert logits_from_frame(data)[3] == 'saved_log_probabilities'
    data['log_prob_A'] = 0
    with pytest.raises(ValueError, match='do not match'):
        logits_from_frame(data)
    with pytest.raises(ValueError, match='positive'):
        transform(frame(), 0)


def test_answer_removal_and_training_prompt_parity():
    assert split_reasoning('Reason.\nAnswer: B\nAnswer: A', list('ABCD')) == ('Reason.', 'B')
    assert split_reasoning('Reason.\n**Final answer:** (D)', list('ABCD')) == ('Reason.', 'D')
    assert split_reasoning('No final line', list('ABCD')) == ('No final line', None)
    item = MCQItem('medqa', 'q', 'Question?', ['one', 'two', 'three', 'four'], 'A')
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages
    from train.grpo.grpo_data import build_messages
    assert reasoning_prompt(item, Tokenizer()) == build_messages(item.question, dict(zip('ABCD', item.options)))


def test_cot_routes_through_shared_scorer_and_restores_state():
    from eval.evaluate import score_cot
    class Tokenizer:
        padding_side = 'right'
        pad_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)
        def __call__(self, prompts, **kwargs):
            return {'input_ids': torch.ones((len(prompts), 4), dtype=torch.long),
                    'attention_mask': torch.ones((len(prompts), 4), dtype=torch.long)}
        def encode(self, text, **kwargs):
            return [1] * 8
        def decode(self, tokens, **kwargs):
            return 'Reason.\nAnswer: B' if tokens[0] == 2 else 'Incomplete reasoning'
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=100)
            self.generation_config = SimpleNamespace(eos_token_id=9)
        def generate(self, **kwargs):
            assert kwargs['do_sample'] is False and not self.training
            return torch.tensor([[1, 1, 1, 1, 2, 9], [1, 1, 1, 1, 3, 3]])
    model, tokenizer = Model(), Tokenizer()
    items = [MCQItem('medqa', str(i), 'Q?', ['a', 'b', 'c', 'd'], 'B') for i in range(2)]
    with patch('eval.evaluate.score_letters', return_value=[{}, {}]) as score:
        output = score_cot(model, tokenizer, items, {}, batch_size=2, max_new_tokens=2)
        prompts = score.call_args.kwargs['prompt_texts']
        assert all(p.endswith('\nAnswer:') for p in prompts)
        assert 'Answer: B' not in prompts[0]
    assert output[0]['generated_correct'] == 1
    assert output[1]['format_ok'] == 0 and output[1]['generation_truncated'] == 1
    assert model.training and tokenizer.padding_side == 'right'

    with pytest.raises(ValueError, match='context limit'):
        score_cot(model, tokenizer, items, {}, max_new_tokens=100)
    assert model.training and tokenizer.padding_side == 'right'


def test_tiny_transformers_generation_and_probability_export():
    """Exercise actual generation/forward calls without downloading model weights."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel
    from eval.evaluate import score_cot, score_letters, build_rows
    from train.dpo.validation import letter_table
    torch.set_num_threads(1)
    backend = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2,
                                  'A': 3, 'B': 4, 'C': 5, 'D': 6, 'Answer': 7, ':': 8},
                                 unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                       pad_token='[PAD]', eos_token='[EOS]')
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    tokenizer.padding_side = 'left'
    model = GPT2LMHeadModel(GPT2Config(vocab_size=9, n_positions=256, n_embd=16, n_layer=1,
                                      n_head=2, bos_token_id=2, eos_token_id=2, pad_token_id=1))
    model.eval()
    items = [MCQItem('medqa', str(i), 'Question ' * (i + 1), ['one', 'two', 'three', 'four'], 'A')
             for i in range(2)]
    table = letter_table(tokenizer)
    direct = score_letters(model, tokenizer, items, table, batch_size=2)
    single = score_letters(model, tokenizer, items, table, batch_size=1)
    assert np.allclose(list(direct[0]['letter_probs'].values()), list(single[0]['letter_probs'].values()), atol=1e-6)
    scores = score_cot(model, tokenizer, items, table, batch_size=2, max_new_tokens=2,
                       max_context_tokens=256)
    output = build_rows(items, scores, 'tiny', 'cot', 4)
    assert len(output) == 2 and 'generated_text' in output
    logits_from_frame(output)
    assert np.isfinite(output[['log_prob_A', 'log_prob_B', 'log_prob_C', 'log_prob_D']]).all().all()
