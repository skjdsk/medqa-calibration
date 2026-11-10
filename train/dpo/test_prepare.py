"""CPU-only regression tests: no model weights, network, or real sampling."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from train.dpo.prepare import generate_question_batch, pending_question_batches


class Inputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    pad_token_id = 0
    padding_side = 'right'

    def __call__(self, prompts, **kwargs):
        assert self.padding_side == 'left'
        assert kwargs['padding'] is True
        return Inputs(input_ids=torch.tensor([[0, 0, 11], [12, 13, 22]]),
                      attention_mask=torch.tensor([[0, 0, 1], [1, 1, 1]]))

    def decode(self, tokens, **kwargs):
        return 'Answer: ' + ('A' if tokens[0] == 31 else 'B')


class Model:
    device = 'cpu'
    config = SimpleNamespace(max_position_embeddings=64)
    generation_config = SimpleNamespace(eos_token_id=[99, 98])

    def __init__(self):
        self.calls = 0

    def generate(self, input_ids, attention_mask, num_return_sequences, **kwargs):
        self.calls += 1
        assert attention_mask.tolist() == [[0, 0, 1], [1, 1, 1]]
        rows = []
        for i, prompt in enumerate(input_ids.tolist()):
            # First question ends early, second reaches the four-token limit.
            tail = [31, 98, 0, 0] if i == 0 else [32, 32, 32, 32]
            rows.extend([prompt + tail for _ in range(num_return_sequences)])
        return torch.tensor(rows)


class BatchTests(unittest.TestCase):
    def test_question_mapping_and_truncation(self):
        for count in (1, 2, 4, 8):
            with self.subTest(samples_per_batch=count):
                model = Model()
                args = SimpleNamespace(samples_per_batch=count, max_new_tokens=4, temperature=1., top_p=1.)
                rows = generate_question_batch(model, Tokenizer(), ['short', 'long'], args, 42)
                self.assertEqual(model.calls, 8 // count)
                for i, answers in enumerate(rows):
                    self.assertEqual([a['sample_id'] for a in answers], list(range(8)))
                    self.assertTrue(all(a['pred'] == ('A' if i == 0 else 'B') for a in answers))
                    self.assertTrue(all(a['truncated'] == bool(i) for a in answers))

    def test_resume_preserves_partial_batch_and_last_batch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            for i in [10, 20, 30]:
                (root / f'medqa4-{i}.json').write_text(json.dumps({
                    'source_index': i, 'config_sha256': 'hash', 'completions': [{}]*8}))
            batches = list(pending_question_batches([10, 20, 30, 40, 50], 2, root, 'hash'))
            self.assertEqual(batches, [([30, 40], {40}), ([50], {50})])

    def test_resume_rejects_wrong_config(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            (root / 'medqa4-10.json').write_text(json.dumps({
                'source_index': 10, 'config_sha256': 'old', 'completions': [{}]*8}))
            with self.assertRaises(ValueError):
                list(pending_question_batches([10], 2, root, 'new'))

    def test_context_overflow_rejected(self):
        model = Model()
        args = SimpleNamespace(max_new_tokens=64)
        with self.assertRaises(ValueError):
            generate_question_batch(model, Tokenizer(), ['a', 'b'], args, 42)
        self.assertEqual(model.calls, 0)


if __name__ == '__main__':
    unittest.main()
