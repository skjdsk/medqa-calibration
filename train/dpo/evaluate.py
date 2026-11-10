"""Post-training direct/CoT evaluation of the base model or a saved DPO adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.dpo.train import resolve_model
from train.dpo.training_data import (
    DEFAULT_DATA, DEFAULT_SPLIT, DEFAULT_LOCAL_DATA, load_source, read_jsonl,
    require, validate_preferences, object_hash,
)
from train.dpo.validation import evaluate_validation, validation_items, write_json


def load_test(source, local_file, revision):
    if source == 'local':
        return read_jsonl(local_file)
    from datasets import load_dataset
    return list(load_dataset('GBaker/MedQA-USMLE-4-options', split='test', revision=revision))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['local', 'hf'], default='hf')
    parser.add_argument('--model')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--dataset_revision', default='main')
    parser.add_argument('--local_train_file', type=Path, default=DEFAULT_LOCAL_DATA)
    parser.add_argument('--local_test_file', type=Path,
                        default=DEFAULT_LOCAL_DATA.with_name('phrases_no_exclude_test.jsonl'))
    parser.add_argument('--data_file', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--split_file', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--quantization', choices=['4bit', 'none'], default='4bit')
    parser.add_argument('--compute_dtype', choices=['auto', 'bf16', 'fp16'], default='auto')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--model_name', default='base')
    parser.add_argument('--modes', nargs='+', choices=['direct', 'cot'], default=['direct', 'cot'])
    parser.add_argument('--splits', nargs='+', choices=['validation', 'test'], default=['validation'])
    parser.add_argument('--n', type=int, default=0, help='Limit each split; zero means all questions')
    parser.add_argument('--cot_max_new_tokens', type=int, default=512)
    parser.add_argument('--cot_max_context_tokens', type=int, default=4096)
    args = parser.parse_args(argv)
    require(bool(args.model_name) and all(c.isalnum() or c in '._-' for c in args.model_name) and args.model_name not in {'.', '..'}, 'Use a simple unique model name without path separators')
    require(args.n >= 0, 'Question limit cannot be negative')
    require(args.cot_max_new_tokens > 0 and args.cot_max_context_tokens > 0, 'Token limits must be positive')
    if args.adapter:
        require((args.adapter / 'adapter_config.json').is_file(), 'Adapter config is missing')
        require((args.adapter / 'adapter_model.safetensors').is_file(), 'Adapter weights are missing')
    require(args.batch_size > 0, 'Batch size must be positive')
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()),
            'Evaluation output already exists. Choose a new output directory.')
    model_name, model_kwargs = resolve_model(args)
    source = load_source(args.source, args.local_train_file, args.dataset_revision)
    _, _, split, audit = validate_preferences(args.data_file, args.split_file, source)
    test = load_test(args.source, args.local_test_file, args.dataset_revision) if 'test' in args.splits else []
    require(bool(test) or 'test' not in args.splits, 'Test split is empty')
    items = {'validation': validation_items(source, split['val_indices']),
             'test': validation_items(test, list(range(len(test))))}
    for item in items['test']:
        item.question_id = item.question_id.replace('medqa4-', 'medqa4-test-')

    items = {name: rows[:args.n] if args.n else rows for name, rows in items.items() if name in args.splits}

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    require(torch.cuda.is_available(), 'CUDA is required for model evaluation')
    bf16 = torch.cuda.is_bf16_supported() if args.compute_dtype == 'auto' else args.compute_dtype == 'bf16'
    require(not bf16 or torch.cuda.is_bf16_supported(), 'The GPU does not support BF16')
    dtype = torch.bfloat16 if bf16 else torch.float16
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(model_kwargs, dtype=dtype, device_map={'': 0}, attn_implementation='sdpa')
    if args.quantization == '4bit':
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    manifest = {'model': model_name, 'requested_revision': args.revision,
                'resolved_model_revision': getattr(model.config, '_commit_hash', None),
                'dataset_revision': args.dataset_revision, 'quantization': args.quantization,
                'compute_dtype': str(dtype), 'data_audit': audit, 'test_sha256': object_hash(test),
                'counts': {name: len(rows) for name, rows in items.items()},
                'protocol': 'Direct and/or dpo-cot-v1; full-vocabulary option probabilities',
                'adapter': str(args.adapter) if args.adapter else None,
                'model_name': args.model_name, 'modes': args.modes,
                'cot_max_new_tokens': args.cot_max_new_tokens,
                'cot_max_context_tokens': args.cot_max_context_tokens, 'seed': 42,
                'selection_policy': 'Test metrics are descriptive only; never use them for tuning.'}
    write_json(args.output_dir / 'manifest.json', manifest)
    summary = {}
    for name, rows in items.items():
        summary[name] = {}
        for mode in dict.fromkeys(args.modes):
            summary[name][mode] = evaluate_validation(
                model, tokenizer, rows, args.batch_size, dtype,
                args.output_dir / name / mode, args.model_name, mode=mode,
                cot_max_new_tokens=args.cot_max_new_tokens, cot_max_context_tokens=args.cot_max_context_tokens)
            prediction_file = args.output_dir / name / mode / f'{args.model_name}_predictions.csv'
            import pandas as pd
            frame = pd.read_csv(prediction_file)
            frame['split'] = name
            frame.to_csv(prediction_file, index=False)
            print(name, mode, json.dumps(summary[name][mode]), flush=True)
    write_json(args.output_dir / 'summary.json', summary)


if __name__ == '__main__':
    main()
