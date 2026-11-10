"""Offline Base sampling and matched SFT/DPO data construction (no training).

Run from repository root, locally or in Colab:
  python -m train.dpo.prepare sample --source local --output_dir outputs/dpo_local
  python -m train.dpo.prepare sample --source hf --output_dir outputs/dpo_data
  python -m train.dpo.prepare build --output_dir outputs/dpo_data

Default: eight stochastic completions per fixed training question. A sample
command can be resumed unchanged; complete question files are written atomically.
Build is CPU-only and never loads model weights. Outputs use TRL conversational
JSONL schemas. Do not pre-apply a chat template when passing these to TRL.

--questions_per_batch controls simultaneous questions; --samples_per_batch
controls simultaneous answers PER question. Their product is the number of
generated sequences in one call. Each question always receives eight answers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random

from data.medqa_split import SPLIT_FILE, load_split
from train.grpo.rewards import extract_answer

PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_MODEL = PROJECT_DIR / 'Qwen2.5-3B-Instruct'
DEFAULT_LOCAL_DATA = PROJECT_DIR / 'MedQA-USMLE-4-options' / 'phrases_no_exclude_train.jsonl'


def load_sampling_source(args, split):
    """Return ordered training rows and model loading settings.

    Local mode performs no Hub requests. Preserve JSONL line order because the
    shared split stores source row indices. HF mode always requests train only.
    """
    from data.medqa_split import DATASET_NAME, check_dataset_size

    if args.source == 'local':
        model_path = Path(args.model or DEFAULT_LOCAL_MODEL).expanduser().resolve()
        data_path = Path(args.local_train_file).expanduser().resolve()
        if not (model_path / 'config.json').is_file():
            raise FileNotFoundError(f'Local model config missing: {model_path / "config.json"}')
        if not data_path.is_file():
            raise FileNotFoundError(f'Local training JSONL missing: {data_path}')
        full = []
        with data_path.open(encoding='utf-8-sig') as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f'Blank row in {data_path}:{line_no}; preserve source row order')
                try:
                    full.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid JSON at {data_path}:{line_no}') from exc
        model_source = str(model_path)
        model_kwargs = {'local_files_only': True}
        provenance = {'source': 'local', 'model': model_source, 'local_train_file': str(data_path)}
    else:
        from datasets import load_dataset
        model_source = args.model or 'Qwen/Qwen2.5-3B-Instruct'
        full = load_dataset(DATASET_NAME, split='train', revision=args.dataset_revision)
        model_kwargs = {'revision': args.revision}
        provenance = {'source': 'hf', 'model': model_source, 'revision': args.revision,
                      'dataset': DATASET_NAME, 'dataset_revision': args.dataset_revision}
    check_dataset_size(len(full), split)
    canonical = []
    for index, row in enumerate(full):
        options = row.get('options', {})
        gold = row.get('answer_idx')
        if (not isinstance(row.get('question'), str) or not isinstance(options, dict)
                or set(options) != set('ABCD') or gold not in options
                or row.get('answer') != options[gold]):
            raise ValueError(f'Invalid MedQA training row {index}')
        canonical.append({'question': row['question'], 'options': options,
                          'answer_idx': gold, 'answer': row['answer']})
    # Detect content/order changes on resume, not just a matching row count.
    provenance['training_rows_sha256'] = digest(canonical)
    return full, model_source, model_kwargs, provenance


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def pending_question_batches(indices, batch_size, records, config_hash):
    """Keep original batch boundaries across resumes, including partially saved batches."""
    for start in range(0, len(indices), batch_size):
        group = indices[start:start + batch_size]
        missing = []
        for index in group:
            path = records / f'medqa4-{index}.json'
            if not path.exists():
                missing.append(index)
                continue
            row = json.loads(path.read_text(encoding='utf-8'))
            if (row.get('source_index') != index or row.get('config_sha256') != config_hash
                    or len(row.get('completions', [])) != 8):
                raise ValueError(f'Invalid existing sample record: {path}')
        if missing:
            # Replay the entire group so removing completed questions cannot
            # change padding, random-number consumption, or batch seeds.
            yield group, set(missing)


def generate_question_batch(model, tokenizer, prompts, args, batch_seed):
    """Return one list of eight completions per prompt, in input order."""
    import torch
    from transformers import set_seed

    tokenizer.padding_side = 'left'
    inputs = tokenizer(prompts, padding=True, return_tensors='pt',
                       add_special_tokens=False).to(model.device)
    prompt_width = inputs['input_ids'].shape[1]
    context = getattr(model.config, 'max_position_embeddings', None)
    if context and prompt_width + args.max_new_tokens > context:
        raise ValueError('Question batch exceeds context budget; no silent truncation')
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
    answers = [[] for _ in prompts]
    for start in range(0, 8, args.samples_per_batch):
        count = min(args.samples_per_batch, 8-start)
        set_seed((batch_seed + start) % (2**32))
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=True, temperature=args.temperature,
                top_p=args.top_p, num_return_sequences=count,
                max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        if len(generated) != len(prompts) * count:
            raise ValueError('Unexpected number of generated sequences')
        # Transformers repeats each prompt count times, keeping those answers adjacent.
        for row_num, seq in enumerate(generated):
            tokens = seq[prompt_width:].tolist()
            end = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
            ended = end is not None
            if ended:
                tokens = tokens[:end]  # Discard EOS and padding after an early finish.
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            dest = answers[row_num // count]
            dest.append(dict(sample_id=len(dest), text=text, pred=extract_answer(text),
                             truncated=not ended and len(tokens) >= args.max_new_tokens))
        del generated
    return answers


def sample(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from train.grpo.grpo_data import build_messages, normalize_options

    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError('temperature must be positive; top_p must be in (0, 1]')
    if args.quantization == '4bit' and not torch.cuda.is_available():
        raise ValueError('4bit sampling requires CUDA; use --quantization none for CPU')
    split = load_split(args.split_file)
    full, model_source, model_kwargs, provenance = load_sampling_source(args, split)
    indices = split['train_indices'][:args.n_questions or None]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {k: getattr(args, k) for k in ('n_questions', 'seed',
              'temperature', 'top_p', 'max_new_tokens', 'quantization', 'samples_per_batch',
              'questions_per_batch')}
    config.update(provenance)
    config.update(samples_per_question=8, train_indices=indices, split_sha256=digest(split),
                  prompt_version='grpo-build_messages-v1', sampling_version='multi-question-v2')
    manifest_path = out / 'sampling_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8'))['config'] != config:
            raise ValueError('Existing sampling configuration differs; use a new output_dir')
    else:
        write_json(manifest_path, {'config': config, 'versions': {
            name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'datasets')}})
    records = out / 'samples'
    records.mkdir(exist_ok=True)
    config_hash = digest(config)
    batches = list(pending_question_batches(indices, args.questions_per_batch, records, config_hash))
    missing_count = sum(len(missing) for _, missing in batches)
    if not missing_count:
        print('All questions already saved; no model loading required.', flush=True)
        return
    print(f'Pending questions: {missing_count}; questions per batch: {args.questions_per_batch}; '
          f'max simultaneous sequences: {args.questions_per_batch * args.samples_per_batch}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_source, **model_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if torch.cuda.is_available() else torch.float32
    kwargs = dict(model_kwargs, torch_dtype=dtype)
    if torch.cuda.is_available():
        kwargs['device_map'] = 'auto'
    if args.quantization == '4bit':
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_source, **kwargs).eval()
    completed = len(indices) - missing_count
    for group, missing in batches:
        rows, prompts = [], []
        for index in group:
            raw = full[index]
            options = normalize_options(raw['options'])
            gold = str(raw['answer_idx']).strip().upper()
            if gold not in options:
                raise ValueError(f'Invalid gold answer at row {index}')
            messages = build_messages(raw['question'], options)
            rows.append(dict(question_id=f'medqa4-{index}', source_index=index,
                prompt=messages, gold=gold, question=raw['question'], options=options,
                config_sha256=config_hash))
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        answers = generate_question_batch(model, tokenizer, prompts, args,
                                         args.seed + group[0] * 1009)
        for row, completions in zip(rows, answers):
            if row['source_index'] not in missing:
                continue
            path = records / f"{row['question_id']}.json"
            write_json(path, dict(row, completions=completions))
            completed += 1
            print(f'{completed}/{len(indices)} saved {path.name}', flush=True)


def build(args):
    out = Path(args.output_dir)
    manifest = json.loads((out / 'sampling_manifest.json').read_text(encoding='utf-8'))
    config = manifest['config']
    split = load_split(args.split_file)
    if digest(split) != config['split_sha256']:
        raise ValueError('Split differs from sampling manifest')
    rows, audit, sft, dpo = [], [], [], []
    for index in config['train_indices']:
        path = out / 'samples' / f'medqa4-{index}.json'
        if not path.exists():
            raise ValueError(f'Missing {path}; finish sampling before building')
        row = json.loads(path.read_text(encoding='utf-8'))
        if row['source_index'] != index or row['config_sha256'] != digest(config) or len(row['completions']) != 8:
            raise ValueError(f'Invalid sample record: {path}')
        correct, wrong = [], []
        for c in row['completions']:
            pred = extract_answer(c['text'])
            # Malformed or cut-off responses are not medical negative examples.
            if c['truncated'] or pred not in ('A', 'B', 'C', 'D'):
                continue
            (correct if pred == row['gold'] else wrong).append(c)
        keep = bool(correct and wrong)
        entry = dict(question_id=row['question_id'], n_correct=len(correct), n_wrong=len(wrong), retained=keep)
        if keep:
            rng = random.Random(args.selection_seed + index)
            chosen, rejected = rng.choice(correct), rng.choice(wrong)
            entry.update(chosen_sample_id=chosen['sample_id'], rejected_sample_id=rejected['sample_id'])
            assistant = lambda text: [{'role': 'assistant', 'content': text}]
            sft.append(dict(prompt=row['prompt'], completion=assistant(chosen['text'])))
            dpo.append(dict(prompt=row['prompt'], chosen=assistant(chosen['text']), rejected=assistant(rejected['text'])))
            rows.append(dict(question_id=row['question_id'], source_index=index))
        audit.append(entry)
    # A directory per selection seed prevents overwriting a different selection.
    target = out / f'selection_seed{args.selection_seed}'
    target.mkdir(exist_ok=True)
    for name, values in [('sft', sft), ('dpo', dpo), ('selection_audit', audit), ('retained_questions', rows)]:
        path = target / f'{name}.jsonl'
        tmp = path.with_suffix('.jsonl.tmp')
        tmp.write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values), encoding='utf-8')
        os.replace(tmp, path)
    write_json(target / 'build_manifest.json', dict(selection_seed=args.selection_seed,
        sampling_config_sha256=digest(config), candidate_questions=len(audit), retained_questions=len(rows),
        excluded_questions=len(audit)-len(rows), policy='valid parsed answer, non-truncated; one random correct/incorrect pair'))
    print(f'Retained {len(rows)}/{len(audit)} questions. Output: {target}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['sample', 'build'])
    p.add_argument('--output_dir', default='outputs/dpo_data')
    p.add_argument('--split_file', default=str(SPLIT_FILE))
    p.add_argument('--source', choices=['local', 'hf'], default='hf',
                   help='sample only: local model + JSONL, or HF model + MedQA train dataset')
    p.add_argument('--model', default=None,
                   help='Local model directory for local mode; repository ID for hf mode')
    p.add_argument('--local_train_file', default=str(DEFAULT_LOCAL_DATA),
                   help='local mode only: original ordered MedQA training JSONL')
    p.add_argument('--revision', default='main', help='Use a commit hash for strict model reproducibility')
    p.add_argument('--dataset_revision', default='main', help='HF dataset revision (hf mode only)')
    p.add_argument('--n_questions', type=int, default=0, help='0 = all fixed 5000 candidate questions')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--selection_seed', type=int, default=42)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_p', type=float, default=1.0)
    p.add_argument('--max_new_tokens', type=int, default=1024)
    p.add_argument('--samples_per_batch', type=int, choices=[1, 2, 4, 8], default=1)
    p.add_argument('--questions_per_batch', type=int, default=1,
                   help='Questions generated together; simultaneous sequences = this * samples_per_batch')
    p.add_argument('--quantization', choices=['none', '4bit'], default='none')
    args = p.parse_args()
    if args.n_questions < 0:
        p.error('n_questions must be >= 0')
    if args.questions_per_batch < 1 or args.max_new_tokens < 1:
        p.error('questions_per_batch and max_new_tokens must be positive')
    (sample if args.stage == 'sample' else build)(args)


if __name__ == '__main__':
    main()
