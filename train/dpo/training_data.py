"""Validate frozen preference artifacts and audit token lengths without model weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from train.grpo.grpo_data import build_messages, normalize_options
from train.grpo.rewards import extract_answer

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / 'outputs/medqa5000_L4_seed42_qpb2/selection_seed42/dpo.jsonl'
DEFAULT_SPLIT = REPO / 'data/medqa_split_seed42.json'
DEFAULT_LOCAL_MODEL = REPO.parent / 'Qwen2.5-3B-Instruct'
DEFAULT_LOCAL_DATA = REPO.parent / 'MedQA-USMLE-4-options/phrases_no_exclude_train.jsonl'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    result = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            raise ValueError(f'Blank JSONL row at {path}:{number}')
        result.append(json.loads(line))
    return result


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_source(source, local_file, revision):
    if source == 'local':
        return read_jsonl(local_file)
    from datasets import load_dataset
    return list(load_dataset('GBaker/MedQA-USMLE-4-options', split='train', revision=revision))


def validate_preferences(data_file, split_file, source_rows):
    """Use retained IDs and original labels; raw sampled responses are not required."""
    path = Path(data_file)
    split = read_json(split_file)
    sampling = read_json(path.parent.parent / 'sampling_manifest.json')['config']
    build = read_json(path.parent / 'build_manifest.json')
    rows = read_jsonl(path)
    retained = read_jsonl(path.parent / 'retained_questions.jsonl')
    train_ids, val_ids = split['train_indices'], split['val_indices']
    require(len(set(train_ids)) == len(train_ids), 'Duplicate training indices')
    require(len(set(val_ids)) == len(val_ids), 'Duplicate validation indices')
    require(not set(train_ids) & set(val_ids), 'Training and validation indices overlap')
    require(len(source_rows) == split['total_rows'], 'Source dataset size differs from the fixed split')
    require(sampling['split_sha256'] == object_hash(split), 'Sampling split hash mismatch')
    require(sampling['train_indices'] == train_ids, 'Expected the complete fixed candidate set')
    canonical = [{k: r[k] for k in ('question', 'options', 'answer_idx', 'answer')} for r in source_rows]
    require(object_hash(canonical) == sampling['training_rows_sha256'], 'Source content/order hash mismatch')
    require(build['sampling_config_sha256'] == object_hash(sampling), 'Build configuration hash mismatch')
    require(len(rows) == len(retained) == build['retained_questions'] > 0, 'Preference count mismatch')
    seen = set()
    for position, (row, identity) in enumerate(zip(rows, retained)):
        index = identity['source_index']
        require(index in train_ids and index not in seen, f'Invalid or duplicate training index: {index}')
        seen.add(index)
        raw = source_rows[index]
        require(identity['question_id'] == f'medqa4-{index}', f'Question ID mismatch at row {position}')
        require(set(row) == {'prompt', 'chosen', 'rejected'}, f'Unexpected preference schema at row {position}')
        require(row['prompt'] == build_messages(raw['question'], normalize_options(raw['options'])),
                f'Prompt/source mismatch at row {position}')
        for key in ('chosen', 'rejected'):
            messages = row[key]
            require(isinstance(messages, list) and len(messages) == 1, f'Invalid {key} messages')
            require(messages[0].get('role') == 'assistant', f'Invalid {key} role')
            text = messages[0].get('content')
            require(isinstance(text, str) and bool(text.strip()), f'Empty {key} response')
            pred = extract_answer(text)
            require(pred in ('A', 'B', 'C', 'D'), f'Unparseable {key} answer at row {position}')
            require((pred == raw['answer_idx']) == (key == 'chosen'), f'Incorrect {key} label at row {position}')
    return rows, retained, split, {
        'pairs': len(rows), 'validation_questions': len(val_ids),
        'data_sha256': file_hash(path), 'retained_ids_sha256': file_hash(path.parent / 'retained_questions.jsonl'),
        'split_sha256': object_hash(split), 'source_sha256': object_hash(canonical),
        'sampling_config_sha256': object_hash(sampling),
    }


def audit_lengths(rows, tokenizer, max_length):
    """Fail instead of truncating the final answer or silently dropping pairs."""
    prompt_lengths, lengths = [], []
    for i, row in enumerate(rows):
        prompt = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True)
        prompt_lengths.append(len(prompt))
        for key in ('chosen', 'rejected'):
            full = tokenizer.apply_chat_template(row['prompt'] + row[key], tokenize=True)
            require(full[:len(prompt)] == prompt, f'Chat template prefix mismatch at row {i}')
            require(len(full) > len(prompt), f'No completion tokens at row {i}')
            require(len(full) <= max_length,
                    f'Row {i} {key} has {len(full)} tokens, exceeding --max_length={max_length}. '
                    'Increase the limit; responses are never silently truncated.')
            lengths.append(len(full))
    return {'max_prompt_tokens': max(prompt_lengths), 'max_sequence_tokens': max(lengths),
            'mean_sequence_tokens': sum(lengths) / len(lengths), 'total_sequence_tokens': sum(lengths)}


def compatible_resume(saved, current):
    require(saved == current, 'Resume configuration or data changed. Restore the original settings or use a new output directory.')
