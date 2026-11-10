"""Batched Transformers CoT evaluation with full-vocabulary answer rescoring."""
from __future__ import annotations

import re
import torch

PROMPT_VERSION = 'dpo-cot-v1'
ANSWER_MARKER = re.compile(r'(?im)^\s*\*{0,2}(?:final\s+)?answer\s*\*{0,2}\s*:\s*\*{0,2}\s*')


def reasoning_prompt(item, tokenizer, system_prompt=None):
    """Match the frozen DPO training prompt for four-option MedQA."""
    from train.grpo.grpo_data import SYSTEM_PROMPT, INSTRUCTION
    letters = item.letters
    instruction = INSTRUCTION if letters == list('ABCD') else (
        'Think through the problem step by step. Then, on the final line, '
        'write your answer in exactly this format:\nAnswer: X\n'
        f'where X is one of {", ".join(letters)}.')
    body = item.question.strip() + '\n\n'
    body += '\n'.join(f'{letter}. {option}' for letter, option in zip(letters, item.options))
    body += '\n\n' + instruction
    if item.context:
        body = f'Context:\n{item.context}\n\n' + body
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT if system_prompt is None else system_prompt},
                {'role': 'user', 'content': body}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def split_reasoning(text, letters):
    """Remove the first final-answer line and everything after it before rescoring."""
    match = ANSWER_MARKER.search(text)
    if match is None:
        return text.rstrip(), None
    answer = re.match(r'\(?([A-J])\b', text[match.end():])
    letter = answer.group(1) if answer and answer.group(1) in letters else None
    return text[:match.start()].rstrip(), letter


@torch.no_grad()
def score_cot(model, tokenizer, items, letter_table, batch_size=1, system_prompt=None,
              max_new_tokens=512, max_context_tokens=4096):
    """Generate greedily, then score letters after reasoning plus a fixed answer cue.

    All questions remain in the denominator, including malformed/truncated outputs.
    Raw generation accuracy is exported separately from forced-answer accuracy.
    """
    from eval.evaluate import score_letters
    from tqdm.auto import tqdm
    if batch_size < 1 or max_new_tokens < 1 or max_context_tokens < 2:
        raise ValueError('Batch size and token limits must be positive')
    config_limit = getattr(model.config, 'max_position_embeddings', max_context_tokens)
    context_limit = min(max_context_tokens, config_limit)
    was_training, padding = model.training, tokenizer.padding_side
    model.eval()
    tokenizer.padding_side = 'left'
    results = []
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    try:
        for start in tqdm(range(0, len(items), batch_size), desc='CoT generation and scoring'):
            batch = items[start:start + batch_size]
            prompts = [reasoning_prompt(item, tokenizer, system_prompt) for item in batch]
            encoded = tokenizer(prompts, padding=True, return_tensors='pt', add_special_tokens=False)
            width = encoded['input_ids'].shape[1]
            if width + max_new_tokens + 16 > context_limit:
                raise ValueError('CoT prompt plus generation exceeds context limit; no silent truncation')
            encoded = {key: value.to(next(model.parameters()).device) for key, value in encoded.items()}
            generated = model.generate(**encoded, do_sample=False, num_beams=1,
                                       max_new_tokens=max_new_tokens, use_cache=True,
                                       pad_token_id=tokenizer.pad_token_id,
                                       return_dict_in_generate=False)
            metadata, scoring_prompts = [], []
            for item, prompt, tokens in zip(batch, prompts, generated[:, width:].tolist()):
                end = next((i for i, token in enumerate(tokens) if token in eos_ids), len(tokens))
                text = tokenizer.decode(tokens[:end], skip_special_tokens=True)
                rationale, answer = split_reasoning(text, item.letters)
                scoring_prompt = prompt + rationale + '\nAnswer:'
                if len(tokenizer.encode(scoring_prompt, add_special_tokens=False)) > context_limit:
                    raise ValueError('CoT rescoring prompt exceeds context limit')
                scoring_prompts.append(scoring_prompt)
                metadata.append({'reasoning': rationale, 'generated_text': text,
                                 'generated_answer': answer, 'format_ok': int(answer is not None),
                                 'generated_correct': int(answer == item.gold),
                                 'generation_truncated': int(end == len(tokens) and end >= max_new_tokens),
                                 'n_generated_tokens': end, 'prompt_version': PROMPT_VERSION})
            scores = score_letters(model, tokenizer, batch, letter_table, batch_size=batch_size,
                                   prompt_texts=scoring_prompts)
            for score, detail in zip(scores, metadata):
                score.update(detail)
            results.extend(scores)
    finally:
        tokenizer.padding_side = padding
        model.train(was_training)
    return results
