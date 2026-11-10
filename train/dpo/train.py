"""Single-GPU offline DPO training with QLoRA, held-out validation, and resume."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time

from train.dpo.training_data import (
    DEFAULT_DATA, DEFAULT_LOCAL_DATA, DEFAULT_LOCAL_MODEL, DEFAULT_SPLIT,
    audit_lengths, compatible_resume, file_hash, load_source, object_hash,
    read_json, require, validate_preferences,
)
from train.dpo.validation import better_validation, evaluate_validation, validation_items, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['local', 'hf'], default='local',
                        help='Source of base model and original MedQA train split; preference pairs remain local files.')
    parser.add_argument('--model', help='Local model directory or Hugging Face model ID.')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--dataset_revision', default='main')
    parser.add_argument('--local_train_file', type=Path, default=DEFAULT_LOCAL_DATA)
    parser.add_argument('--data_file', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--split_file', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--output_dir', type=Path, default=Path('outputs/dpo_train_seed1'))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--quantization', choices=['4bit', 'none'], default='4bit')
    parser.add_argument('--compute_dtype', choices=['auto', 'bf16', 'fp16'], default='auto')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Active LoRA dropout during training; use 0 to disable.')
    parser.add_argument('--baseline_validation', action=argparse.BooleanOptionalAction, default=True,
                        help='Evaluate the initial model in this run; disable when using the notebook baseline stage.')
    parser.add_argument('--lora_target_modules', default='all-linear')
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--num_train_epochs', type=float, default=1.0)
    parser.add_argument('--max_steps', type=int, default=-1, help='Override the epoch schedule for a separate pilot run.')
    parser.add_argument('--stop_after_step', type=int, default=0,
                        help='Stop and checkpoint at this absolute step without changing the full LR schedule.')
    parser.add_argument('--batch_size', type=int, default=1, help='Preference pairs per GPU microbatch; each pair has two sequences.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=2048, help='Full prompt plus response length; overflow is an error.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--precompute_ref_log_probs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save_steps', type=int, default=50)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--logging_steps', type=int, default=5)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--n_pairs', type=int, default=0, help='First N fixed pairs for a pilot; 0 uses all pairs.')
    parser.add_argument('--n_val', type=int, default=0, help='First N held-out questions for a pilot; 0 uses all 500.')
    parser.add_argument('--resume', nargs='?', const='auto', default=None,
                        help='Resume latest checkpoint, or a checkpoint path under this run. Missing checkpoints are errors.')
    parser.add_argument('--check_data', action='store_true',
                        help='Validate data and tokenize it, then exit without loading weights or creating a run.')
    args = parser.parse_args(argv)
    for name in ('batch_size', 'gradient_accumulation_steps', 'max_length', 'lora_r', 'lora_alpha',
                 'save_steps', 'save_total_limit', 'logging_steps', 'eval_batch_size'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.num_train_epochs <= 0 or args.learning_rate <= 0 or args.beta <= 0 or args.max_grad_norm <= 0:
        parser.error('Epochs, learning rate, beta, and max_grad_norm must be positive')
    if args.max_steps == 0 or args.max_steps < -1 or min(args.n_pairs, args.n_val, args.stop_after_step) < 0:
        parser.error('Invalid step count or subset size')
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.lora_dropout < 1 or args.weight_decay < 0:
        parser.error('Invalid warmup ratio, dropout, or weight decay')
    return args


def resolve_model(args):
    if args.source == 'local':
        model = Path(args.model or DEFAULT_LOCAL_MODEL).expanduser().resolve()
        require((model / 'config.json').is_file(), f'Missing model config: {model}')
        require(not (model / 'adapter_config.json').exists(), 'Initialize from the original base model, not a trained adapter')
        return str(model), {'local_files_only': True}
    return args.model or 'Qwen/Qwen2.5-3B-Instruct', {'revision': args.revision}


def disable_base_dropout(model):
    """Call before attaching LoRA so only adapter dropout remains stochastic."""
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, 'attention_dropout'):
            module.attention_dropout = 0.0


def run_signature(args, data_audit, model_name, tokenizer):
    excluded = {'resume', 'stop_after_step', 'check_data', 'output_dir'}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in excluded}
    return {'implementation_version': 2, 'arguments': config, 'data': data_audit, 'model': model_name,
            'tokenizer_sha256': object_hash(tokenizer.backend_tokenizer.to_str()),
            'chat_template_sha256': object_hash(tokenizer.chat_template)}


def resolve_resume(output, resume, signature):
    """Protect previous experiments and reject partial or unrelated checkpoints."""
    output = Path(output)
    if not resume:
        require(not output.exists() or not any(output.iterdir()),
                'Output directory is not empty. Use a new directory or --resume.')
        return None
    require((output / 'run_manifest.json').is_file(), 'Resume manifest is missing')
    compatible_resume(read_json(output / 'run_manifest.json')['signature'], signature)
    checkpoints = output / 'checkpoints'
    if resume == 'auto':
        candidates = [p for p in checkpoints.glob('checkpoint-*') if p.name.split('-')[-1].isdigit()]
        require(bool(candidates), 'No checkpoint exists; cannot resume')
        checkpoint = max(candidates, key=lambda p: int(p.name.split('-')[-1]))
    else:
        checkpoint = Path(resume).resolve()
    require(checkpoint.resolve().parent == checkpoints.resolve(), 'Checkpoint must belong to this run')
    for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'adapter_config.json', 'run_signature.json'):
        require((checkpoint / name).is_file(), f'Incomplete checkpoint: missing {name}')
    require(any(checkpoint.glob('adapter_model.*')), 'Checkpoint adapter weights are missing')
    require(any(checkpoint.glob('rng_state*.pth')), 'Checkpoint RNG state is missing')
    compatible_resume(read_json(checkpoint / 'run_signature.json'), signature)
    return str(checkpoint)


def make_callback(args, tokenizer, items, dtype, signature):
    import torch
    from transformers import TrainerCallback

    class RunCallback(TrainerCallback):
        def __init__(self):
            self.output = args.output_dir
            self.steps = []
            self.started = None
            self.last_eval_step = None
            best_path = self.output / 'best_validation.json'
            self.best = read_json(best_path)['metrics'] if best_path.exists() else None

        def on_train_begin(self, training_args, state, control, model=None, **kwargs):
            self.started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            if not args.resume and args.baseline_validation:
                metrics = evaluate_validation(model, tokenizer, items, args.eval_batch_size, dtype,
                                              self.output / 'validation', 'base')
                write_json(self.output / 'baseline_validation.json', metrics)
                print('Baseline validation:', json.dumps(metrics), flush=True)

        def on_step_begin(self, training_args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.step_started = time.perf_counter()

        def on_step_end(self, training_args, state, control, **kwargs):
            torch.cuda.synchronize()
            seconds = time.perf_counter() - self.step_started
            self.steps.append(seconds)
            record = {'step': state.global_step, 'epoch': state.epoch, 'seconds': seconds,
                      'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                      'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
            with (self.output / 'step_times.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
            if args.stop_after_step and state.global_step >= args.stop_after_step:
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            with (self.output / 'training_metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'step': state.global_step, 'epoch': state.epoch, **(logs or {})}) + '\n')

        def validate(self, state, model):
            if self.last_eval_step == state.global_step:
                return
            tag = f'step_{state.global_step:06d}'
            started = time.perf_counter()
            metrics = evaluate_validation(model, tokenizer, items, args.eval_batch_size, dtype,
                                          self.output / 'validation', tag)
            result = {'step': state.global_step, 'epoch': state.epoch, 'metrics': metrics,
                      'seconds': time.perf_counter() - started}
            if better_validation(metrics, self.best):
                model.save_pretrained(self.output / 'best_adapter', safe_serialization=True)
                tokenizer.save_pretrained(self.output / 'best_adapter')
                write_json(self.output / 'best_validation.json', result)
                self.best = metrics
            print('Validation:', json.dumps(result), flush=True)
            self.last_eval_step = state.global_step

        def on_epoch_end(self, training_args, state, control, model=None, **kwargs):
            self.validate(state, model)
            control.should_save = True
            return control

        def on_save(self, training_args, state, control, **kwargs):
            write_json(Path(training_args.output_dir) / f'checkpoint-{state.global_step}' / 'run_signature.json', signature)

        def on_train_end(self, training_args, state, control, model=None, **kwargs):
            self.validate(state, model)

    return RunCallback()


def main(argv=None):
    args = parse_args(argv)
    require(int(os.environ.get('WORLD_SIZE', '1')) == 1, 'This entry point supports one GPU only')
    model_name, model_kwargs = resolve_model(args)
    source_rows = load_source(args.source, args.local_train_file, args.dataset_revision)
    rows, identities, split, audit = validate_preferences(args.data_file, args.split_file, source_rows)
    require(args.n_pairs <= len(rows) and args.n_val <= len(split['val_indices']), 'Requested subset exceeds available data')
    rows = rows[:args.n_pairs or None]
    val_indices = split['val_indices'][:args.n_val or None]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
    require(tokenizer.is_fast and bool(tokenizer.chat_template), 'A fast tokenizer with a chat template is required')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    lengths = audit_lengths(rows, tokenizer, args.max_length)
    audit.update(selected_pairs=len(rows), selected_validation_questions=len(val_indices), lengths=lengths)
    if args.check_data:
        print(json.dumps(audit, indent=2))
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from trl import DPOConfig, DPOTrainer
    require(importlib.metadata.version('trl') == '1.16.0', 'Install train/dpo/requirements.txt (TRL 1.16.0 required)')
    require(torch.cuda.is_available(), 'CUDA is required for training. --check_data can run on CPU.')
    require(torch.cuda.device_count() == 1, 'Expose exactly one GPU with CUDA_VISIBLE_DEVICES')
    bf16 = torch.cuda.is_bf16_supported() if args.compute_dtype == 'auto' else args.compute_dtype == 'bf16'
    require(not bf16 or torch.cuda.is_bf16_supported(), 'The GPU does not support BF16')
    dtype = torch.bfloat16 if bf16 else torch.float16
    signature = run_signature(args, audit, model_name, tokenizer)
    checkpoint = resolve_resume(args.output_dir, args.resume, signature)
    if checkpoint and args.stop_after_step:
        require(args.stop_after_step > read_json(Path(checkpoint) / 'trainer_state.json')['global_step'],
                '--stop_after_step must exceed the resumed step')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'trl', 'peft', 'datasets', 'accelerate')}
    if args.quantization == '4bit':
        versions['bitsandbytes'] = importlib.metadata.version('bitsandbytes')
    if not checkpoint:
        write_json(args.output_dir / 'run_manifest.json', {
            'signature': signature, 'versions': versions, 'gpu': torch.cuda.get_device_name(0),
            'compute_dtype': str(dtype), 'reference': 'Frozen base model with adapters disabled',
            'selection_rule': 'Highest validation accuracy, then lowest validation NLL',
            'effective_lora_dropout': args.lora_dropout, 'disable_dropout': False,
        })
    set_seed(args.seed)
    started = time.perf_counter()
    kwargs = dict(model_kwargs, dtype=dtype, device_map={'': 0}, attn_implementation='sdpa')
    if args.quantization == '4bit':
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    require(args.max_length <= model.config.max_position_embeddings, 'Requested sequence length exceeds model context')
    model.config.use_cache = False
    disable_base_dropout(model)
    if args.quantization == '4bit':
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing,
                                               gradient_checkpointing_kwargs={'use_reentrant': False})
    targets = args.lora_target_modules if args.lora_target_modules == 'all-linear' else args.lora_target_modules.split(',')
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                            target_modules=targets, task_type='CAUSAL_LM', bias='none')
    config = DPOConfig(
        output_dir=str(args.output_dir / 'checkpoints'), num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate,
        beta=args.beta, loss_type=['sigmoid'], max_length=args.max_length, truncation_mode='keep_start',
        bf16=bf16, fp16=not bf16, gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False}, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, lr_scheduler_type='cosine',
        optim='adamw_torch', precompute_ref_log_probs=args.precompute_ref_log_probs,
        precompute_ref_batch_size=args.batch_size, disable_dropout=False,
        eval_strategy='no', save_strategy='steps', save_steps=args.save_steps,
        save_total_limit=args.save_total_limit, save_only_model=False,
        logging_steps=args.logging_steps, logging_first_step=True, report_to='none',
        seed=args.seed, data_seed=args.seed, dataloader_num_workers=0, dataset_num_proc=None,
        remove_unused_columns=True, push_to_hub=False,
    )
    callback = make_callback(args, tokenizer, validation_items(source_rows, val_indices), dtype, signature)
    trainer = DPOTrainer(model=model, ref_model=None, args=config, train_dataset=Dataset.from_list(rows),
                         processing_class=tokenizer, peft_config=peft_config, callbacks=[callback])
    require(len(trainer.train_dataset) == len(rows), 'Trainer unexpectedly dropped preference pairs')
    for row in trainer.train_dataset:
        require(max(len(row['prompt_ids']) + len(row[k]) for k in ('chosen_ids', 'rejected_ids')) <= args.max_length,
                'Trainer tokenization would truncate a response')
    # Keep small optimizer updates representable even when TRL casts QLoRA adapters to BF16.
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            require('lora_' in name, f'Unexpected trainable base parameter: {name}')
            param.data = param.data.float()
    trainer.model.print_trainable_parameters()
    write_json(args.output_dir / 'effective_training_config.json', config.to_dict())
    setup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.perf_counter() - started
    trainer.save_model(str(args.output_dir / 'last_adapter'))
    tokenizer.save_pretrained(args.output_dir / 'last_adapter')
    trainer.save_state()
    steps_per_epoch = math.ceil(math.ceil(len(rows) / args.batch_size) / args.gradient_accumulation_steps)
    steady = callback.steps[3:]
    mean_step = statistics.mean(steady) if steady else None
    write_json(args.output_dir / 'training_summary.json', {
        'global_step': trainer.state.global_step, 'epoch': trainer.state.epoch,
        'stopped_early': trainer.state.global_step < trainer.state.max_steps,
        'setup_seconds': setup_seconds, 'training_and_validation_seconds_this_session': elapsed,
        'optimizer_steps_this_session': len(callback.steps), 'steady_step_seconds': mean_step,
        'estimated_training_seconds_per_epoch': mean_step * steps_per_epoch if mean_step else None,
        'timing_note': 'Estimate excludes validation, checkpoint I/O, model loading, and reference preprocessing.',
        'effective_batch_pairs': args.batch_size * args.gradient_accumulation_steps,
        'steps_per_epoch_estimate': steps_per_epoch, 'trainer_metrics': result.metrics,
        'best_validation': read_json(args.output_dir / 'best_validation.json'),
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
    })
    print(f'Training finished. Artifacts: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
