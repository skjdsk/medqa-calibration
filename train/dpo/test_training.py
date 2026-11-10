"""Offline regression tests for dataset integrity, resume, and the DPO integration."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from train.dpo.training_data import audit_lengths, object_hash, validate_preferences
from train.dpo.train import parse_args, resolve_resume, disable_base_dropout
from train.dpo.validation import better_validation, write_json
from train.grpo.grpo_data import build_messages


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.selected = self.root / 'run/selection_seed42'
        self.selected.mkdir(parents=True)
        self.raw = [dict(question=f'Question {i}', options=dict(A='Yes', B='No', C='Other', D='Unknown'),
                         answer_idx='A', answer='Yes') for i in range(3)]
        self.split = dict(train_indices=[0, 1], val_indices=[2], total_rows=3)
        write_json(self.root / 'split.json', self.split)
        canonical = [{k:r[k] for k in ('question','options','answer_idx','answer')} for r in self.raw]
        sampling = dict(train_indices=[0, 1], split_sha256=object_hash(self.split), training_rows_sha256=object_hash(canonical))
        write_json(self.selected.parent / 'sampling_manifest.json', {'config':sampling})
        write_json(self.selected / 'build_manifest.json', dict(retained_questions=1, sampling_config_sha256=object_hash(sampling)))
        self.rows = [dict(prompt=build_messages(self.raw[0]['question'], self.raw[0]['options']),
                          chosen=[dict(role='assistant',content='Reasoning. Answer: A')],
                          rejected=[dict(role='assistant',content='Reasoning. Answer: B')])]
        self.write_rows()
        self.write_ids(0)

    def tearDown(self):
        self.temp.cleanup()

    def write_rows(self):
        (self.selected / 'dpo.jsonl').write_text('\n'.join(map(json.dumps,self.rows)),encoding='utf-8')

    def write_ids(self,index):
        (self.selected / 'retained_questions.jsonl').write_text(json.dumps(dict(source_index=index,question_id=f'medqa4-{index}')),encoding='utf-8')

    def validate(self):
        return validate_preferences(self.selected/'dpo.jsonl',self.root/'split.json',self.raw)

    def test_valid_without_archived_samples(self):
        self.assertEqual(self.validate()[-1]['pairs'],1)

    def test_rejects_wrong_chosen_and_validation_leak(self):
        self.rows[0]['chosen'][0]['content']='Answer: B'
        self.write_rows()
        with self.assertRaisesRegex(ValueError,'Incorrect chosen'):self.validate()
        self.write_ids(2)
        with self.assertRaisesRegex(ValueError,'training index'):self.validate()

    def test_rejects_source_changes_and_prompt_mismatch(self):
        self.raw[0]['question']='Changed'
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.validate()
        self.raw[0]['question']='Question 0'
        self.rows[0]['prompt'][1]['content']='Another question'
        self.write_rows()
        with self.assertRaisesRegex(ValueError,'Prompt/source mismatch'):self.validate()

    def test_length_overflow_is_not_silently_truncated(self):
        class Tokenizer:
            def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False):
                return [1,2] if add_generation_prompt else [1,2,3,4,5]
        self.assertEqual(audit_lengths(self.rows,Tokenizer(),5)['max_sequence_tokens'],5)
        with self.assertRaisesRegex(ValueError,'never silently truncated'):
            audit_lengths(self.rows,Tokenizer(),4)

    def test_resume_rejects_nonempty_missing_or_changed_runs(self):
        out=self.root/'output'; out.mkdir()
        self.assertIsNone(resolve_resume(out,None,{'data':'a'}))
        write_json(out/'run_manifest.json',{'signature':{'data':'a'}})
        with self.assertRaisesRegex(ValueError,'not empty'):resolve_resume(out,None,{'data':'a'})
        with self.assertRaisesRegex(ValueError,'changed'):resolve_resume(out,'auto',{'data':'b'})
        with self.assertRaisesRegex(ValueError,'No checkpoint'):resolve_resume(out,'auto',{'data':'a'})
        ckpt=out/'checkpoints/checkpoint-5';ckpt.mkdir(parents=True)
        for name in ['trainer_state.json','optimizer.pt','scheduler.pt','adapter_config.json','adapter_model.safetensors','rng_state.pth']:
            (ckpt/name).write_text('{}')
        write_json(ckpt/'run_signature.json',{'data':'a'})
        self.assertEqual(Path(resolve_resume(out,'auto',{'data':'a'})),ckpt)

    def test_selection_and_cli(self):
        best=dict(accuracy=0.5,nll=0.8)
        self.assertTrue(better_validation(dict(accuracy=0.6,nll=1.0),best))
        self.assertTrue(better_validation(dict(accuracy=0.5,nll=0.7),best))
        self.assertFalse(better_validation(dict(accuracy=0.4,nll=0.1),best))
        args=parse_args(['--source','hf','--resume','--no-precompute_ref_log_probs'])
        self.assertEqual(args.resume,'auto')
        self.assertFalse(args.precompute_ref_log_probs)

    def test_baseline_uses_fixed_validation_and_separate_test(self):
        from train.dpo.baseline import main
        from unittest.mock import MagicMock
        output = self.root / 'baseline'
        model = MagicMock()
        model.config._commit_hash = 'revision'
        tokenizer = MagicMock()
        with patch('train.dpo.baseline.resolve_model', return_value=('base', {})), \
             patch('train.dpo.baseline.load_source', return_value=self.raw), \
             patch('train.dpo.baseline.load_test', return_value=self.raw[:2]), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.is_bf16_supported', return_value=True), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
             patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=model), \
             patch('train.dpo.baseline.evaluate_validation', return_value={'accuracy': 0.5}) as evaluate:
            arguments = ['--data_file', str(self.selected/'dpo.jsonl'), '--split_file', str(self.root/'split.json'),
                         '--output_dir', str(output), '--quantization', 'none']
            main(arguments)
            self.assertEqual(len(evaluate.call_args_list[0].args[2]), 1)
            self.assertEqual(evaluate.call_args_list[0].args[2][0].question_id, 'medqa4-2')
            self.assertEqual(len(evaluate.call_args_list[1].args[2]), 2)
            self.assertEqual(evaluate.call_args_list[1].args[2][0].question_id, 'medqa4-test-0')
            self.assertTrue((output/'summary.json').is_file())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                main(arguments)

    def test_validation_callback_saves_best_and_resume_signature(self):
        from train.dpo.train import make_callback
        args=parse_args(['--output_dir',str(self.root/'callback'),'--stop_after_step','2'])
        args.output_dir.mkdir()
        class Artifact:
            training=True
            def save_pretrained(self,path,**kwargs):
                Path(path).mkdir(parents=True,exist_ok=True)
                (Path(path)/'saved.txt').write_text('saved')
        model=Artifact()
        callback=make_callback(args,Artifact(),[],None,{'data':'fixed'})
        control=SimpleNamespace(should_save=False,should_training_stop=False)
        state=SimpleNamespace(global_step=2,epoch=0.5)
        training_args=SimpleNamespace(output_dir=str(args.output_dir/'checkpoints'))
        callback.step_started=0
        with patch('torch.cuda.synchronize'),patch('torch.cuda.max_memory_allocated',return_value=1024),patch('torch.cuda.max_memory_reserved',return_value=2048):
            callback.on_step_end(training_args,state,control)
        self.assertTrue(control.should_save and control.should_training_stop)
        with patch('train.dpo.train.evaluate_validation',return_value={'accuracy':0.5,'nll':0.7}) as evaluate:
            callback.on_epoch_end(training_args,state,control,model=model)
            callback.on_train_end(training_args,state,control,model=model)
            self.assertEqual(evaluate.call_count,1)
        callback.on_save(training_args,state,control)
        self.assertTrue((args.output_dir/'best_adapter/saved.txt').is_file())
        signature=json.loads((args.output_dir/'checkpoints/checkpoint-2/run_signature.json').read_text())
        self.assertEqual(signature,{'data':'fixed'})


@unittest.skipUnless(importlib.util.find_spec('trl') and importlib.util.find_spec('peft'), 'Optional training dependencies are not installed')
class TinyTrainerTests(unittest.TestCase):
    def test_adapter_updates_base_stays_frozen_and_checkpoint_resumes(self):
        """Use a random CPU model and a synthetic tokenizer; no downloads or pretrained training."""
        import torch
        from datasets import Dataset
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel, set_seed
        from peft import LoraConfig
        from trl import DPOConfig,DPOTrainer
        from train.dpo.validation import letter_table
        torch.set_num_threads(1)
        tok=Tokenizer(WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2,'user':3,'assistant':4,'Question':5,'Answer':6,'A':7,'B':8,'C':9,'D':10,':':11},unk_token='[UNK]'))
        tok.pre_tokenizer=Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
        tokenizer.chat_template="{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}{% if m['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
        tokenizer.padding_side='left'
        self.assertEqual(set(letter_table(tokenizer)),set('ABCD'))
        rows=[dict(prompt=[dict(role='user',content='Question')],chosen=[dict(role='assistant',content='Answer: A')],rejected=[dict(role='assistant',content='Answer: B')])]*4
        with tempfile.TemporaryDirectory() as tmp:
            def make_trainer():
                set_seed(1)
                model=GPT2LMHeadModel(GPT2Config(vocab_size=12,n_positions=32,n_embd=16,n_layer=1,n_head=2,bos_token_id=2,eos_token_id=2,pad_token_id=1))
                disable_base_dropout(model)
                config=DPOConfig(output_dir=tmp,max_steps=2,per_device_train_batch_size=1,gradient_accumulation_steps=2,
                    learning_rate=1e-3,max_length=32,bf16=False,fp16=False,use_cpu=True,gradient_checkpointing=False,
                    save_steps=1,logging_steps=1,report_to='none',disable_tqdm=True,precompute_ref_log_probs=True,disable_dropout=False,
                    dataloader_pin_memory=False,seed=1)
                return DPOTrainer(model=model,args=config,train_dataset=Dataset.from_list(rows),processing_class=tokenizer,
                                  peft_config=LoraConfig(r=2,lora_alpha=4,lora_dropout=0.2,target_modules=['c_attn'],task_type='CAUSAL_LM'))
            trainer=make_trainer()
            active=[m for n,m in trainer.model.named_modules() if 'lora_dropout' in n and isinstance(m,torch.nn.Dropout)]
            self.assertTrue(active)
            self.assertTrue(all(m.p == 0.2 for m in active))
            trainer.model.eval()
            self.assertTrue(all(not m.training for m in active))
            trainer.model.train()
            self.assertTrue(all(m.training for m in active))
            before={n:p.detach().clone() for n,p in trainer.model.named_parameters()}
            reference_batch=next(iter(trainer.get_train_dataloader()))
            reference_before=trainer.compute_ref_log_probs(trainer.model,reference_batch)
            result=trainer.train()
            self.assertEqual(trainer.state.global_step,2)
            self.assertTrue(math_isfinite(result.training_loss))
            self.assertTrue(any(not torch.equal(before[n],p) for n,p in trainer.model.named_parameters() if p.requires_grad))
            self.assertTrue(all(torch.equal(before[n],p) for n,p in trainer.model.named_parameters() if not p.requires_grad))
            reference=trainer.compute_ref_log_probs(trainer.model,reference_batch)
            self.assertTrue(all(torch.isfinite(v).all() for v in reference))
            for a,b in zip(reference,reference_before):torch.testing.assert_close(a,b)
            resumed=make_trainer()
            resumed.train(resume_from_checkpoint=str(Path(tmp)/'checkpoint-1'))
            self.assertEqual(resumed.state.global_step,2)
            for n,p in resumed.model.named_parameters():
                torch.testing.assert_close(p,dict(trainer.model.named_parameters())[n])


def math_isfinite(value):
    import math
    return math.isfinite(value)


if __name__=='__main__':
    unittest.main()
