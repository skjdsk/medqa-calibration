Final training datasets
=======================
selection_seed42/dpo.jsonl: 3322 preference pairs.
selection_seed42/sft.jsonl: 3322 matching chosen responses for supervised fine-tuning.
Both exports come from the same retained questions, sampled from 5000 MedQA candidates with eight responses per question.
Every SFT prompt/completion was verified against the corresponding DPO prompt/chosen response.
Final-answer labels were checked during construction; this does not certify the medical correctness of every generated rationale.

Keep the sampling/build manifests, retained question indices, selection audit, summary, and download audit for provenance.
Raw samples and detailed generation logs remain in the sibling backup directory under cleanup_20261105.
The SFT export was restored from that archive on 2026-11-10.
Experiment outputs and caches were moved to the sibling backup directory under experiment_outputs_cleanup_20261110.
The executed preparation notebook is a Colab run record; restore raw sampled inputs before rerunning its build/statistics cells.
