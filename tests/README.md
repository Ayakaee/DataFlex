# LEGO data/interface checks

Run from the repository root with Python 3.11+ and `torch`, `numpy`,
`transformers`, `accelerate`, `datasets`, `pyyaml`, and `pytest` installed:

```bash
python -m pytest -q tests/test_lego_metadata.py tests/test_lego_data_flow.py
```

These CPU tests use synthetic image patches and a tiny differentiable model.
The data-flow tests execute the real `LegoTrainer`, `SelectTrainer`, pipeline
and ScoreBoard. They run against both the Transformers `Seq2SeqTrainer`
backend and, when installed, the actual LlamaFactory backend. The Transformers
variant skips eager registration of unrelated optional algorithms; the
LlamaFactory variant imports the normal component registry. They do not
validate a real VLM processor, a complete training loop, CUDA, DDP or DeepSpeed.

The interface established here is:

- `lego_get_dataset` assigns `idx` and `domain_id` after preprocessing and
  concatenation, including single-domain training. `idx` means a processed row
  in this dataset; a packed row may contain multiple original records. Subsets,
  reordering and repeated draws retain that ID. IDs must not be reused across
  different preprocessing results or a different order of domains.
- `MetadataCollator` takes scheduling metadata out before invoking the existing
  modality collator and restores it afterward. The collator must preserve row
  count and order. Unlabelled evaluation/anchor batches are supported. Mixed
  batches with partially missing metadata fail explicitly.
- `split_batch` removes only the reserved metadata keys (`idx`, `domain_id`,
  `__record_indices__`). Labels and all modality inputs are preserved. Weight
  stages still receive metadata; model forward calls do not.
- The resolved training collator is also used by pipeline components, the
  ScoreBoard and existing reorder score providers. The wrapper preserves
  tokenizer, processor and template attribute access.
- `PlanContext.from_trainer` provides `step_id`, `accelerator`, model, dataset
  and collator to components. It does not replace the Trainer used for backward
  or optimizer operations. Existing weighter lifecycle/context migration is
  subsequent work.

This part does not implement multimodal weight refresh under LEGO, replace
content-hash weight lookup, change ODM feedback, optimize ADAPT forwards, or
add cross-component sharing controls. A successful test run does not establish
that the full multimodal LEGO composition is ready to train.
