# LEGO data and lifecycle checks

Run from the repository root with Python 3.11+, the project training dependencies,
and pytest:

```bash
python -m pytest -q tests
```

The tests use synthetic image patches and a tiny differentiable model. Data-flow
tests exercise both a Transformers backend and the real LlamaFactory backend
when installed. Lifecycle/training tests require LlamaFactory. No model or image
download is needed.

For two-process CPU/Gloo scoring and interrupted-training checks:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=src:tests \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
tests/lego_ddp_smoke.py /path/to/scratch-output
```

Use a separate scratch directory for each run. The DDP script normalizes
`torch.load(map_location=cpu:N)` to `cpu` inside the test, because the current
Accelerate/PyTorch 2.6 combination otherwise fails to load the optimizer on CPU.
Production checkpoint loading is not patched.

## Data contract

- `idx` identifies a row in the concatenated, preprocessed training dataset,
  including single-domain runs. A packed row may represent multiple raw records.
  Subsets, ordering and repeated draws retain the same ID.
- `MetadataCollator` removes scheduling metadata before the existing modality
  collator and restores it afterward. That collator must preserve row count and
  order. Unlabelled evaluation/anchor batches are valid; partially labelled
  batches fail explicitly.
- `split_batch` removes only `idx`, `domain_id`, and `__record_indices__` before
  model forward calls. Labels and all modality fields are preserved. Weight
  stages retain access to metadata.
- Training, pipeline components, ScoreBoard and reorder providers use the
  resolved collator, including its tokenizer/processor/template attributes.

## Component lifecycle

`LegoTrainer.training_step` calls `pipeline.prepare_loss_stages` before the
training forward. The base standalone weighter training step calls the same
`prepare` entry. `PlanContext` supplies the optimizer step, model and accelerator;
backward and optimizer operations remain with the Trainer.

- Multimodal Loss and DeltaLoss refresh their full-dataset weight table. LEGO
  indexes it by `W[batch["idx"]]`, without hashing text or flattened image patches.
  The legacy hash lookup remains for standalone datasets without IDs.
- Multimodal ADAPT refreshes its anchor embeddings in `prepare`. Current-batch
  embeddings still use a separate forward. No ADAPT feature-reuse optimization
  is implemented here.
- First activation always initializes state. Subsequent refreshes occur when
  `global_step % refresh_interval == 0`, at most once per optimizer step.
  Between refreshes, every training batch is weighted. A weight stage's
  `warmup_step` controls its own activation; trainer-level selection warmup does
  not implicitly change it. Weight-stage `every` must remain 1.
- Evaluation uses the ordinary loss and does not refresh component state.
- Scalar `get_weighted_loss` remains supported. Sample/token reductions and
  ADAPT's configured normalization retain their existing formulas.
- The `delta_loss` selector means initial loss minus current loss. Its first
  invocation establishes a baseline and returns the existing random selection;
  LEGO requires `update_times >= 2`. The `multimodal_deltaloss` weighter instead
  means no-image loss minus image-conditioned loss.
- Loss/DeltaLoss selector step-only JSON files are still written for inspection,
  but LEGO does not reuse them as a model/run cache. Filter selection requires
  no replacement and validates the returned IDs and budget.

## Checkpoints and coverage

Each Trainer checkpoint includes `dataflex_selection_rank_<rank>.json` for the
active sample sequence/batch cursor and `dataflex_lego_rank_<rank>.pt` for:

- pipeline plans, stage outputs and scheduling clocks;
- DeltaLoss selector's initial baseline;
- weight tables, ADAPT anchors and refresh clocks;
- ODM domain weights, estimated rewards, exploration rate and pending batch feedback;
- reorder pending queues/cursors and ScoreBoard values/timestamps.

Tensor/primitive-only state is loaded with `weights_only=True`. Resume verifies
dataset identity, component configuration, preprocessing information, schedule,
batch/worker settings and checkpoint step. External media identity includes
path, size and modification time; media and external score sources must remain
immutable snapshots. Old checkpoints lacking LEGO state are rejected.

Resume covers selectors loss/delta_loss/adapt_inf/random, mixers static/random/odm,
the existing static/dynamic reorder implementations (including named pattern
presets), and the three multimodal weighters plus legacy loss. ODM algorithms
and feedback attribution are unchanged: a mixed batch's loss still goes to its
first domain ID. Resume restores each rank's pending feedback without applying
it early or resetting the policy. Domain order, initial proportions and ODM
hyperparameters are validated. Older checkpoints marked `mixer:odm` unsupported,
or missing ODM component state, cannot be resumed; create a new checkpoint with
this version.

Tests cover ID alignment, refresh guards, scalar loss/gradient behavior,
standalone weighting, gradient accumulation, warmup/evaluation, and continuous
versus interrupted training at both interval boundaries and mid-interval. The
ODM cases cover warmup and already-updated policies, compare subsequent mixture
probabilities/rewards/pending feedback as well as samples/loss/model parameters,
and include a two-rank resume check in the DDP script. The
Adapt-inf test covers the LEGO adapter boundary; its gradient projection and
image-IG algorithm are not reimplemented or fully exercised by these fixtures.

These tests do not validate real VLM processors, CUDA, ZeRO/FSDP, benchmark
quality or GPU-hours. Cross-component sharing controls and efficiency metrics
remain subsequent work.
