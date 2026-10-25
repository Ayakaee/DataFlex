"""Lifecycle, composition and checkpoint regression tests on synthetic images."""

import copy
import importlib
from types import SimpleNamespace

import pytest
import torch
import yaml
from datasets import Dataset
from test_lego_data_flow import make_trainer
from test_lego_metadata import ImageCollator
from transformers import PretrainedConfig, Seq2SeqTrainingArguments, TrainerCallback
from transformers.modeling_outputs import CausalLMOutput

from dataflex.train.data.metadata import DomainLabeledDataset, MetadataCollator


@pytest.fixture(scope="module")
def lego():
    pytest.importorskip("llamafactory")
    from dataflex.train.lego import PlanContext, ScoreBoard
    from dataflex.train.trainer.lego_trainer import LegoTrainer

    return SimpleNamespace(Trainer=LegoTrainer, Context=PlanContext, Board=ScoreBoard, real_backend=True)


class FixtureTokenizer:
    padding_side = "right"

    def get_vocab(self):
        return {"a": 0, "b": 1, "c": 2, "image": 3}

    def save_pretrained(self, directory):
        return ()


class VisionCollator(ImageCollator):
    tokenizer = FixtureTokenizer()

    def __call__(self, rows):
        batch = super().__call__(rows)
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        return batch


class VisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))
        self.config = PretrainedConfig(image_token_id=3)
        self.calls = []

    def forward(
        self,
        input_ids,
        labels=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        output_hidden_states=False,
        use_cache=False,
    ):
        n = len(input_ids)
        image = (
            torch.zeros(n, device=input_ids.device) if pixel_values is None else pixel_values.reshape(n, 2).mean(-1)
        )
        self.calls.append((n, pixel_values is not None, torch.is_grad_enabled()))
        a = self.scale * (image[:, None] + input_ids.float())
        hidden = torch.stack((a, torch.ones_like(a)), dim=-1)
        logits = torch.stack((a, -a, a * 0.5, -a * 0.5), dim=-1)
        loss = (
            None
            if labels is None
            else torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 4),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        )
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=(hidden,) if output_hidden_states else None)


def dataset(n=8):
    rows = [{"input_ids": [0, 3, 1, 2], "labels": [-100, -100, 1, 2], "images": [float(i + 1)]} for i in range(n)]
    return DomainLabeledDataset(Dataset.from_list(rows), [i % 2 for i in range(n)])


def weighter(name, trainer, data):
    module_name, class_name = {
        "loss": ("multimodal_loss_weighter", "MultimodalLossWeighter"),
        "delta": ("multimodal_deltaloss_weighter", "MultimodalDeltalossWeighter"),
        "adapt": ("multimodal_adapt_weighter", "MultimodalAdaptWeighter"),
    }[name]
    cls = getattr(importlib.import_module(f"dataflex.train.weighter.{module_name}"), class_name)
    common = dict(data_collator=MetadataCollator(VisionCollator()), refresh_interval=2)
    if name == "adapt":
        return cls(eval_dataset=data.dataset.select([0, 2]), pool="answer", **common)
    return cls(dataset=data, accelerator=trainer.accelerator, score_batch_size=3, score_num_workers=0, **common)


@pytest.mark.parametrize("name", ["loss", "delta", "adapt"])
def test_refresh_once_per_optimizer_step_and_id_lookup(lego, tmp_path, name):
    trainer = make_trainer(lego, tmp_path)
    data, model = dataset(), VisionModel()
    component = weighter(name, trainer, data)
    ctx = lego.Context.from_trainer(trainer, model=model, dataset=data, dataset_size=len(data))
    component.prepare(ctx, model)
    first_calls = len(model.calls)
    assert first_calls > 0 and model.training
    component.prepare(ctx, model)
    ctx.step_id = 1
    component.prepare(ctx, model)
    assert len(model.calls) == first_calls
    ctx.step_id = 2
    component.prepare(ctx, model)
    assert len(model.calls) == first_calls * 2
    batch = component.data_collator([data[7], data[1], data[7]])
    losses = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    value = component.get_weighted_loss(losses, ctx=ctx, model=model, inputs=batch)
    assert torch.isfinite(value)
    value.backward()
    assert losses.grad is not None and torch.isfinite(losses.grad).all()
    if name != "adapt":
        assert component._weight_map == {}  # LEGO never hashes flattened vision patches.
        assert component.W[0] != component.W[7]
        torch.testing.assert_close(losses.grad, component.W[[7, 1, 7]] / 3)
    else:
        assert model.training and losses.grad[0] != losses.grad[1]


@pytest.mark.parametrize("name", ["loss", "delta", "adapt"])
def test_component_resume_retains_refresh_schedule_and_weights(lego, tmp_path, name):
    trainer = make_trainer(lego, tmp_path)
    data, model = dataset(), VisionModel()
    original = weighter(name, trainer, data)
    ctx = lego.Context.from_trainer(trainer, model=model, step_id=0)
    original.prepare(ctx, model)
    restored = weighter(name, trainer, data)
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    batch = original.data_collator([data[1], data[7]])
    ctx.step_id = 1
    before = len(model.calls)
    restored.prepare(ctx, model)
    assert len(model.calls) == before
    losses = torch.tensor([2.0, 3.0])
    torch.testing.assert_close(
        restored.get_weighted_loss(losses, ctx=ctx, model=model, inputs=batch),
        original.get_weighted_loss(losses, ctx=ctx, model=model, inputs=batch),
    )


def test_delta_selector_baseline_resume_does_not_need_cache_files(lego, tmp_path):
    from dataflex.train.selector.delta_loss_selector import DeltaLossSelector

    trainer = make_trainer(lego, tmp_path)
    data, model = dataset(), VisionModel()
    kwargs = dict(dataset=data, accelerator=trainer.accelerator, data_collator=MetadataCollator(VisionCollator()))
    original = DeltaLossSelector(cache_dir=str(tmp_path / "original"), **kwargs)
    original.reuse_selection_cache = False
    original.set_score_source(lambda positions: [float(i + 2) for i in positions])
    original.select(model, 0, 4, current_update_times=1, update_times=3)
    restored = DeltaLossSelector(cache_dir=str(tmp_path / "fresh"), **kwargs)
    restored.load_state_dict(original.state_dict())
    assert restored.path_to_initial_losses is None and not restored.first_time
    for component in (original, restored):
        component.set_score_source(lambda positions: [float(i + 1) for i in positions])
    assert restored.select(model, 2, 4, current_update_times=2, update_times=3) == original.select(
        model,
        2,
        4,
        current_update_times=2,
        update_times=3,
    )


def test_pipeline_checkpoint_restores_plan_reorder_and_scoreboard(lego, tmp_path):
    from dataflex.train.lego.checkpoint import pipeline_state, restore_pipeline, save_atomic
    from dataflex.train.lego.pipeline import SchedulePipeline
    from dataflex.train.lego.stages import ReorderStage
    from dataflex.train.reorder.dynamic_reorder import DynamicReorder

    data = dataset()
    trainer = make_trainer(lego, tmp_path)
    board = lego.Board(
        dataset=data, accelerator=trainer.accelerator, data_collator=MetadataCollator(VisionCollator()), num_workers=0
    )

    def new_pipeline():
        component = DynamicReorder(
            dataset=data, score_source="precomputed", score_params={"scores": list(range(len(data)))}, reorder_every=3
        )
        return SchedulePipeline([ReorderStage(component, "named_pattern_preset")], [], scoreboard=board)

    original = new_pipeline()
    ctx = lego.Context(dataset=data, dataset_size=len(data), interval_size=2, step_id=0)
    original.build_plan(ctx)
    board.put("loss", [1, 4], [0.1, 0.4], step_id=0)
    path = tmp_path / "state.pt"
    save_atomic(path, pipeline_state(original))
    restored = new_pipeline()
    restore_pipeline(restored, torch.load(path, weights_only=True), len(data))
    for step in (1, 2, 3, 4):
        ctx.step_id = step
        assert restored.build_plan(ctx).as_list() == original.build_plan(ctx).as_list()
    assert restored.scoreboard._scored_at["loss"][4] == 0


def test_weight_warmup_prepare_and_eval_do_not_refresh(lego, tmp_path):
    from dataflex.train.lego.stages import WeightStage

    trainer = make_trainer(lego, tmp_path)
    data, model = dataset(), VisionModel()
    component = weighter("loss", trainer, data)
    trainer.pipeline.loss_stages = [WeightStage(component, "multimodal_loss", warmup_step=2)]
    for step in (0, 1):
        ctx = trainer.plan_context(0, step, model=model)
        trainer.pipeline.prepare_loss_stages(ctx, model)
    assert component.W is None
    trainer.pipeline.prepare_loss_stages(trainer.plan_context(0, 2, model=model), model)
    assert component._last_refresh_step == 2
    model.eval()
    trainer.state.global_step = 4
    trainer.compute_loss(model, component.data_collator([data[0], data[2]]))
    assert component._last_refresh_step == 2


def test_checkpoint_rejects_changed_data_and_odm_resume(lego, tmp_path):
    from dataflex.train.lego.checkpoint import dataset_identity, pipeline_state, restore_pipeline

    left, right = dataset(), dataset()
    right.dataset = right.dataset.add_column("different", [True] * len(right))
    assert dataset_identity(left) != dataset_identity(right)
    trainer = make_trainer(lego, tmp_path)
    state = pipeline_state(trainer.pipeline)
    state["unsupported"] = ["mixer:odm"]
    with pytest.raises(ValueError, match="ODM"):
        restore_pipeline(trainer.pipeline, state, len(trainer.train_dataset))


def training_trainer(lego, directory, selector, mixer, reorder, weight):
    from dataflex.train.hparams.dynamic_params import DynamicFinetuningArguments

    directory.mkdir(parents=True, exist_ok=True)
    data = dataset(16)
    cfg = {
        "pipelines": {
            "test": {
                "params": {
                    "score_num_workers": 0,
                    "stages": [
                        {
                            "family": "selector",
                            "component": selector,
                            "mode": "filter",
                            "keep_ratio": 0.75,
                            "every": 4,
                        },
                        {"family": "mixer", "component": mixer},
                        {"family": "reorder", "component": reorder},
                        {"family": "weighter", "component": weight},
                    ],
                }
            }
        },
        "selectors": {selector: {"params": {"cache_dir": str(directory / "selection"), "score_num_workers": 0}}},
        "mixers": {mixer: {"params": {"proportions": [0.5, 0.5]} if mixer == "static" else {"seed": 7}}},
        "reorders": {
            reorder: {
                "params": {
                    "apply_at": "index",
                    "score_source": "precomputed",
                    "score_params": {"scores": list(range(16))},
                }
            }
        },
        "weighters": {weight: {"params": {"refresh_interval": 3, "score_num_workers": 0, "score_batch_size": 4}}},
    }
    path = directory / "components.yaml"
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        path.write_text(yaml.safe_dump(cfg))
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    args = Seq2SeqTrainingArguments(
        output_dir=str(directory),
        use_cpu=True,
        report_to=[],
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=2,
        logging_steps=1,
        disable_tqdm=True,
        dataloader_pin_memory=False,
        seed=123,
    )
    args.fp8 = False
    fa = DynamicFinetuningArguments(
        component_name="test",
        components_cfg_file=str(path),
        warmup_step=0,
        update_step=2,
        update_times=3,
        train_step=6,
        disable_shuffling=True,
    )
    trainer = lego.Trainer(
        model=VisionModel(),
        tokenizer=None,
        args=args,
        finetuning_args=fa,
        train_dataset=data,
        eval_dataset=data.dataset.select([0, 2]),
        data_collator=VisionCollator(),
        domain_ids=data.domain_ids,
        domain_names=["first", "second"],
    )
    trainer.trace = []
    component = trainer.pipeline.loss_stages[0].component
    original = component.get_weighted_loss

    def record(losses, *, ctx, model, inputs):
        loss = original(losses, ctx=ctx, model=model, inputs=inputs)
        trainer.trace.append((ctx.step_id, inputs["idx"].tolist(), float(loss.detach()), component._last_refresh_step))
        return loss

    component.get_weighted_loss = record
    return trainer


@pytest.mark.parametrize(
    "selector,mixer,reorder,weight",
    [
        ("loss", "static", "dynamic", "multimodal_loss"),
        ("delta_loss", "random", "static", "multimodal_deltaloss"),
        ("loss", "static", "dynamic", "multimodal_adapt"),
    ],
)
@pytest.mark.parametrize("stop_step", [1, 2])
def test_interrupted_training_matches_continuous_run(lego, tmp_path, selector, mixer, reorder, weight, stop_step):
    class StopAfterCheckpoint(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == stop_step:
                control.should_training_stop = True
                control.should_save = True
            return control

    full = training_trainer(lego, tmp_path / "full", selector, mixer, reorder, weight)
    full.train()
    partial = training_trainer(lego, tmp_path / "partial", selector, mixer, reorder, weight)
    partial.add_callback(StopAfterCheckpoint())
    partial.train()
    assert partial.state.global_step == stop_step
    checkpoint = tmp_path / f"partial/checkpoint-{stop_step}"
    assert (checkpoint / "dataflex_lego_rank_0.pt").exists()
    resumed = training_trainer(lego, tmp_path / "resumed", selector, mixer, reorder, weight)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    actual = partial.trace + resumed.trace
    assert [(s, ids, refresh) for s, ids, _, refresh in actual] == [
        (s, ids, refresh) for s, ids, _, refresh in full.trace
    ]
    torch.testing.assert_close(torch.tensor([row[2] for row in actual]), torch.tensor([row[2] for row in full.trace]))
    torch.testing.assert_close(resumed.model.scale, full.model.scale)
    assert resumed.state.global_step == full.state.global_step == 6
    if selector == "loss" and weight == "multimodal_loss" and stop_step == 1:
        changed = training_trainer(lego, tmp_path / "changed", selector, mixer, reorder, weight)
        changed.train_dataset.dataset = changed.train_dataset.dataset.add_column("changed", [1] * 16)
        with pytest.raises(ValueError, match="dataset/config/schedule/step"):
            changed.train(resume_from_checkpoint=str(checkpoint))


def test_accumulation_scaling_for_models_accepting_loss_kwargs(lego, tmp_path):
    trainer = training_trainer(lego, tmp_path, "loss", "static", "dynamic", "multimodal_loss")
    model = trainer.model
    component = trainer.pipeline.loss_stages[0].component
    component.W = torch.ones(len(trainer.train_dataset))
    component._last_refresh_step = 0
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer.current_gradient_accumulation_steps = 2
    trainer.model_accepts_loss_kwargs = True
    rows = [trainer.train_dataset[i] for i in range(4)]
    full = trainer.data_collator(rows)
    expected_loss = trainer.compute_loss(model, full)
    expected_loss.backward()
    expected = model.scale.grad.clone()
    model.zero_grad()
    for part in (rows[:2], rows[2:]):
        trainer.training_step(model, trainer.data_collator(part), num_items_in_batch=torch.tensor(8))
    torch.testing.assert_close(model.scale.grad, expected)
    assert trainer.model_accepts_loss_kwargs is True


@pytest.mark.parametrize("name", ["loss", "delta", "adapt"])
def test_standalone_weight_trainer_uses_same_prepare_entry(lego, tmp_path, name):
    from dataflex.train.hparams.dynamic_params import DynamicFinetuningArguments
    from dataflex.train.trainer.weight_trainer import WeightTrainer

    registered = {"loss": "multimodal_loss", "delta": "multimodal_deltaloss", "adapt": "multimodal_adapt"}[name]
    path = tmp_path / "weights.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "weighters": {
                    registered: {
                        "params": {
                            "refresh_interval": 2,
                            "score_num_workers": 0,
                            "score_batch_size": 3,
                        }
                    }
                }
            }
        )
    )
    data = dataset()
    args = Seq2SeqTrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[])
    args.fp8 = False
    trainer = WeightTrainer(
        model=VisionModel(),
        tokenizer=None,
        args=args,
        finetuning_args=DynamicFinetuningArguments(component_name=registered, components_cfg_file=str(path)),
        train_dataset=data.dataset,
        eval_dataset=data.dataset.select([0, 2]),
        data_collator=VisionCollator(),
    )
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
    trainer.current_gradient_accumulation_steps = 1
    for step in (0, 0, 1, 2):
        trainer.state.global_step = step
        batch = trainer.data_collator([data.dataset[0], data.dataset[3]])
        loss = trainer.training_step(trainer.model, batch)
        assert torch.isfinite(loss)
        assert trainer.weighter._last_refresh_step == (2 if step == 2 else 0)


def test_adapt_inf_stage_passes_pool_budget_and_runtime(lego, tmp_path, monkeypatch):
    from dataflex.train.lego.plan import DataPlan
    from dataflex.train.lego.stages import SelectStage
    from dataflex.train.selector.adapt_inf_selector import AdaptInfSelector

    trainer = make_trainer(lego, tmp_path)
    data = dataset()
    component = AdaptInfSelector(
        dataset=data,
        accelerator=trainer.accelerator,
        data_collator=MetadataCollator(VisionCollator()),
        cache_dir=str(tmp_path),
        num_clusters=1,
        cluster_grid=[1],
        proj_dim=8,
        parameter_pattern="scale",
    )
    calls = []

    def select(model, step_id, num_samples, **kwargs):
        calls.append((component.candidate_positions(), step_id, num_samples, kwargs["tokenizer"]))
        return component.candidate_positions()[:num_samples]

    # The algorithm's gradient projection/IG is unchanged; this tests its LEGO adapter boundary.
    monkeypatch.setattr(component, "select", select)
    ctx = lego.Context(dataset=data, dataset_size=len(data), step_id=4, tokenizer="tokenizer", model=VisionModel())
    result = SelectStage(component, "adapt_inf", mode="filter", keep_ratio=0.5).apply(DataPlan.over([7, 3, 0, 2]), ctx)
    assert result.as_list() == [7, 3]
    assert calls == [([7, 3, 0, 2], 4, 2, "tokenizer")]


def test_missing_ids_and_filter_replacement_are_rejected(lego, tmp_path):
    from dataflex.train.lego.stages import SelectStage

    trainer = make_trainer(lego, tmp_path)
    data, model = dataset(), VisionModel()
    component = weighter("loss", trainer, data)
    component.prepare(lego.Context.from_trainer(trainer), model)
    batch = component.data_collator([data[0], data[1]])
    batch.pop("idx")
    with pytest.raises(ValueError, match="idx is required"):
        component.get_weighted_loss(torch.ones(2), inputs=batch)
    selector = trainer.pipeline.index_stages[0].component
    selector.replacement = True
    with pytest.raises(ValueError, match="replacement: false"):
        SelectStage(selector, "loss", mode="filter")
