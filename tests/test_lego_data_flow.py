"""Exercise real LEGO/SelectTrainer code on the CPU Transformers backend.

One fixture variant uses Seq2SeqTrainer without unrelated optional algorithms;
another uses the real LlamaFactory backend when installed. Neither variant
claims to test real VLM processors, GPU training or distributed runs.
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from test_lego_metadata import ImageCollator, image_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.modeling_outputs import CausalLMOutput

from dataflex.train.data.metadata import MetadataCollator


@pytest.fixture(scope="module", params=["transformers", "llamafactory"])
def lego(request):
    if request.param == "llamafactory":
        pytest.importorskip("llamafactory")
        from dataflex.train.lego import PlanContext, ScoreBoard
        from dataflex.train.trainer.lego_trainer import LegoTrainer

        yield SimpleNamespace(Trainer=LegoTrainer, Context=PlanContext, Board=ScoreBoard, real_backend=True)
        return
    # Only the package-registration side effects and LF backend are replaced;
    # context, pipeline, stages, selectors, scoreboard and both trainers are real.
    train_root = Path(__file__).resolve().parents[1] / "src/dataflex/train"
    before = set(sys.modules)
    with pytest.MonkeyPatch.context() as patch:
        for family in ("lego", "selector", "weighter"):
            name = f"dataflex.train.{family}"
            package = ModuleType(name)
            package.__path__ = [str(train_root / family)]
            patch.setitem(sys.modules, name, package)
        context = importlib.import_module("dataflex.train.lego.context")
        pipeline = importlib.import_module("dataflex.train.lego.pipeline")
        board = importlib.import_module("dataflex.train.lego.scoreboard")
        package = sys.modules["dataflex.train.lego"]
        package.PlanContext = context.PlanContext
        package.SchedulePipeline = pipeline.SchedulePipeline
        package.ScoreBoard = board.ScoreBoard
        for module in ("selector.loss_selector", "selector.delta_loss_selector", "weighter.loss_weighter"):
            importlib.import_module(f"dataflex.train.{module}")

        class CPUBackend(Seq2SeqTrainer):
            def __init__(self, finetuning_args, processor=None, model_args=None, gen_kwargs=None, **kwargs):
                self.finetuning_args = finetuning_args
                super().__init__(**kwargs)

        backend = ModuleType("llamafactory.train.sft.trainer")
        backend.CustomSeq2SeqTrainer = CPUBackend
        patch.setitem(sys.modules, backend.__name__, backend)
        trainer = importlib.import_module("dataflex.train.trainer.lego_trainer")
        yield SimpleNamespace(
            Trainer=trainer.LegoTrainer,
            Context=context.PlanContext,
            Board=board.ScoreBoard,
            real_backend=False,
        )
    # Do not leave test-specific module imports/registry entries in other tests.
    for name in set(sys.modules) - before:
        if name.startswith("dataflex."):
            sys.modules.pop(name, None)


class TinyVisionModel(torch.nn.Module):
    """Strict forward signature catches leaked IDs; pixels affect the answer."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        input_ids,
        labels=None,
        pixel_values=None,
        image_grid_thw=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        batch_size, length = input_ids.shape
        if pixel_values is None:
            vision = torch.zeros(batch_size)
        else:
            assert image_grid_thw.shape == (batch_size, 3)
            vision = pixel_values.reshape(batch_size, 2).mean(-1)
        hidden = (input_ids.float() + vision[:, None]).unsqueeze(-1) * self.scale
        logits = hidden * torch.arange(3, dtype=torch.float)
        loss = (
            None
            if labels is None
            else F.cross_entropy(
                logits[:, :-1].reshape(-1, 3),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        )
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=(hidden,) if output_hidden_states else None)


def make_trainer(lego, tmp_path, *, weighted=True, prune=True):
    stages = [{"family": "selector", "component": "loss", "mode": "draw"}]
    if weighted:
        stages.append({"family": "weighter", "component": "loss"})
    cfg = tmp_path / "components.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "pipelines": {"test": {"params": {"stages": stages, "score_num_workers": 0}}},
                "selectors": {"loss": {"params": {"cache_dir": str(tmp_path / "scores"), "score_num_workers": 0}}},
                "weighters": {"loss": {"params": {"strategy": "uniform"}}},
            }
        )
    )
    finetuning_kwargs = dict(
        component_name="test",
        components_cfg_file=str(cfg),
        update_times=2,
        disable_shuffling=True,
    )
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        per_device_train_batch_size=3,
        remove_unused_columns=prune,
        dataloader_pin_memory=False,
    )
    backend_kwargs = {}
    if lego.real_backend:
        from dataflex.train.hparams.dynamic_params import DynamicFinetuningArguments

        finetuning_args = DynamicFinetuningArguments(**finetuning_kwargs)
        # LF normally adds this field in its argument parser. No FP8 is used here.
        args.fp8 = False
        backend_kwargs["tokenizer"] = None
    else:
        finetuning_args = SimpleNamespace(**finetuning_kwargs)
    return lego.Trainer(
        finetuning_args=finetuning_args,
        model=TinyVisionModel(),
        args=args,
        train_dataset=image_dataset(),
        data_collator=ImageCollator(),
        domain_ids=np.array([0, 1]),
        domain_names=["first", "second"],
        **backend_kwargs,
    )


@pytest.mark.parametrize("prune", [True, False])
@pytest.mark.parametrize("weighted", [True, False])
def test_training_and_scoring_keep_metadata_outside_model(lego, tmp_path, prune, weighted):
    trainer = make_trainer(lego, tmp_path, weighted=weighted, prune=prune)
    loader = trainer.get_train_dataloader([1, 0, 1])
    batch = next(iter(loader))
    assert batch["idx"].tolist() == [1, 0, 1]
    assert batch["domain_id"].tolist() == [1, 0, 1]
    resolved = trainer.selector.data_collator
    assert resolved.tokenizer == ImageCollator.tokenizer
    assert resolved.processor == ImageCollator.processor
    assert trainer.scoreboard.data_collator is resolved
    assert all(stage.component.data_collator is resolved for stage in trainer.pipeline.index_stages)
    assert trainer.plan_context(3, 5).data_collator is resolved

    if weighted:
        # The scalar weighter interface must retain both metadata and gradients.
        def weighted_loss(losses, *, ctx, model, inputs):
            assert inputs["idx"].tolist() == [1, 0, 1]
            assert inputs["domain_id"].tolist() == [1, 0, 1]
            assert inputs["pixel_values"] is batch["pixel_values"]
            assert ctx.accelerator is trainer.accelerator
            return (losses * torch.tensor([2.0, 1.0, 2.0])).mean()

        trainer.pipeline.loss_stages[0].component.get_weighted_loss = weighted_loss
    loss = trainer.compute_loss(trainer.model, batch)
    loss.backward()
    assert torch.isfinite(loss) and trainer.model.scale.grad.abs() > 0
    assert "labels" in batch and "idx" in batch

    scores = trainer.scoreboard._compute_losses(trainer.model, [1, 0, 1], step_id=0)
    assert set(scores) == {0, 1}
    assert scores[0] != scores[1]  # Same text, different image, distinct IDs/scores.
    assert trainer.model.training

    selected = trainer.selector.select(trainer.model, step_id=0, num_samples=2)
    assert sorted(selected) == [0, 1]
    assert trainer._active_plan.as_list() == selected


def test_hf_dataset_column_pruning_keeps_raw_images_and_metadata(lego, tmp_path):
    trainer = make_trainer(lego, tmp_path)
    dataset = trainer.train_dataset.dataset.add_column("idx", [0, 1]).add_column("domain_id", [0, 1])
    dataset = dataset.add_column("unrelated_column", ["drop", "drop"])
    trainer.train_dataset = dataset
    batch = next(iter(trainer.get_train_dataloader()))
    assert batch["idx"].tolist() == [0, 1]
    assert batch["pixel_values"].flatten().tolist() == [1, 1, 5, 5]
    columns = trainer._remove_unused_columns(dataset).column_names
    assert "unrelated_column" not in columns
    assert set(columns) >= {"images", "idx", "domain_id"}


def test_context_conversion_and_explicit_planning_overrides(lego, tmp_path):
    trainer = make_trainer(lego, tmp_path)
    trainer.state.global_step = 7
    ctx = lego.Context.from_trainer(trainer)
    assert ctx.step_id == 7 and ctx.dataset_size == 2
    assert ctx.accelerator.device.type == "cpu"
    assert ctx.accelerator.is_main_process
    ctx = trainer.plan_context(5, 9, tokenizer="explicit", current_update_times=2)
    assert (ctx.step_id, ctx.interval_size, ctx.current_update_times) == (9, 5, 2)
    assert ctx.as_component_kwargs()["tokenizer"] == "explicit"


def test_existing_reorder_provider_gets_resolved_collator(lego, tmp_path):
    from dataflex.train.lego.stages import ReorderStage
    from dataflex.train.reorder.dynamic_reorder import DynamicReorder

    trainer = make_trainer(lego, tmp_path)
    reorder = DynamicReorder(
        dataset=trainer.train_dataset,
        accelerator=trainer.accelerator,
        data_collator=trainer.data_collator,
        score_source="model_loss",
    )
    provider = reorder._ensure_provider()
    trainer.pipeline.index_stages.append(ReorderStage(reorder, "dynamic"))
    trainer.get_train_dataloader([1, 0])
    resolved = trainer.selector.data_collator
    assert reorder.data_collator is resolved and provider.data_collator is resolved
    assert isinstance(resolved, MetadataCollator)
