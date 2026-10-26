"""Shared/independent scoring parity, cache conditions, and resume contracts."""

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_lego_lifecycle import VisionCollator, VisionModel, dataset, training_trainer
from transformers import TrainerCallback

from dataflex.train.data.metadata import MetadataCollator


@pytest.fixture(scope="module")
def lego():
    pytest.importorskip("llamafactory")
    from dataflex.train.lego import ScoreBoard
    from dataflex.train.trainer.lego_trainer import LegoTrainer

    return SimpleNamespace(Trainer=LegoTrainer, Board=ScoreBoard)


def board(lego, mode="shared", data=None, batch_size=2):
    return lego.Board(
        dataset=dataset(7) if data is None else data, data_collator=MetadataCollator(VisionCollator()),
        batch_size=batch_size, num_workers=0, score_sharing=mode,
    )


def test_partial_pool_reuse_and_separate_ablation(lego):
    results, samples = {}, {}
    for mode in ("shared", "independent"):
        cache, model = board(lego, mode), VisionModel()
        first = cache.loss_scores(model, [6, 0, 6], 0, consumer="selector")
        reordered = cache.loss_scores(model, [0, 6], 0, consumer="reorder")
        full = cache.loss_scores(model, range(7), 0, consumer="weighter")
        np.testing.assert_allclose(first, full[[6, 0, 6]])
        np.testing.assert_allclose(reordered, full[[0, 6]])
        samples[mode] = sum(n for n, _, _ in model.calls)
        before = len(model.calls)
        np.testing.assert_array_equal(full, cache.loss_scores(model, range(7), 0, consumer="weighter"))
        assert len(model.calls) == before  # Own cache works in both modes.
        no_image = cache.loss_scores(model, range(7), 0, consumer="weighter", metric="loss_no_image")
        assert all(not with_image for _, with_image, _ in model.calls[before:])
        assert not np.allclose(full, no_image)
        before = sum(n for n, _, _ in model.calls)
        cache.loss_scores(model, range(7), 1, consumer="weighter")
        assert sum(n for n, _, _ in model.calls) - before == 7  # No cross-step reuse.
        results[mode] = (full, no_image)
    assert samples == {"shared": 7, "independent": 11}
    np.testing.assert_allclose(results["shared"], results["independent"])


def test_invalid_answer_is_cached_as_nan_and_model_change_invalidates(lego):
    data = dataset(2)
    labels = [[-100] * 4, [-100, -100, 1, 2]]
    data.dataset = data.dataset.remove_columns("labels").add_column("labels", labels)
    cache, model = board(lego, data=data), VisionModel()
    values = cache.loss_scores(model, [0, 1], 0, consumer="selector")
    assert np.isnan(values[0]) and np.isfinite(values[1])
    calls = len(model.calls)
    assert np.isnan(cache.loss_scores(model, [0], 0, consumer="weighter")[0])
    assert len(model.calls) == calls  # Known-invalid differs from never scored.
    other = VisionModel()
    with torch.no_grad():
        other.scale.fill_(0.7)
    assert cache.loss_scores(other, [1], 0, consumer="weighter")[0] != values[1]
    assert other.calls


@pytest.mark.parametrize("fail", [False, True])
def test_scoring_preserves_rng_modes_rope_and_gradients(lego, fail):
    class ProbeModel(VisionModel):
        def forward(self, *args, **kwargs):
            random.random()
            np.random.random()
            torch.rand(3)
            self.rope_deltas = torch.tensor([99])
            if fail:
                raise RuntimeError("scoring failure")
            return super().forward(*args, **kwargs)

    cache, model = board(lego), ProbeModel()
    model.add_module("child", torch.nn.Dropout().eval())
    model.rope_deltas = torch.tensor([7])
    model.scale.grad = torch.tensor(2.0)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    if fail:
        with pytest.raises(RuntimeError, match="scoring failure"):
            cache.loss_scores(model, [0, 1], 0, consumer="selector")
        assert not cache._values
    else:
        cache.loss_scores(model, [0, 1], 0, consumer="selector")
    assert random.getstate() == py_state
    np.testing.assert_equal(np.random.get_state(), np_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert model.training and not model.child.training
    assert model.rope_deltas.item() == 7 and model.scale.grad.item() == 2.0


def test_shared_scores_match_standalone_weight_formulas(lego, tmp_path):
    from test_lego_lifecycle import weighter

    from dataflex.train.lego import PlanContext

    trainer = training_trainer(lego, tmp_path, "loss", "static", "dynamic", "multimodal_loss")
    for name in ("loss", "delta"):
        standalone = weighter(name, trainer, trainer.train_dataset)
        shared = weighter(name, trainer, trainer.train_dataset)
        ctx = PlanContext.from_trainer(trainer)
        standalone.prepare(ctx, trainer.model)
        shared.set_score_source(lambda positions, metric: trainer.scoreboard.loss_scores(
            trainer.model, positions, 0, consumer="weighter", metric=metric,
        ))
        shared.prepare(ctx, trainer.model)
        torch.testing.assert_close(shared.W, standalone.W)


@pytest.mark.parametrize("selector,weight", [
    ("loss", "multimodal_loss"), ("delta_loss", "multimodal_deltaloss"), ("loss", "multimodal_adapt"),
])
def test_shared_independent_training_parity(lego, tmp_path, selector, weight):
    runs = {}
    for mode in ("shared", "independent"):
        trainer = training_trainer(
            lego, tmp_path / mode, selector, "static", "dynamic", weight,
            score_sharing=mode, score_source="model_loss", refresh_interval=2, score_batch_size=3,
        )
        trainer.train()
        runs[mode] = trainer
    shared, independent = runs["shared"], runs["independent"]
    assert [(s, ids, r) for s, ids, _, r in shared.trace] == [(s, ids, r) for s, ids, _, r in independent.trace]
    torch.testing.assert_close(torch.tensor([x[2] for x in shared.trace]), torch.tensor([x[2] for x in independent.trace]))
    torch.testing.assert_close(shared.model.scale, independent.model.scale)
    if weight != "multimodal_adapt":
        torch.testing.assert_close(shared.pipeline.loss_stages[0].component.W, independent.pipeline.loss_stages[0].component.W)
    assert sum(n for n, _, grad in shared.model.calls if not grad) < sum(
        n for n, _, grad in independent.model.calls if not grad
    )
    assert "loss_full" in shared.scoreboard._values
    assert all("::" in key for key in independent.scoreboard._values)


@pytest.mark.parametrize("mode", ["shared", "independent"])
def test_shared_scoring_resume_and_contract_validation(lego, tmp_path, mode):
    class Stop(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 3:
                control.should_save = control.should_training_stop = True
            return control

    def trainer(folder, sharing=mode):
        return training_trainer(
            lego, tmp_path / folder, "delta_loss", "odm", "dynamic", "multimodal_deltaloss",
            score_sharing=sharing, score_source="model_loss", refresh_interval=2,
        )

    full, partial = trainer("full"), trainer("partial")
    full.train()
    partial.add_callback(Stop())
    partial.train()
    path = str(tmp_path / "partial/checkpoint-3")
    resumed = trainer("resumed")
    resumed.train(resume_from_checkpoint=path)
    actual = partial.trace + resumed.trace
    assert [(s, ids, r) for s, ids, _, r in actual] == [(s, ids, r) for s, ids, _, r in full.trace]
    torch.testing.assert_close(torch.tensor([x[2] for x in actual]), torch.tensor([x[2] for x in full.trace]))
    torch.testing.assert_close(resumed.model.scale, full.model.scale)
    for key in full.scoreboard._values:
        np.testing.assert_allclose(resumed.scoreboard._values[key], full.scoreboard._values[key])
        np.testing.assert_array_equal(resumed.scoreboard._scored_at[key], full.scoreboard._scored_at[key])
    other = trainer("wrong-mode", "independent" if mode == "shared" else "shared")
    with pytest.raises(ValueError, match="dataset/config/schedule/step"):
        other.train(resume_from_checkpoint=path)
    state = copy.deepcopy(full.scoreboard.state_dict())
    state["contract"]["no_image"] = "zero-pixels"
    with pytest.raises(ValueError, match="scoring contract"):
        resumed.scoreboard.load_state_dict(state)


def test_adapt_inf_publication_only_shares_compatible_full_loss(lego, tmp_path):
    from dataflex.train.selector.adapt_inf_selector import AdaptInfSelector
    from dataflex.utils.multimodal_scoring import answer_statistics

    cache, model = board(lego, batch_size=1), VisionModel()
    positions = [6, 0, 3]
    computed = []
    for position in positions:
        batch = cache.data_collator([cache.dataset[position]])
        outputs = model(input_ids=batch["input_ids"], pixel_values=batch["pixel_values"])
        # The same differentiable, answer-only loss used by ADAPT-INF.
        stats = answer_statistics(outputs.logits, batch["labels"])
        torch.autograd.grad(stats["loss"], [model.scale])
        computed.append(float(stats["loss"].detach()))
    np.save(tmp_path / "dataset_positions.npy", np.asarray(positions))
    scores = np.zeros((3, 4))
    scores[:, 0] = computed
    scores[:, 3] = 999  # IG must never be mislabelled as delete-image loss.
    np.save(tmp_path / "scores.npy", scores)
    publisher = cache.adapt_inf_publisher(model, 0, consumer="selector:adapt_inf")
    assert publisher is not None
    # Publishing has no dependency on projection/clustering implementation.
    component = object.__new__(AdaptInfSelector)
    component._publish_loss_scores(tmp_path, positions, publisher)
    before = len(model.calls)
    np.testing.assert_allclose(cache.loss_scores(model, positions, 0, consumer="weighter"), scores[:, 0])
    assert len(model.calls) == before
    independent = board(lego, "independent", batch_size=1)
    np.testing.assert_allclose(independent.loss_scores(model, positions, 0, consumer="weighter"), scores[:, 0])
    before = len(model.calls)
    cache.loss_scores(model, positions, 0, consumer="weighter", metric="loss_no_image")
    assert len(model.calls) > before
    # Missing old score artifacts simply leave the board empty.
    (tmp_path / "scores.npy").unlink()
    other = board(lego, batch_size=1)
    component._publish_loss_scores(tmp_path, positions, other.adapt_inf_publisher(model, 0, consumer="selector"))
    assert not other._values
    assert board(lego, "independent", batch_size=1).adapt_inf_publisher(model, 0, consumer="selector") is None
    assert board(lego, batch_size=2).adapt_inf_publisher(model, 0, consumer="selector") is None
    model.config.model_type = "llava"
    assert other.adapt_inf_publisher(model, 0, consumer="selector") is None


def test_configuration_and_old_checkpoint_fail_explicitly(lego, tmp_path):
    from dataflex.train.lego import PlanContext
    from dataflex.train.lego.checkpoint import pipeline_state, restore_pipeline

    with pytest.raises(ValueError, match="score_sharing"):
        board(lego, "off")
    trainer = training_trainer(lego, tmp_path, "loss", "static", "dynamic", "multimodal_loss")
    stage = trainer.pipeline.index_stages[0]
    stage.score_max_age = 1
    ctx = PlanContext.from_trainer(trainer, scoreboard=trainer.scoreboard)
    with pytest.raises(ValueError, match="score_max_age"):
        stage.board_score_fn(ctx)
    stage.score_max_age = 0
    stage.component.score_params = {"max_samples": 2}
    with pytest.raises(ValueError, match="max_samples"):
        stage.board_score_fn(ctx)
    state = pipeline_state(trainer.pipeline)
    state["version"] = 1
    with pytest.raises(ValueError, match="before shared scoring"):
        restore_pipeline(trainer.pipeline, state, len(trainer.train_dataset))


@pytest.mark.parametrize("kind", ["qwen", "llava"])
def test_delete_visual_tokens_retains_responses(kind):
    from dataflex.utils.vision_ablation import strip_vision_tokens

    ids = [[0, 10, 3, 3, 11, 1, 2, 0], [0, 10, 3, 11, 1, 2, 0, 0]]
    if kind == "llava":
        ids = [[3 if token in (10, 11) else token for token in row] for row in ids]
    batch = {
        "input_ids": torch.tensor(ids),
        "attention_mask": torch.tensor([[1] * 7 + [0], [1] * 6 + [0, 0]]),
        "labels": torch.tensor([[-100] * 5 + [1, 2, -100], [-100] * 4 + [1, 2, -100, -100]]),
        "pixel_values": torch.ones(2, 2),
    }
    result = strip_vision_tokens(batch, (10, 11, 3) if kind == "qwen" else (None, None, 3))
    assert result["input_ids"].tolist() == [[0, 1, 2], [0, 1, 2]]
    assert result["labels"].tolist() == [[-100, 1, 2], [-100, 1, 2]]
    assert "pixel_values" not in result and "pixel_values" in batch


def test_shipped_lego_presets_use_exact_scoring(lego, tmp_path):
    from dataflex.train.lego import SchedulePipeline
    from dataflex.utils.load_component import load_component

    trainer = training_trainer(lego, tmp_path, "loss", "static", "dynamic", "multimodal_loss")
    path = str(Path(__file__).resolve().parents[1] / "src/dataflex/configs/components.yaml")
    runtime = dict(
        dataset=trainer.train_dataset, eval_dataset=trainer.eval_dataset, accelerator=trainer.accelerator,
        data_collator=trainer.data_collator, mixture_manager=trainer.domain_view, output_dir=str(tmp_path),
    )
    for name in ("hierarchical", "select_then_reorder", "multimodal_lego"):
        cfg = load_component("pipelines", path, name)
        pipeline = SchedulePipeline.from_config(
            cfg["stages"], path, runtime, scoreboard=trainer.scoreboard, cache_dir=str(tmp_path / name),
        )
        for stage in pipeline.index_stages:
            stage.validate_score_options()


def test_shared_export_retains_selector_cache_loss_alias(lego, tmp_path):
    cache = board(lego)
    cache.cache_dir = str(tmp_path)
    expected = cache.loss_scores(VisionModel(), [0, 1], 0, consumer="selector")
    payload = json.loads(Path(cache.save(0, [0, 1])).read_text())
    assert payload["metric"]["loss"] == payload["metric"]["loss_full"] == expected.tolist()
