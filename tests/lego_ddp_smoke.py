"""Run with torchrun --standalone --nproc_per_node=2 (CPU/Gloo, no model download)."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from test_lego_lifecycle import VisionModel, assert_odm_resume_equal, dataset, training_trainer, weighter
from transformers import TrainerCallback

from dataflex.train.lego import PlanContext, ScoreBoard
from dataflex.train.selector.adapt_inf_selector import AdaptInfSelector
from dataflex.train.trainer.lego_trainer import LegoTrainer


class StopAtStep(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_training_stop = True
            control.should_save = True
        return control


def main(root):
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    lego = SimpleNamespace(Trainer=LegoTrainer)
    runs = {}
    scenarios = [
        ("loss", "loss", "static", "loss", "precomputed", "shared"),
        ("delta", "delta", "static", "loss", "precomputed", "shared"),
        ("adapt", "adapt", "static", "loss", "precomputed", "shared"),
        ("odm", "loss", "odm", "loss", "precomputed", "shared"),
        ("sharing", "delta", "odm", "delta_loss", "model_loss", "shared"),
        ("independent", "delta", "odm", "delta_loss", "model_loss", "independent"),
    ]
    for name, weight_name, mixer, selector, score_source, mode in scenarios:
        weight = {"loss": "multimodal_loss", "delta": "multimodal_deltaloss", "adapt": "multimodal_adapt"}[weight_name]

        def trainer(folder):
            return training_trainer(
                lego, root / name / folder, selector, mixer, "dynamic", weight,
                score_sharing=mode, score_source=score_source, score_batch_size=3,
            )

        full = trainer("full")
        # Odd scoring pool exercises Accelerate's replicated tail rows.
        component = weighter(weight_name, full, dataset(7))
        component.prepare(PlanContext.from_trainer(full), VisionModel())
        table = component.anchor_emb if weight_name == "adapt" else component.W
        tables = [torch.empty_like(table) for _ in range(dist.get_world_size())]
        dist.all_gather(tables, table)
        for other in tables:
            torch.testing.assert_close(table, other)
        full.train()
        partial = trainer("partial")
        stop_step = 3 if mixer == "odm" else 1
        partial.add_callback(StopAtStep(stop_step))
        partial.train()
        dist.barrier()
        checkpoint = root / name / f"partial/checkpoint-{stop_step}"
        assert (checkpoint / f"dataflex_lego_rank_{rank}.pt").exists()
        resumed = trainer("resumed")
        # Accelerate may report cpu:0/cpu:1; PyTorch 2.6's storage loader only
        # recognizes "cpu". Normalize this test-only optimizer-load argument.
        original_load = torch.load

        def cpu_load(*args, **kwargs):
            location = kwargs.get("map_location")
            if isinstance(location, (str, torch.device)) and torch.device(location).type == "cpu":
                kwargs["map_location"] = "cpu"
            return original_load(*args, **kwargs)

        with patch("torch.load", side_effect=cpu_load):
            resumed.train(resume_from_checkpoint=str(checkpoint))
        actual = partial.trace + resumed.trace
        assert [(s, ids, refresh) for s, ids, _, refresh in actual] == [
            (s, ids, refresh) for s, ids, _, refresh in full.trace
        ]
        torch.testing.assert_close(torch.tensor([x[2] for x in actual]), torch.tensor([x[2] for x in full.trace]))
        torch.testing.assert_close(resumed.model.scale, full.model.scale)
        if mixer == "odm":
            assert_odm_resume_equal(full, partial, resumed)
        runs[name] = full
        dist.barrier()
        if rank == 0:
            print(f"PASS: {name}, two-rank refresh and mid-interval resume", flush=True)
    shared, independent = runs["sharing"], runs["independent"]
    assert [(s, ids, r) for s, ids, _, r in shared.trace] == [(s, ids, r) for s, ids, _, r in independent.trace]
    torch.testing.assert_close(torch.tensor([x[2] for x in shared.trace]), torch.tensor([x[2] for x in independent.trace]))
    torch.testing.assert_close(shared.model.scale, independent.model.scale)
    assert sum(n for n, _, grad in shared.model.calls if not grad) < sum(
        n for n, _, grad in independent.model.calls if not grad
    )
    if rank == 0:
        print("PASS: shared/independent two-rank training parity and fewer scored samples", flush=True)
    publication = root / "publication"
    positions, losses = [6, 0, 3], [0.4, 1.2, 2.3]
    if rank == 0:
        publication.mkdir(parents=True, exist_ok=True)
        np.save(publication / "dataset_positions.npy", np.asarray(positions))
        scores = np.zeros((3, 4))
        scores[:, 0] = losses
        np.save(publication / "scores.npy", scores)
    dist.barrier()
    cache = ScoreBoard(
        dataset=dataset(7), accelerator=shared.accelerator, data_collator=shared.data_collator,
        batch_size=1, num_workers=0,
    )
    model = VisionModel()
    selector = object.__new__(AdaptInfSelector)
    selector._publish_loss_scores(publication, positions, cache.adapt_inf_publisher(model, 0, consumer="selector"))
    np.testing.assert_allclose(cache.loss_scores(model, positions, 0, consumer="weighter"), losses)
    assert not model.calls
    if rank == 0:
        print("PASS: ADAPT-INF publication reaches all ranks without another forward", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
