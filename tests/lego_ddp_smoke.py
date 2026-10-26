"""Run with torchrun --standalone --nproc_per_node=2 (CPU/Gloo, no model download)."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from test_lego_lifecycle import VisionModel, assert_odm_resume_equal, dataset, training_trainer, weighter
from transformers import TrainerCallback

from dataflex.train.lego import PlanContext
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
    for name in ("loss", "delta", "adapt", "odm"):
        weight_name = "loss" if name == "odm" else name
        weight = {"loss": "multimodal_loss", "delta": "multimodal_deltaloss", "adapt": "multimodal_adapt"}[weight_name]

        def trainer(folder):
            mixer = "odm" if name == "odm" else "static"
            return training_trainer(lego, root / name / folder, "loss", mixer, "dynamic", weight)

        full = trainer("full")
        # Odd scoring pool exercises Accelerate's replicated tail rows.
        component = weighter(weight_name, full, dataset(7))
        component.prepare(PlanContext.from_trainer(full), VisionModel())
        table = component.anchor_emb if name == "adapt" else component.W
        tables = [torch.empty_like(table) for _ in range(dist.get_world_size())]
        dist.all_gather(tables, table)
        for other in tables:
            torch.testing.assert_close(table, other)
        full.train()
        partial = trainer("partial")
        stop_step = 3 if name == "odm" else 1
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
        if name == "odm":
            assert_odm_resume_equal(full, partial, resumed)
        dist.barrier()
        if rank == 0:
            print(f"PASS: {name}, two-rank refresh and mid-interval resume", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
