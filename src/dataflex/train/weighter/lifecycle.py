"""Lifecycle helpers shared by standalone weighting and LEGO."""

from contextlib import contextmanager

import torch


def component_context(ctx, *, model=None, batch=None):
    if hasattr(ctx, "step_id"):
        return ctx
    # Lazy import: importing LEGO also registers the component families.
    from dataflex.train.lego.context import PlanContext

    return PlanContext.from_trainer(ctx, model=model, batch=batch)


@contextmanager
def scoring_mode(model):
    """Restore even mixed submodule training modes after scoring or an error."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


class IndexedWeightLifecycle:
    """Periodic full-dataset refresh, followed by lookup in processed-row space."""

    def prepare(self, ctx, model):
        ctx = component_context(ctx, model=model)
        if self.refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
        step = ctx.step_id
        if self.W is None or (step != self._last_refresh_step and step % self.refresh_interval == 0):
            if not len(self.dataset):
                raise ValueError("Cannot refresh weights on an empty training dataset")
            self._uses_sample_ids = "idx" in self.dataset[0]
            with scoring_mode(model), ctx.accelerator.autocast():
                self._refresh(ctx, model, step)
            self._last_refresh_step = step

    def _batch_weights(self, inputs, losses, key_fn):
        if self.W is None:
            raise RuntimeError("Weights are uninitialized; call prepare() before the training forward")
        if "idx" in inputs:
            indices = torch.as_tensor(inputs["idx"], dtype=torch.long).detach().cpu().view(-1)
            if len(indices) != losses.numel() or (indices < 0).any() or (indices >= len(self.W)).any():
                raise ValueError("Batch idx does not match the global weight table")
            return self.W[indices].to(device=losses.device, dtype=losses.dtype)
        # Compatibility for standalone WeightTrainer datasets without row IDs.
        if getattr(self, "_uses_sample_ids", False):
            raise ValueError("Batch idx is required for weights scored in global sample-ID space")
        return torch.tensor(
            [self._weight_map.get(k, 1.0) for k in key_fn(inputs)], dtype=losses.dtype, device=losses.device
        )

    def _get_length_order(self):
        if self._order is None:
            # Works on both HF datasets and DomainLabeledDataset/Subset, without
            # forwarding HF .map() through a wrapper and losing ID alignment.
            lengths = [len(self.dataset[i]["input_ids"]) for i in range(len(self.dataset))]
            self._order = torch.tensor(lengths).argsort(descending=True, stable=True).tolist()
        return self._order

    def state_dict(self):
        return {
            "weights": self.W,
            "last_refresh_step": self._last_refresh_step,
            "weight_map": self._weight_map,
            "uses_sample_ids": getattr(self, "_uses_sample_ids", False),
        }

    def load_state_dict(self, state):
        weights = state["weights"]
        if weights is not None:
            if weights.shape != (len(self.dataset),) or not torch.isfinite(weights).all():
                raise ValueError("Checkpoint weights do not match the training dataset")
            weights = weights.detach().cpu().clone()
        self.W = weights
        self._weight_map = dict(state["weight_map"])
        self._last_refresh_step = int(state["last_refresh_step"])
        self._uses_sample_ids = bool(state["uses_sample_ids"])
