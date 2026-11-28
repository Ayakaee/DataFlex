"""Online task budgets from paired answer losses, with fixed DINO cluster ranks."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from dataflex.core.registry import register_selector
from dataflex.train.weighter.base_weighter import scoring_mode
from dataflex.utils.multimodal_selection import file_fingerprint
from .base_selector import Selector


def allocate(weights, capacity, budget):
    """Proportional integer quotas, redistributing overflow from saturated groups."""
    capacity = np.asarray(capacity, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if budget < 0 or budget > capacity.sum() or (capacity < 0).any():
        raise ValueError("PreSel budget exceeds the available pool")
    quota = np.zeros_like(capacity)
    remaining = int(budget)
    while remaining:
        free = np.flatnonzero(quota < capacity)
        w = weights[free]
        ideal = remaining * w / w.sum()
        room = capacity[free] - quota[free]
        saturated = ideal >= room
        if saturated.any():
            quota[free[saturated]] += room[saturated]
            remaining -= int(room[saturated].sum())
            continue
        take = ideal.astype(np.int64)
        quota[free] += take
        remaining -= int(take.sum())
        order = np.argsort(-(ideal - take), kind="stable")
        quota[free[order[:remaining]]] += 1
        break
    return quota


def task_scores(ratios, tasks, count):
    means = np.bincount(tasks, weights=ratios, minlength=count) / np.bincount(tasks, minlength=count)
    logits = -means * np.sqrt(count)
    weights = np.exp(np.maximum(logits - logits.max(), -700))
    return means, weights / weights.sum()


@register_selector("dynamic_presel")
class DynamicPreSel(Selector):
    def __init__(self, dataset, accelerator, data_collator, cache_dir, artifacts, diagnostic_step=22,
                 cutoff_len=2048, seed=42, verify_presel_rows=None):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.seed, self.diagnostic_step, self.cutoff_len = seed, diagnostic_step, cutoff_len
        root = Path(artifacts)
        manifest_path = root / "manifest.json"
        self.identity = file_fingerprint(manifest_path)
        meta = json.loads(manifest_path.read_text())
        if meta["version"] != 1 or meta["rows"] != len(dataset):
            raise ValueError("PreSel cache does not match the processed dataset size")
        if verify_presel_rows is None:
            raise ValueError("Dynamic PreSel requires row-order verification from the LEGO data loader")
        verify_presel_rows(dataset, meta)
        for name in ("ranks.npz", "reference.json"):
            if file_fingerprint(root / name) != meta["files"][name]:
                raise ValueError(f"PreSel artifact changed: {name}")
        with np.load(root / "ranks.npz") as arrays:
            self.task = arrays["task"]
            self.cluster = arrays["cluster"]
            self.order = arrays["order"]
        self.reference = json.loads((root / "reference.json").read_text())
        self.task_names = meta["tasks"]
        self.seen = torch.zeros(len(dataset), dtype=torch.uint8)
        self.interval = torch.zeros_like(self.seen)
        self.previous_interval = torch.zeros_like(self.seen)
        self.previous_candidates = []
        self.logs = Path(cache_dir)
        self.logs.mkdir(exist_ok=True, parents=True)

    def observe_trained(self, ids):
        self.interval[torch.as_tensor(ids).detach().cpu().long().reshape(-1)] = 1

    def state_dict(self):
        return {"artifacts": self.identity, "seen": self.seen, "interval": self.interval,
                "previous_interval": self.previous_interval, "previous_candidates": self.previous_candidates}

    def load_state_dict(self, state):
        if state["artifacts"] != self.identity or state["seen"].shape != self.seen.shape:
            raise ValueError("PreSel checkpoint and prepared artifacts differ")
        for key in ("seen", "interval", "previous_interval"):
            getattr(self, key).copy_(state[key])
        self.previous_candidates = state["previous_candidates"]

    def close_training_interval(self):
        current = self.interval.to(self.accelerator.device)
        if dist.is_initialized():
            dist.all_reduce(current, op=dist.ReduceOp.MAX)
        current = current.cpu()
        self.seen.bitwise_or_(current)
        report = {"trained_in_interval": int(current.sum()),
                  "trained_in_previous_and_current": int((self.previous_interval & current).sum()),
                  "trained_unique": int(self.seen.sum())}
        self.previous_interval.copy_(current)
        self.interval.zero_()
        return report

    def finish_training(self):
        report = self.close_training_interval()
        if self.accelerator.is_main_process:
            (self.logs / "training_coverage.json").write_text(json.dumps(report, indent=2))

    def _score(self, model):
        from dataflex.offline_selector.offline_presel_selector import paired_inputs
        from dataflex.offline_selector.record_presel_selector import without_question_record

        device = self.accelerator.device
        distributed = dist.is_initialized()
        rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
        losses = torch.zeros((len(self.reference), 2), device=device, dtype=torch.float64)
        model = self.accelerator.unwrap_model(model)
        processor = self.data_collator.processor
        with scoring_mode(model), torch.inference_mode(), self.accelerator.autocast():
            for j in range(rank, len(self.reference), world):
                row = self.reference[j]
                image = self.dataset[row["idx"]]["images"][0]
                if not (Path(image) == Path(row["image"]) or Path(image).as_posix().endswith("/" + row["image"])):
                    raise ValueError(f"PreSel reference image mismatch at row {row['idx']}")
                for column, record in enumerate((row, without_question_record(row))):
                    batch = paired_inputs(processor, record, image, cutoff_len=self.cutoff_len)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    labels = batch.pop("labels")[:, 1:]
                    logits = model(**batch, use_cache=False).logits[:, :-1]
                    active = labels.ne(-100)
                    losses[j, column] = torch.nn.functional.cross_entropy(logits[active].float(), labels[active])
        if distributed:
            dist.all_reduce(losses)
        losses = losses.cpu().numpy()
        if not np.isfinite(losses).all() or np.any(losses[:, 1] <= 0):
            raise ValueError("Invalid paired PreSel answer losses")
        return losses[:, 0] / losses[:, 1]

    def _measure(self, model, step_id, budget):
        coverage = self.close_training_interval()
        ratios = self._score(model)
        ref_ids = np.array([r["idx"] for r in self.reference])
        tasks = self.task[ref_ids]
        means, weights = task_scores(ratios, tasks, len(self.task_names))
        pool_mask = np.zeros(len(self.dataset), dtype=bool)
        pool_mask[self.candidate_positions()] = True
        ordered = self.order[pool_mask[self.order]]
        capacity = np.bincount(self.task[ordered], minlength=len(weights))
        quotas = allocate(weights, capacity, budget)
        report = dict(step=step_id, budget=budget, tasks=self.task_names, mean_irs=means.tolist(),
                      quotas=quotas.tolist(), **coverage)
        if self.accelerator.is_main_process:
            np.savez_compressed(self.logs / f"irs_{step_id}.npz", idx=ref_ids, irs=ratios)
        return ratios, tasks, capacity, ordered, quotas, report

    def _log(self, report):
        if self.accelerator.is_main_process:
            dest = self.logs / f"refresh_{report['step']}.json"
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, indent=2))
            tmp.replace(dest)

    def diagnose(self, model, step_id, budget):
        ratios, tasks, capacity, _, _, report = self._measure(model, step_id, budget)
        rng = np.random.default_rng(self.seed)
        halves = np.zeros(len(tasks), dtype=bool)
        for t in range(len(capacity)):
            ids = rng.permutation(np.flatnonzero(tasks == t))
            halves[ids[::2]] = True
        report.update(diagnostic_only=True, split_quotas=None)
        if all(len(np.unique(tasks[mask])) == len(capacity) for mask in (halves, ~halves)):
            report["split_quotas"] = [
                allocate(task_scores(ratios[m], tasks[m], len(capacity))[1], capacity, budget).tolist()
                for m in (halves, ~halves)
            ]
        self._log(report)

    def select(self, model, step_id, num_samples, **kwargs):
        _, _, _, ordered, quotas, report = self._measure(model, step_id, num_samples)
        seen = self.seen.numpy().astype(bool)
        selected = []
        # Cache order groups by task, cluster, then descending visual typicality.
        for group in np.split(ordered, np.flatnonzero(np.diff(self.task[ordered])) + 1):
            if not len(group):
                continue
            clusters = np.split(group, np.flatnonzero(np.diff(self.cluster[group])) + 1)
            sizes = np.array([len(c) for c in clusters])
            counts = allocate(sizes, sizes, quotas[self.task[group[0]]])
            for ids, n in zip(clusters, counts):
                fresh = ids[~seen[ids]][:n]
                selected.extend(fresh.tolist())
                selected.extend(ids[seen[ids]][:n-len(fresh)].tolist())
        if len(selected) != num_samples or len(set(selected)) != num_samples:
            raise RuntimeError("PreSel selected an incorrect budget")
        report.update(diagnostic_only=False, previously_trained=int(seen[selected].sum()))
        report["candidate_overlap_previous"] = len(set(selected).intersection(self.previous_candidates))
        self.previous_candidates = selected
        self._log(report)
        return selected
