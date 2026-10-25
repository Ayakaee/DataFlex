"""Keep scheduling metadata separate from a model's multimodal inputs."""

from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


SAMPLE_METADATA_KEYS = ("idx", "domain_id")
# Some multimodal collators attach this bookkeeping field themselves.
BATCH_METADATA_KEYS = (*SAMPLE_METADATA_KEYS, "__record_indices__")


def split_batch(batch):
    """Return shallow copies of model inputs and metadata, without a model-field allowlist.

    Labels and all vision/audio fields remain model inputs. Tensor objects are
    shared with the original batch; removing keys never mutates that batch.
    """
    return (
        {key: value for key, value in batch.items() if key not in BATCH_METADATA_KEYS},
        {key: value for key, value in batch.items() if key in BATCH_METADATA_KEYS},
    )


class DomainLabeledDataset(Dataset):
    """Attach identity in the final, preprocessed training dataset's index space.

    Wrap once after concatenating domains, before Subset/sampling/reordering.
    Repeated draws keep the same idx. These IDs identify processed rows (packed
    sequences when packing is enabled), not raw records or content hashes, and
    are only stable for the same processed dataset and domain order.
    """

    def __init__(self, dataset, domain_ids):
        self.dataset = dataset
        self.domain_ids = np.asarray(domain_ids, dtype=np.int64)
        if self.domain_ids.ndim != 1 or len(self.domain_ids) != len(dataset):
            raise ValueError("domain_ids must be a one-dimensional label array aligned with the dataset")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        item = self.dataset[idx]
        if not isinstance(item, Mapping):
            raise TypeError("LEGO requires dataset rows to be mappings")
        return {**item, "idx": idx, "domain_id": int(self.domain_ids[idx])}

    @property
    def column_names(self):
        names = getattr(self.dataset, "column_names", None)
        return list(dict.fromkeys([*names, *SAMPLE_METADATA_KEYS])) if names is not None else None


class MetadataCollator:
    """Let the original collator handle modalities, then restore row metadata.

    The wrapped collator must keep one output sequence per input row, in order.
    This is the contract of the preprocessed PT/SFT collators used by LEGO;
    collators which expand, pack or reorder rows here need a separate mapping.
    Eval/anchor batches without training IDs are valid. Partially labelled
    batches fail instead of silently misaligning IDs and model inputs.
    """

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, features):
        metadata = {}
        for key in SAMPLE_METADATA_KEYS:
            present = [key in feature for feature in features]
            if any(present):
                if not all(present):
                    raise ValueError(f"Inconsistent {key} metadata within a batch")
                metadata[key] = torch.tensor([feature[key] for feature in features], dtype=torch.long)
        rows = [
            {key: value for key, value in feature.items() if key not in SAMPLE_METADATA_KEYS} for feature in features
        ]
        batch = dict(self.collator(rows))
        input_ids = batch.get("input_ids")
        if metadata and input_ids is not None and len(input_ids) != len(features):
            raise ValueError("LEGO collator changed the number of rows; sample IDs cannot be aligned")
        batch.update(metadata)
        return batch

    def __getattr__(self, name):
        # __dict__.get avoids recursion during unpickling in DataLoader workers.
        collator = self.__dict__.get("collator")
        if collator is None:
            raise AttributeError(name)
        try:
            return getattr(collator, name)
        except AttributeError:
            # Hugging Face's RemoveColumnsCollator does not forward attributes.
            delegate = getattr(collator, "data_collator", None)
            if delegate is None:
                raise
            return getattr(delegate, name)
