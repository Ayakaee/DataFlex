"""CPU contract tests; no pretrained model, image download or GPU is needed."""

import pickle

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, Subset

from dataflex.train.data.metadata import DomainLabeledDataset, MetadataCollator, split_batch


class ImageCollator:
    tokenizer = "test-tokenizer"
    processor = "test-processor"
    template = "test-template"

    def __call__(self, rows):
        # Strictly reject leaked metadata, and exercise a collator that pops keys.
        assert all("idx" not in row and "domain_id" not in row for row in rows)
        images = [row.pop("images")[0] for row in rows]
        return {
            "input_ids": torch.tensor([row["input_ids"] for row in rows]),
            "labels": torch.tensor([row["labels"] for row in rows]),
            # Two patches per image: leading dimension is NOT the batch size.
            "pixel_values": torch.tensor(images, dtype=torch.float).repeat_interleave(2).unsqueeze(-1),
            "image_grid_thw": torch.tensor([[1, 1, 2]] * len(rows)),
        }


def image_dataset(domains=(0, 1)):
    return DomainLabeledDataset(
        Dataset.from_list(
            [
                {"input_ids": [0, 1, 2], "labels": [-100, 1, 2], "images": [1.0]},
                {"input_ids": [0, 1, 2], "labels": [-100, 1, 2], "images": [5.0]},
            ]
        ),
        domains,
    )


@pytest.mark.parametrize("domains", [(0, 0), (0, 1)])
def test_ids_survive_nested_subsets_and_repeated_draws(domains):
    dataset = image_dataset(domains)
    subset = Subset(Subset(dataset, [1, 0]), [0, 1, 0])
    batch = next(iter(DataLoader(subset, batch_size=3, collate_fn=MetadataCollator(ImageCollator()))))
    assert batch["idx"].tolist() == [1, 0, 1]
    assert batch["domain_id"].tolist() == [domains[1], domains[0], domains[1]]
    assert batch["pixel_values"].flatten().tolist() == [5, 5, 1, 1, 5, 5]
    assert dataset[-1]["idx"] == 1
    assert dataset[0]["input_ids"] == dataset[1]["input_ids"]
    assert set(dataset.column_names) >= {"idx", "domain_id", "images"}


def test_collator_preserves_features_and_supports_unlabelled_anchors():
    dataset = image_dataset()
    rows = [dataset[1], dataset[0]]
    collator = MetadataCollator(ImageCollator())
    collator(rows)
    assert [row["images"] for row in rows] == [[5.0], [1.0]]
    batch = collator([dataset.dataset[0]])
    assert "idx" not in batch and "domain_id" not in batch
    restored = pickle.loads(pickle.dumps(collator))
    assert restored.tokenizer == ImageCollator.tokenizer
    assert restored.processor == ImageCollator.processor
    assert restored.template == ImageCollator.template
    assert restored(rows)["idx"].tolist() == [1, 0]


@pytest.mark.parametrize("key", ["idx", "domain_id"])
def test_partial_metadata_fails_instead_of_misaligning(key):
    rows = [image_dataset()[0], image_dataset()[1]]
    rows[1].pop(key)
    with pytest.raises(ValueError, match=f"Inconsistent {key}"):
        MetadataCollator(ImageCollator())(rows)


def test_collator_rejects_changed_row_count():
    with pytest.raises(ValueError, match="number of rows"):
        MetadataCollator(lambda rows: {"input_ids": torch.zeros(1, 3)})(list(image_dataset()))


def test_split_preserves_all_model_fields_without_copying_tensors():
    batch = MetadataCollator(ImageCollator())([image_dataset()[0]])
    batch["future_modality_field"] = object()
    batch["__record_indices__"] = [42]
    model_inputs, metadata = split_batch(batch)
    assert set(metadata) == {"idx", "domain_id", "__record_indices__"}
    assert set(model_inputs) == set(batch) - set(metadata)
    assert all(model_inputs[key] is batch[key] for key in model_inputs)
    model_inputs.pop("labels")
    assert "labels" in batch and "idx" in batch


def test_dataset_rejects_misaligned_domain_ids():
    with pytest.raises(ValueError, match="aligned"):
        DomainLabeledDataset([{}, {}], [0])
