"""The delete-visual-tokens condition used by the DeltaLoss weighter.

This is intentionally different from ADAPT-INF's zero-pixel IG condition.
"""


def vision_token_ids(config):
    image = getattr(config, "image_token_id", None) or getattr(config, "image_token_index", None)
    return getattr(config, "vision_start_token_id", None), getattr(config, "vision_end_token_id", None), image


def strip_vision_tokens(batch, token_ids):
    """Keep the existing Qwen-VL/LLaVA ablation and response labels unchanged."""
    vs, ve, image = token_ids
    if vs is None and image is None:
        return batch
    ids, am, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    rows = []
    for i in range(ids.size(0)):
        valid = am[i] != 0
        row_ids, row_labels = ids[i][valid], labels[i][valid]
        if vs is not None:
            start, end = (row_ids == vs).long(), (row_ids == ve).long()
            inside = (start.cumsum(0) - end.cumsum(0) + end) > 0
        else:
            inside = row_ids == image
        rows.append((row_ids[~inside], row_labels[~inside]))
    length = max(len(row[0]) for row in rows)
    new_ids = ids.new_zeros((len(rows), length))
    new_labels = labels.new_full((len(rows), length), -100)
    new_am = am.new_zeros((len(rows), length))
    for i, (row_ids, row_labels) in enumerate(rows):
        n = len(row_ids)
        new_ids[i, :n], new_labels[i, :n], new_am[i, :n] = row_ids, row_labels, 1
    return {"input_ids": new_ids, "attention_mask": new_am, "labels": new_labels}
