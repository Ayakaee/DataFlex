"""Build a reusable dynamic PreSel cache in LEGO's concatenated record order."""
import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from dataflex.utils.multimodal_selection import file_fingerprint
from .record_presel_selector import reference_indices


def row_key(row):
    image = row["images"][0] if "images" in row else row["image"]
    return image, row["conversations"]


def row_digest(row):
    value = json.dumps(row_key(row), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(value.encode()).digest()


def task_records(prepared, domains, split_zip, limit):
    images, original, records, sources = [], {}, [], {}
    for domain in domains:
        path = prepared / (domain + ".json")
        source = hashlib.sha256()
        for row in json.loads(path.read_text())[:limit]:
            key = row["original_index"]
            if key in original:
                raise ValueError(f"duplicate original index {key}")
            digest = row_digest(row)
            source.update(digest)
            original[key] = (len(images), digest)
            image, _ = row_key(row)
            images.append(image)
            records.append(row)
        sources[domain] = source.hexdigest()
    tasks = np.zeros(len(images), dtype=np.int32)
    with zipfile.ZipFile(split_zip) as archive:
        for task in range(1, 11):
            with archive.open(f"file{task}_665.json") as stream:
                for row in json.load(stream):
                    match = original.get(row["unique_idx"])
                    if match is None:
                        continue
                    idx, digest = match
                    if tasks[idx] or row_digest(row) != digest:
                        raise ValueError(f"author split differs from training row {idx}")
                    tasks[idx] = task
    if not tasks.all():
        raise ValueError("PreSel task splits did not cover every training record")
    names, tasks = np.unique(tasks, return_inverse=True)
    return records, images, names.tolist(), tasks, sources


def encode(paths, encoder, revision, device, batch_size):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(encoder, revision=revision)
    model = AutoModel.from_pretrained(encoder, revision=revision).to(device).eval()
    if model.config.model_type != "dinov2":
        raise ValueError("PreSel needs DINOv2 features")
    features = np.empty((len(paths), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as im:
                    batch.append(im.convert("RGB"))
            inputs = processor(images=batch, return_tensors="pt").to(device)
            features[start:start + len(batch)] = model(**inputs).last_hidden_state[:, 0].float().cpu().numpy()
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Invalid DINO embeddings")
    return features / norms


def visual_order(task, image_ids, features, seed):
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    cluster = np.zeros(len(task), dtype=np.int32)
    nc = np.zeros(len(task), dtype=np.float32)
    for t in np.unique(task):
        idx = np.flatnonzero(task == t)
        vectors = features[image_ids[idx]]
        labels = KMeans(n_clusters=max(1, len(idx)//100), random_state=seed).fit_predict(vectors)
        cluster[idx] = labels
        for c in np.unique(labels):
            members = np.flatnonzero(labels == c)
            if len(members) > 1:
                x = vectors[members]
                k = min(5, len(x)-1)
                nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(x)
                # X=None excludes each point itself, including in a cluster of duplicate images.
                distances, _ = nn.kneighbors()
                nc[idx[members]] = 1 - distances.mean(axis=1)
    return cluster, np.lexsort((np.arange(len(task)), -nc, cluster, task))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--domains", nargs="+", default=["coco", "gqa", "ocr_vqa", "textvqa", "vg"])
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--task-splits", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-domain-limit", type=int)
    p.add_argument("--encoder", default="facebook/dinov2-base")
    p.add_argument("--revision", default="f9e44c814b77203eaa57a6bdbbd535f21ede1415")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    records, images, names, task, sources = task_records(a.prepared, a.domains, a.task_splits, a.per_domain_limit)
    reference, _ = reference_indices([{"task": str(t)} for t in task], a.seed)
    # Tiny smoke pools can contain a task that rounded down to zero. Preserve
    # the exact total while reserving one reference per represented task.
    if len(reference) < len(names):
        raise ValueError("5% budget cannot cover all tasks; use a larger preparation pool")
    rng = np.random.default_rng(a.seed)
    for missing in sorted(set(task) - set(task[reference])):
        counts = np.bincount(task[reference], minlength=len(names))
        donor = int(np.argmax(counts))
        position = int(rng.choice(np.flatnonzero(task[reference] == donor)))
        reference[position] = int(rng.choice(np.flatnonzero(task == missing)))
    reference.sort()
    refs = [dict(idx=i, image=images[i], conversations=records[i]["conversations"]) for i in reference]
    del records
    paths, image_ids = np.unique(images, return_inverse=True)
    features = encode([a.images / str(path) for path in paths], a.encoder, a.revision, a.device, a.batch_size)
    cluster, order = visual_order(task, image_ids, features, a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.output.parent) as temp:
        out = Path(temp) / "ready"
        out.mkdir()
        np.savez_compressed(out / "ranks.npz", task=task, cluster=cluster, order=order)
        (out / "reference.json").write_text(json.dumps(refs, ensure_ascii=False))
        meta = dict(version=1, rows=len(task), tasks=names, domains=a.domains, source_rows=sources,
                    input_files={d + ".json": file_fingerprint(a.prepared / (d + ".json")) for d in a.domains},
                    per_domain_limit=a.per_domain_limit, seed=a.seed, reference_count=len(refs), unique_images=len(paths), encoder=a.encoder,
                    revision=a.revision, splits_sha256=file_fingerprint(a.task_splits),
                    files={f:file_fingerprint(out / f) for f in ("ranks.npz", "reference.json")})
        (out / "manifest.json").write_text(json.dumps(meta, indent=2))
        out.rename(a.output)
    print(f"Prepared {len(task)} records, {len(refs)} references, {len(names)} tasks in {a.output}")


if __name__ == "__main__":
    main()
