"""Sharded packed dataset + manifest hashing (audit P0).

Monolithic tensors do not scale: sequences are written as row-shards
(`train/shard-00000.pt`, ...) plus an `index.json` recording counts,
hashes, tokenizer, and provenance. `load_sharded` reassembles the
legacy (input_ids, labels) tensors; `verify_index` recomputes hashes.
"""
import hashlib
import json
from pathlib import Path

import torch

from .packed import PackedLMDataset


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_shards(
    pairs: list[tuple[list[int], list[int]]],
    out_dir: str | Path,
    split: str,
    rows_per_shard: int,
    extra: dict | None = None,
) -> dict:
    """Write row-shards + per-split index entry. Returns the entry."""
    if rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be positive")
    d = Path(out_dir) / split
    d.mkdir(parents=True, exist_ok=True)
    shards = []
    for i in range(0, len(pairs), rows_per_shard):
        rows = pairs[i : i + rows_per_shard]
        blob = {
            "input_ids": torch.tensor([r[0] for r in rows], dtype=torch.long),
            "labels": torch.tensor([r[1] for r in rows], dtype=torch.long),
        }
        name = f"shard-{len(shards):05d}.pt"
        torch.save(blob, d / name)
        shards.append({
            "file": name,
            "rows": len(rows),
            "tokens": sum(len(r[0]) for r in rows),
            "sha256": sha256_file(d / name),
        })
    entry = {"split": split, "shards": shards,
             "rows": len(pairs), "tokens": sum(len(r[0]) for r in pairs)}
    if extra:
        entry.update(extra)
    return entry


def write_index(out_dir: str | Path, entries: list[dict], meta: dict) -> dict:
    """Write index.json combining split entries + run metadata."""
    index = {"version": "1.0.0", "splits": {e["split"]: e for e in entries}, **meta}
    p = Path(out_dir) / "index.json"
    p.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def load_sharded(out_dir: str | Path, split: str) -> tuple[PackedLMDataset, dict]:
    """Load a split's shards into one PackedLMDataset + its index entry."""
    d = Path(out_dir)
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    entry = index["splits"][split]
    pairs = []
    for sh in entry["shards"]:
        blob = torch.load(d / split / sh["file"], weights_only=True)
        pairs.extend(zip(blob["input_ids"].tolist(), blob["labels"].tolist()))
    return PackedLMDataset(pairs), entry


def verify_index(out_dir: str | Path) -> dict[str, bool]:
    """Recompute every shard hash. Returns {split/file: ok}."""
    d = Path(out_dir)
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    report = {}
    for split, entry in index["splits"].items():
        for sh in entry["shards"]:
            report[f"{split}/{sh['file']}"] = sha256_file(d / split / sh["file"]) == sh["sha256"]
    return report
