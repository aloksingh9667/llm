"""Encode + pack with document-level split and EOS boundaries (audit P0).

Usage:
    python scripts/prepare_sequences.py [--config configs/data-pack.yaml]

Flow (doc section 9 + audit sections 13/14/16):
    train.txt --split on EOS boundary--> docs --stable md5 hash--> train/val docs
        --> BPE encode (EOS marker becomes the special id) --> pack --> blocks
        --> legacy train.pt/val.pt + sharded train/shard-*.pt + index.json

Hash split is order-independent and leak-free: identical documents always
land in the same split.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.data.packed import pack_tokens
from myai.data.sharded import sha256_file, write_index, write_shards
from myai.tokenizer.bpe import BPETokenizer


def split_docs(text: str, eos_text: str | None) -> list[str]:
    """Split corpus text into documents on the EOS boundary (or blank lines)."""
    if eos_text:
        parts = [p.strip() for p in text.split(eos_text)]
    else:
        parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def hash_split(docs: list[str], val_ratio: float) -> tuple[list[str], list[str]]:
    """Stable md5 split: identical docs always share a split (no leakage)."""
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    cutoff = int(val_ratio * 1000)
    train, val = [], []
    for doc in docs:
        bucket = int(hashlib.md5(doc.encode("utf-8")).hexdigest(), 16) % 1000
        (val if bucket < cutoff else train).append(doc)
    if not train or not val:
        raise ValueError(f"degenerate split: {len(train)} train / {len(val)} val docs")
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data-pack.yaml")
    parser.add_argument("--rows-per-shard", type=int, default=4096)
    parser.add_argument("--corpus-manifest", default="data/tokenizer_corpus/manifest.json",
                        help="corpus manifest carrying eos_text (per-half manifests supported)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tok = BPETokenizer.load(cfg["tokenizer_path"])
    corpus_manifest = {}
    cm = Path(args.corpus_manifest)
    if cm.exists():
        corpus_manifest = json.loads(cm.read_text(encoding="utf-8"))
    eos_text = corpus_manifest.get("eos_text")

    text = Path(cfg["corpus_path"]).read_text(encoding="utf-8")
    docs = split_docs(text, eos_text)
    train_docs, val_docs = hash_split(docs, cfg.get("val_ratio", 0.05))

    out = Path(cfg.get("output_dir", "data/processed/seed-3b"))
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    seq_len = cfg["seq_len"]
    for split, sdocs in (("train", train_docs), ("val", val_docs)):
        ids: list[int] = []
        for doc in sdocs:
            ids.extend(tok.encode(doc + (eos_text or "")))
        pairs = pack_tokens(ids, seq_len)
        if not pairs:
            raise ValueError(f"no {split} sequences — corpus too small for seq_len={seq_len}?")
        blob = {
            "input_ids": torch.tensor([p[0] for p in pairs], dtype=torch.long),
            "labels": torch.tensor([p[1] for p in pairs], dtype=torch.long),
        }
        torch.save(blob, out / f"{split}.pt")
        entries[split] = write_shards(pairs, out, split, args.rows_per_shard)

    tok_hash = sha256_file(cfg["tokenizer_path"])
    manifest = {
        "version": cfg.get("version", "0.1.0"),
        "tokenizer": cfg["tokenizer_path"],
        "tokenizer_sha256": tok_hash,
        "vocab_size": len(tok),
        "seq_len": seq_len,
        "split_method": "stable-md5-doc-hash",
        "eos_text": eos_text,
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        "train_sequences": entries["train"]["rows"],
        "val_sequences": entries["val"]["rows"],
        "total_tokens": entries["train"]["tokens"] + entries["val"]["tokens"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_index(out, [entries["train"], entries["val"]], {k: v for k, v in manifest.items()
                                                          if k not in ("train_sequences", "val_sequences")})
    print(f"docs {len(train_docs)}+{len(val_docs)} -> "
          f"{entries['train']['rows']}+{entries['val']['rows']} x{seq_len} "
          f"(+ {len(entries['train']['shards'])}+{len(entries['val']['shards'])} shards) -> {out}")


if __name__ == "__main__":
    main()
