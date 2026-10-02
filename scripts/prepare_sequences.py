"""Encode + pack the corpus into train/val sequence tensors (Step 10).

Usage:
    python scripts/prepare_sequences.py [--config configs/data-pack.yaml]

Reads corpus text -> BPE encode -> pack (seq_len+1 blocks, shift-by-one
labels) -> train/val split -> saves input_ids/labels .pt + manifest.json.
Trains the tokenizer first if its file is missing.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.data.packed import PackedLMDataset, pack_tokens, train_val_split
from myai.tokenizer.bpe import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data-pack.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tok_path = Path(cfg["tokenizer_path"])
    if not tok_path.exists():
        raise SystemExit(f"tokenizer missing: {tok_path} — run scripts/train_tokenizer.py first")
    tok = BPETokenizer.load(str(tok_path))
    text = Path(cfg["corpus_path"]).read_text(encoding="utf-8")
    ids = tok.encode(text)
    pairs = pack_tokens(ids, cfg["seq_len"])
    train_pairs, val_pairs = train_val_split(pairs, cfg.get("val_ratio", 0.05))
    train_ds, val_ds = PackedLMDataset(train_pairs), PackedLMDataset(val_pairs)

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": train_ds.inputs, "labels": train_ds.labels}, out / "train.pt")
    torch.save({"input_ids": val_ds.inputs, "labels": val_ds.labels}, out / "val.pt")
    manifest = {
        "version": cfg.get("version", "0.1.0"),
        "tokenizer": str(tok_path),
        "vocab_size": len(tok),
        "seq_len": cfg["seq_len"],
        "total_tokens": len(ids),
        "train_sequences": len(train_ds),
        "val_sequences": len(val_ds),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"packed {len(ids)} tokens -> {len(train_ds)} train + {len(val_ds)} val "
        f"x{cfg['seq_len']} -> {out}"
    )


if __name__ == "__main__":
    main()
