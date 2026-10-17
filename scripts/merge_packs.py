"""Merge two packed half-datasets into one training dir (K6 legs data).

Both halves must share the SAME tokenizer (verified via manifest
tokenizer_sha256) and seq_len. Each half carries {train,val}.pt blobs of
packed [num_sequences, seq_len] rows plus a manifest.json. The merged
output re-exposes the same three-file contract train.py expects
(manifest.json + train.pt + val.pt) with stacked rows.
"""
import json
import sys
from pathlib import Path

import torch


def merge_packs(half_dirs: list[str], out_dir: str,
                name: str = "myai-fwab-32k",
                expect_sha: str | None = None) -> dict:
    """Concatenate train/val packed rows across halves. Returns the manifest.

    expect_sha pins the required tokenizer sha256 — with single-half inputs
    (or any silent mismatch) the merge is rejected unless it matches.
    """
    if not half_dirs:
        raise ValueError("merge_packs needs at least one half dir")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = json.loads((Path(half_dirs[0]) / "manifest.json").read_text(encoding="utf-8"))
    for h in half_dirs:
        m = json.loads((Path(h) / "manifest.json").read_text(encoding="utf-8"))
        for key in ("tokenizer_sha256", "vocab_size", "seq_len", "eos_text"):
            if base[key] != m[key]:
                raise ValueError(f"halves disagree on {key}: {base[key]!r} vs {m[key]!r}")
    if expect_sha is not None and base["tokenizer_sha256"] != expect_sha:
        raise ValueError(f"tokenizer sha {base['tokenizer_sha256'][:12]}... does not "
                         f"match expected {expect_sha[:12]}...")

    totals = {"train_sequences": 0, "val_sequences": 0, "train_docs": 0, "val_docs": 0}
    for split in ("train", "val"):
        chunks = [torch.load(Path(h) / f"{split}.pt", weights_only=True) for h in half_dirs]
        merged = {
            "input_ids": torch.cat([c["input_ids"] for c in chunks], dim=0),
            "labels": torch.cat([c["labels"] for c in chunks], dim=0),
        }
        torch.save(merged, out / f"{split}.pt")
        tot = merged["input_ids"].shape[0]
        key = "train_sequences" if split == "train" else "val_sequences"
        totals[key] = tot
        totals["train_docs" if split == "train" else "val_docs"] = sum(
            json.loads((Path(h) / "manifest.json").read_text(encoding="utf-8"))[
                "train_docs" if split == "train" else "val_docs"] for h in half_dirs)

    manifest = dict(base)
    manifest.update({
        "name": name,
        "halves": [str(Path(h).name) for h in half_dirs],
        **totals,
        "total_tokens": totals["train_sequences"] * base["seq_len"],  # packed rows
    })
    for h in half_dirs:
        hm = json.loads((Path(h) / "manifest.json").read_text(encoding="utf-8"))
        manifest.setdefault("total_tokens_real", 0)
        manifest["total_tokens_real"] += hm.get("total_tokens", 0)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("halves", nargs="+", help="packed half dirs (e.g. data/processed/fwA-32k)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", default="myai-fwab-32k")
    parser.add_argument("--expect-sha", default=None,
                        help="reject the merge unless every half's tokenizer sha matches")
    args = parser.parse_args()
    m = merge_packs(args.halves, args.out_dir, args.name, args.expect_sha)
    print(f"merged -> {args.out_dir}: train={m['train_sequences']} "
          f"val={m['val_sequences']} rows ({m['total_tokens_real'] / 1e6:.1f}M tokens)")


if __name__ == "__main__":
    sys.exit(main())
