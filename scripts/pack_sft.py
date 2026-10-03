"""Pack instruction pairs into masked SFT blocks (Phase 1).

Usage:
    python scripts/pack_sft.py --pairs data/sft/seed.jsonl --tokenizer data/tokenizer_corpus/tokenizer-3b.json \\
        --seq-len 64 --out-dir data/processed/sft-seed

Format: "Instruction: {instruction} {input}\\nResponse: {output}" with the
whole prompt region masked (-100); loss trains on response + EOS only.
Writes train.pt/val.pt (doc-hash split) + shards + manifest.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.data.sharded import write_index, write_shards
from myai.tokenizer.bpe import BPETokenizer
from myai.training.sft import IGNORE, build_sft_pair, pack_sft_pairs
from scripts.prepare_sequences import hash_split

EOS = "<|endoftext|>"


def format_pair(p: dict) -> tuple[str, str]:
    prompt = f"Instruction: {p['instruction']}" + (f"\nInput: {p['input']}" if p.get("input") else "")
    return prompt, p["output"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/sft/seed.jsonl")
    parser.add_argument("--tokenizer", default="data/tokenizer_corpus/tokenizer-3b.json")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--out-dir", default="data/processed/sft-seed")
    args = parser.parse_args()

    pairs = [json.loads(l) for l in Path(args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    tok = BPETokenizer.load(args.tokenizer)
    eos_ids = tok.encode(EOS)
    assert len(eos_ids) == 1, "EOS must be a single special id"
    eos_id = eos_ids[0]
    docs = [json.dumps(p, sort_keys=True) for p in pairs]
    train_docs, val_docs = hash_split(docs, args.val_ratio)
    by_doc = {json.dumps(p, sort_keys=True): p for p in pairs}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    dropped_total = 0
    for split, sdocs in (("train", train_docs), ("val", val_docs)):
        sft_pairs = []
        for d in sdocs:
            prompt, response = format_pair(by_doc[d])
            p_ids = tok.encode(prompt)
            r_ids = tok.encode(response)
            sft_pairs.append(build_sft_pair(p_ids, r_ids, eos_id))
        blocks, dropped = pack_sft_pairs(sft_pairs, args.seq_len)
        dropped_total += dropped
        if not blocks:
            raise ValueError(f"no {split} blocks — seq_len too large for data?")
        blob_in = torch.tensor([b[0] for b in blocks], dtype=torch.long)
        blob_lab = torch.tensor([b[1] for b in blocks], dtype=torch.long)
        torch.save({"input_ids": blob_in, "labels": blob_lab}, out / f"{split}.pt")
        entries[split] = write_shards(blocks, out, split, rows_per_shard=4096)
    manifest = {"version": "0.1.0", "pairs_total": len(pairs),
                "train_pairs": len(train_docs), "val_pairs": len(val_docs),
                "train_blocks": entries["train"]["rows"], "val_blocks": entries["val"]["rows"],
                "dropped_all_masked": dropped_total,
                "seq_len": args.seq_len, "vocab_size": len(tok),
                "mask": "prompt -100, response+EOS trained"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_index(out, [entries["train"], entries["val"]], manifest)
    print(f"sft {len(train_docs)}+{len(val_docs)} pairs -> "
          f"{entries['train']['rows']}+{entries['val']['rows']} x{args.seq_len} -> {out}")


if __name__ == "__main__":
    main()
