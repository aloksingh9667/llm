"""Train the from-scratch BPE tokenizer (Step 3b).

Usage:
    python scripts/train_tokenizer.py [--config configs/tokenizer-train.yaml]
    python scripts/train_tokenizer.py --vocab-size 500

Prints merge count, vocab size, and a round-trip + compression sample.
"""
import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.tokenizer.bpe import BPETokenizer

SAMPLE = "Language modeling predicts the next token. def causal_loss(logits, labels):"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer-train.yaml")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    corpus = args.corpus or cfg["corpus_path"]
    out = args.out or cfg["output_path"]
    vocab_size = args.vocab_size or cfg["vocab_size"]

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(
        corpus,
        vocab_size=vocab_size,
        special_tokens=cfg.get("special_tokens"),
        min_frequency=cfg.get("min_frequency", 1),
    )
    dt = time.time() - t0
    tok.save(out)

    ids = tok.encode(SAMPLE)
    back = tok.decode(ids)
    raw_bytes = len(SAMPLE.encode("utf-8"))
    print(f"trained in {dt:.2f}s: vocab={len(tok)} merges={len(tok.merges)} -> {out}")
    print(f"sample_ids={ids[:20]}{'...' if len(ids) > 20 else ''} ({len(ids)} tokens)")
    print(f"round_trip_ok={back == SAMPLE} tokens_per_byte={len(ids) / raw_bytes:.3f}")


if __name__ == "__main__":
    main()
