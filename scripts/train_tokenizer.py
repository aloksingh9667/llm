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
    parser.add_argument("--fast", action="store_true",
                        help="indexed trainer: identical merges, much faster on large corpora")
    parser.add_argument("--sample-chars", type=int, default=None,
                        help="train BPE on the first N corpus chars only (statistics "
                             "saturate early; bounds RAM/time on 100M+ corpora; "
                             "recorded in the tokenizer sidecar)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    corpus = args.corpus or cfg["corpus_path"]
    out = args.out or cfg["output_path"]
    vocab_size = args.vocab_size or cfg["vocab_size"]
    specials = cfg.get("special_tokens")

    tok = BPETokenizer()
    t0 = time.time()
    train_corpus = corpus
    if args.sample_chars:
        text = Path(corpus).read_text(encoding="utf-8")
        sample = text[: args.sample_chars]
        # Cut at a doc boundary so no half-document poisons statistics.
        cut = sample.rfind("\n\n")
        sample = sample[:cut] if cut > 0 else sample
        sample_path = str(Path(out).parent / "bpe-train-sample.txt")
        Path(sample_path).write_text(sample, encoding="utf-8")
        train_corpus = sample_path
        print(f"sampling {len(sample)} chars for BPE (of {len(text)})")
    if args.fast:
        from myai.tokenizer.trainer import train_indexed

        vocab, merges = train_indexed(train_corpus, vocab_size=vocab_size,
                                      special_tokens=specials,
                                      min_frequency=cfg.get("min_frequency", 1))
        tok = BPETokenizer(vocab=vocab, merges=merges,
                           special_tokens=specials or ["<|endoftext|>"])
    else:
        tok.train(
            train_corpus,
            vocab_size=vocab_size,
            special_tokens=specials,
            min_frequency=cfg.get("min_frequency", 1),
        )
    dt = time.time() - t0
    tok.save(out)
    if args.sample_chars:
        import json

        sidecar = {"bpe_sample_chars": len(open(train_corpus, encoding="utf-8").read()),
                   "bpe_sample_of_chars": len(open(corpus, encoding="utf-8").read()),
                   "bpe_trainer": "indexed" if args.fast else "reference"}
        Path(str(out) + ".sample.json").write_text(json.dumps(sidecar, indent=2) + "\n",
                                                   encoding="utf-8")

    ids = tok.encode(SAMPLE)
    back = tok.decode(ids)
    raw_bytes = len(SAMPLE.encode("utf-8"))
    print(f"trained in {dt:.2f}s: vocab={len(tok)} merges={len(tok.merges)} -> {out}")
    print(f"sample_ids={ids[:20]}{'...' if len(ids) > 20 else ''} ({len(ids)} tokens)")
    print(f"round_trip_ok={back == SAMPLE} tokens_per_byte={len(ids) / raw_bytes:.3f}")


if __name__ == "__main__":
    main()
