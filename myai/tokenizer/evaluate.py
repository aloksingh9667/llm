"""Tokenizer evaluation metrics (Step 3c).

Per-split numbers that decide the freeze (doc section 8):
  tokens_per_byte, chars_per_token (compression), fertility
  (tokens per whitespace-word), encode throughput, round-trip rate.
Stdlib only.
"""
import time

from .bpe import BPETokenizer


def evaluate_split(tok: BPETokenizer, docs: list[str]) -> dict:
    """Score one split (e.g. prose or code). Deterministic."""
    n_tokens = 0
    n_bytes = 0
    n_chars = 0
    n_words = 0
    round_trip_ok = 0
    t0 = time.perf_counter()
    for doc in docs:
        ids = tok.encode(doc)
        back = tok.decode(ids)
        n_tokens += len(ids)
        n_bytes += len(doc.encode("utf-8"))
        n_chars += len(doc)
        n_words += len(doc.split())
        round_trip_ok += back == doc
    dt = max(time.perf_counter() - t0, 1e-9)
    return {
        "n_docs": len(docs),
        "n_chars": n_chars,
        "n_bytes": n_bytes,
        "n_tokens": n_tokens,
        "tokens_per_byte": n_tokens / n_bytes if n_bytes else 0.0,
        "chars_per_token": n_chars / n_tokens if n_tokens else 0.0,
        "fertility": n_tokens / n_words if n_words else 0.0,
        "tokens_per_sec": n_tokens / dt,
        "round_trip_rate": round_trip_ok / len(docs) if docs else 0.0,
    }


def evaluate_tokenizer(tok: BPETokenizer, splits: dict[str, list[str]]) -> dict:
    """Score every split plus a pooled overall row."""
    report = {
        "vocab_size": len(tok),
        "n_merges": len(tok.merges),
        "splits": {name: evaluate_split(tok, docs) for name, docs in splits.items()},
    }
    all_docs = [d for docs in splits.values() for d in docs]
    report["overall"] = evaluate_split(tok, all_docs)
    return report
