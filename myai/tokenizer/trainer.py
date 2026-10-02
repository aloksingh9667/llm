"""Scalable BPE trainer: heap + occurrence index (audit P1).

The reference `BPETokenizer.train` rescans the whole corpus per merge:
O(merges x corpus). This trainer produces IDENTICAL merges by keeping:

  - word splits (mutable symbol lists per unique word)
  - pair -> count, pair -> {word ids containing it}
  - max-heap on (-count, pair) with lazy stale-entry refresh

Each merge touches only words containing the pair. Tie-breaking matches
the reference exactly (highest count, then larger byte pair), so
`test_trainer_matches_reference` pins equivalence on the seed corpus.
"""
import heapq
from collections import Counter

from .bpe import DEFAULT_SPECIAL_TOKENS, PRETOKEN_RE


class _MaxPair(tuple):
    """Tuple comparing in REVERSE: lets a min-heap pop the reference's
    max-(count, pair) winner (highest count, then larger byte pair)."""

    def __lt__(self, other) -> bool:
        return tuple.__gt__(self, other)


def train_indexed(corpus_path: str, vocab_size: int,
                  special_tokens: list[str] | None = None,
                  min_frequency: int = 1) -> tuple[list[bytes], list[tuple[bytes, bytes]]]:
    """Return (vocab, merges) identical to the reference trainer."""
    from pathlib import Path

    specials = list(special_tokens) if special_tokens is not None else list(DEFAULT_SPECIAL_TOKENS)
    text = Path(corpus_path).read_text(encoding="utf-8")
    freq: Counter[tuple[bytes, ...]] = Counter()
    for match in PRETOKEN_RE.finditer(text):
        piece = match.group(0).encode("utf-8")
        if piece:
            freq[tuple(bytes([b]) for b in piece)] += 1
    if min_frequency > 1:
        freq = Counter({w: c for w, c in freq.items() if c >= min_frequency})
    if not freq:
        raise ValueError("no words to train on — corpus empty or min_frequency filtered everything out")

    words = list(freq.keys())
    counts = [freq[w] for w in words]
    splits: list[list[bytes]] = [list(w) for w in words]

    pair_count: Counter[tuple[bytes, bytes]] = Counter()
    pair_words: dict[tuple[bytes, bytes], set[int]] = {}
    for i, symbols in enumerate(splits):
        for a, b in zip(symbols, symbols[1:]):
            pair_count[(a, b)] += counts[i]
            pair_words.setdefault((a, b), set()).add(i)

    heap: list[tuple[int, _MaxPair, tuple[bytes, bytes]]] = [
        (-c, _MaxPair(p), p) for p, c in pair_count.items()
    ]
    heapq.heapify(heap)

    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    for tok in specials:
        b = tok.encode("utf-8")
        if b not in vocab:
            vocab.append(b)
    merges: list[tuple[bytes, bytes]] = []
    num_merges = max(0, vocab_size - len(vocab))

    for _ in range(num_merges):
        while heap:
            negc, _, pair = heapq.heappop(heap)
            if pair_count.get(pair, 0) == -negc and -negc > 0:
                break
        else:
            break
        a, b = pair
        new_token = a + b
        merges.append(pair)
        vocab.append(new_token)
        affected = sorted(pair_words.pop(pair, ()))
        pair_count.pop(pair, None)
        for i in affected:
            symbols = splits[i]
            # Rebuild split merging all (a, b) occurrences left-to-right.
            out: list[bytes] = []
            j = 0
            changed = False
            while j < len(symbols):
                if j < len(symbols) - 1 and symbols[j] == a and symbols[j + 1] == b:
                    out.append(new_token)
                    j += 2
                    changed = True
                else:
                    out.append(symbols[j])
                    j += 1
            if not changed:
                continue
            old = symbols
            for x, y in zip(old, old[1:]):
                key = (x, y)
                if key == pair:
                    continue
                pair_count[key] -= counts[i]
                heapq.heappush(heap, (-pair_count[key], _MaxPair(key), key))
                members = pair_words.get(key)
                if members is not None:
                    members.discard(i)
                    if not members:
                        del pair_words[key]
            splits[i] = out
            for x, y in zip(out, out[1:]):
                key = (x, y)
                pair_count[key] += counts[i]
                heapq.heappush(heap, (-pair_count[key], _MaxPair(key), key))
                if key in pair_words:
                    pair_words[key].add(i)
                else:
                    pair_words[key] = {i}
    return vocab, merges
