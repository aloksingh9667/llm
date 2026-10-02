"""Byte-level BPE tokenizer, implemented from scratch (Step 3b).

English-first, stdlib only. Design notes:
  - Base vocab: 256 single bytes (UTF-8 bytes of pre-tokens), so any
    unicode string round-trips, including code and punctuation.
  - Pre-tokenization: GPT-2-style regex (ASCII-only, no third-party
    `regex` module needed). Keeps leading spaces with the word.
  - Training: classic BPE merge loop over pre-token frequencies.
    Deterministic: ties broken by explicit byte-order comparison, never
    by dict insertion order.
  - Save format: JSON with latin-1 reversible encoding of token bytes.

Complexity is O(words * merges * word_len); fine for the seed corpus.
The 32k run on 10M+ tokens (Step 3c) will need either a heap-optimized
Python loop or the HF `tokenizers` trainer as a *reference oracle* in
tests only — the implementation here stays ours.
"""
import json
import re
from collections import Counter
from pathlib import Path

# GPT-2-style pre-token pattern, ASCII-only (stdlib `re` has no \p{}).
# Order matters: contractions -> letter/number runs (optional leading
# space) -> symbol runs -> whitespace runs.
PRETOKEN_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[A-Za-z]+"
    r"| ?[0-9]+"
    r"| ?[^\\sA-Za-z0-9]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
PRETOKEN_RE = re.compile(PRETOKEN_PATTERN)

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def _latin_encode(s: str) -> bytes:
    return s.encode("latin-1")


def _latin_decode(b: bytes) -> str:
    return b.decode("latin-1")


class BPETokenizer:
    """Minimal byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: list[bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ):
        self.special_tokens = list(special_tokens) if special_tokens else []
        self.vocab: list[bytes] = list(vocab) if vocab else []
        self.merges: list[tuple[bytes, bytes]] = list(merges) if merges else []
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self.token_to_id: dict[bytes, int] = {tok: i for i, tok in enumerate(self.vocab)}
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        self._special_ids = {
            tok: self.token_to_id[tok.encode("utf-8")] for tok in self.special_tokens
        }

    # ---- training ----

    def train(
        self,
        corpus_path: str,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        min_frequency: int = 1,
    ) -> "BPETokenizer":
        """Learn merges from a normalized corpus file. Returns self."""
        self.special_tokens = list(special_tokens) if special_tokens is not None else list(
            DEFAULT_SPECIAL_TOKENS
        )
        text = Path(corpus_path).read_text(encoding="utf-8")
        words: Counter[tuple[bytes, ...]] = Counter()
        for match in PRETOKEN_RE.finditer(text):
            piece = match.group(0).encode("utf-8")
            if not piece:
                continue
            words[tuple(bytes([b]) for b in piece)] += 1

        # Filter rare words (keeps the common-case loop small).
        if min_frequency > 1:
            words = Counter({w: c for w, c in words.items() if c >= min_frequency})
        if not words:
            raise ValueError(
                "no words to train on — corpus empty or "
                "min_frequency filtered everything out"
            )

        self.vocab = [bytes([i]) for i in range(256)]
        self.merges = []
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in self.vocab:
                self.vocab.append(b)
        num_merges = max(0, vocab_size - len(self.vocab))

        for _ in range(num_merges):
            pair_counts: Counter[tuple[bytes, bytes]] = Counter()
            for word, freq in words.items():
                for a, b in zip(word, word[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            # Deterministic pick: highest count, ties -> larger byte pair.
            best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            new_token = best[0] + best[1]
            self.merges.append(best)
            self.vocab.append(new_token)
            words = self._apply_merge(words, best, new_token)

        self._rebuild_index()
        return self

    @staticmethod
    def _apply_merge(
        words: Counter[tuple[bytes, ...]], pair: tuple[bytes, bytes], new_token: bytes
    ) -> Counter[tuple[bytes, ...]]:
        merged: Counter[tuple[bytes, ...]] = Counter()
        a, b = pair
        for word, freq in words.items():
            out: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(new_token)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            merged[tuple(out)] += freq
        return merged

    # ---- encoding ----

    def _encode_word(self, piece: bytes) -> list[int]:
        word = tuple(bytes([b]) for b in piece)
        if len(word) == 1:
            return [self.token_to_id[word[0]]]
        while len(word) > 1:
            best_rank = None
            best_idx = -1
            for i in range(len(word) - 1):
                rank = self.merge_rank.get((word[i], word[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            word = word[:best_idx] + (word[best_idx] + word[best_idx + 1],) + word[best_idx + 2 :]
        return [self.token_to_id[t] for t in word]

    def encode(self, text: str) -> list[int]:
        """Encode text to ids. Special tokens pass through by exact match."""
        if not self.vocab:
            raise ValueError("tokenizer is untrained — call train() or load() first")
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]
        ids: list[int] = []
        for part in parts:
            if part in self._special_ids:
                ids.append(self._special_ids[part])
            elif part:
                for match in PRETOKEN_RE.finditer(part):
                    piece = match.group(0).encode("utf-8")
                    if piece:
                        ids.extend(self._encode_word(piece))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode ids back to text. Lossless for encoder-produced ids."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ---- persistence ----

    def save(self, path: str) -> None:
        payload = {
            "version": "0.1.0",
            "pretoken_pattern": PRETOKEN_PATTERN,
            "special_tokens": self.special_tokens,
            "vocab": [_latin_decode(t) for t in self.vocab],
            "merges": [[_latin_decode(a), _latin_decode(b)] for a, b in self.merges],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = [_latin_encode(t) for t in payload["vocab"]]
        merges = [(_latin_encode(a), _latin_encode(b)) for a, b in payload["merges"]]
        return cls(vocab=vocab, merges=merges, special_tokens=payload.get("special_tokens", []))

    def __len__(self) -> int:
        return len(self.vocab)
