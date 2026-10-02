"""Near-deduplication via MinHash + LSH banding (audit P0).

Exact SHA-256 dedup misses reposts with a word changed. Each document
is sketched by K min-hash values over char 5-grams; LSH bands propose
candidate pairs and Jaccard similarity on sketches confirms them.
Deterministic: fixed hash seeds, no randomness.

Keeps the first occurrence, drops later near-dupes above threshold.
"""
import hashlib
from itertools import combinations


def _shingles(text: str, k: int = 5) -> set[bytes]:
    t = " ".join(text.split()).lower().encode("utf-8", "ignore")
    if len(t) <= k:
        return {t}
    return {t[i : i + k] for i in range(len(t) - k + 1)}


def _minhash(shingles: set[bytes], num_perm: int = 64) -> tuple[int, ...]:
    sig = []
    for seed in range(num_perm):
        best = None
        prefix = seed.to_bytes(4, "little")
        for s in shingles:
            h = int.from_bytes(hashlib.blake2b(prefix + s, digest_size=8).digest(), "little")
            if best is None or h < best:
                best = h
        sig.append(best if best is not None else 0)
    return tuple(sig)


def _jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(x == y for x, y in zip(a, b)) / max(1, len(a))


def near_dedup(
    docs: list[str],
    threshold: float = 0.8,
    num_perm: int = 64,
    bands: int = 8,
) -> tuple[list[str], list[int]]:
    """Return (kept_docs, removed_indices). First occurrence wins."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    sigs = [_minhash(_shingles(d), num_perm) for d in docs]
    rows = num_perm // bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for i, sig in enumerate(sigs):
        for b in range(bands):
            key = (b, sig[b * rows : (b + 1) * rows])
            buckets.setdefault(key, []).append(i)

    removed: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(members), 2):
            if (a, b) in seen_pairs or b in removed:
                continue
            seen_pairs.add((a, b))
            if _jaccard(sigs[a], sigs[b]) >= threshold:
                removed.add(b)
    kept = [d for i, d in enumerate(docs) if i not in removed]
    return kept, sorted(removed)
