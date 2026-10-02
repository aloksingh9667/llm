"""P0 data gates: PII scrubbing + near-dedup behavior."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.data.dedup import near_dedup
from myai.data.filtering import scrub_doc, scrub_docs


def test_scrub_masks_and_counts():
    text = "Contact ada@example.com or +1-555-123-4567. api_key = 'sk-live-12345' here."
    clean, hits = scrub_doc(text)
    assert "ada@example.com" not in clean
    assert "sk-live-12345" not in clean
    assert hits.get("email") == 1
    assert "Contact" in clean and "here." in clean, "prose must survive"


def test_scrub_private_key_block():
    text = "header\n-----BEGIN RSA PRIVATE KEY-----\nMIIBfake\n-----END RSA PRIVATE KEY-----\nfooter"
    clean, hits = scrub_doc(text)
    assert "MIIBfake" not in clean and hits.get("private_key") == 1
    assert "header" in clean and "footer" in clean


def test_scrub_docs_aggregates():
    docs = ["mail bob@x.io", "nothing sensitive here", "call +1-555-000-1111 now"]
    _, total = scrub_docs(docs)
    assert total.get("email") == 1 and total.get("phone") == 1


def test_near_dedup_exact_and_near():
    base = "the quick brown fox jumps over the lazy dog near the river bank today"
    near = "the quick brown fox jumps over the lazy dog near the river bank today!"
    other = "quantum chromodynamics describes quarks and gluons inside hadrons fully"
    kept, removed = near_dedup([base, base, near, other], threshold=0.8)
    assert removed == [1, 2], f"exact + near-dupe must drop, got {removed}"
    assert kept == [base, other]


def test_near_dedup_keeps_distinct_and_first_wins():
    docs = ["alpha beta gamma delta epsilon zeta eta theta", "completely different words here now please",
            "alpha beta gamma delta epsilon zeta eta theta"]
    kept, removed = near_dedup(docs, threshold=0.8)
    assert removed == [2] and kept == docs[:2]
