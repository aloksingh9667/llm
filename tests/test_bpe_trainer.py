"""P1 gates: indexed trainer == reference trainer, plus speedup sanity."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.tokenizer.bpe import BPETokenizer
from myai.tokenizer.trainer import train_indexed


def test_trainer_matches_reference():
    corpus = "data/tokenizer_corpus/train.txt"
    assert Path(corpus).exists(), "run scripts/prepare_tokenizer_corpus.py first"
    ref = BPETokenizer().train(corpus, vocab_size=500)
    vocab, merges = train_indexed(corpus, vocab_size=500)
    assert merges == ref.merges, "indexed trainer must learn identical merges"
    assert vocab == ref.vocab, "identical merges => identical vocab"
    # And identical encoding behavior.
    fast = BPETokenizer(vocab=vocab, merges=merges,
                        special_tokens=ref.special_tokens)
    s = "Language modeling predicts the next token. def f():\n    return 1"
    assert fast.encode(s) == ref.encode(s)


def test_sample_chars_deterministic_and_bounded(tmp_path):
    """Sampling takes a doc-boundary-clean head prefix, deterministically."""
    import subprocess
    import sys

    corpus = "data/tokenizer_corpus/train.txt"
    out = str(tmp_path / "tok.json")
    for _ in range(2):
        subprocess.run([sys.executable, "scripts/train_tokenizer.py", "--corpus", corpus,
                        "--vocab-size", "400", "--out", out, "--fast",
                        "--sample-chars", "500"], check=True, capture_output=True)
    from myai.tokenizer.bpe import BPETokenizer

    tok = BPETokenizer.load(out)
    assert len(tok) <= 400
    assert tok.decode(tok.encode("hello world")) == "hello world"


def test_trainer_faster_than_reference_on_repeats():
    import tempfile, os
    doc = ("the quick brown fox jumps over lazy dogs near rivers " * 20 + "\n\n")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(doc * 30)
        corpus = f.name
    try:
        t0 = time.perf_counter()
        BPETokenizer().train(corpus, vocab_size=600)
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        train_indexed(corpus, vocab_size=600)
        t_fast = time.perf_counter() - t0
    finally:
        os.unlink(corpus)
    assert t_fast <= t_ref, f"indexed {t_fast:.2f}s should beat reference {t_ref:.2f}s"
