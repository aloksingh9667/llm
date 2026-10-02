"""Step 3b quality gates: byte coverage, round-trip, determinism, save/load."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.tokenizer.bpe import BPETokenizer

SAMPLES = [
    "Language modeling predicts the next token.",
    "def causal_loss(logits, labels):\n    return loss",
    "Data quality decides model quality! 123, test...",
    "caf\u00e9 na\u00efve \u2014 unicode must round-trip \U0001f60a",
    "   leading spaces and\ttabs\n\nnewlines",
]


def _trained(vocab_size: int = 300) -> BPETokenizer:
    corpus = Path("data/tokenizer_corpus/train.txt")
    assert corpus.exists(), "run scripts/prepare_tokenizer_corpus.py first"
    return BPETokenizer().train(str(corpus), vocab_size=vocab_size)


def test_byte_coverage():
    tok = _trained()
    base = {bytes([i]) for i in range(256)}
    assert base.issubset(set(tok.vocab)), "every single byte must be encodable"


def test_round_trip():
    tok = _trained()
    for s in SAMPLES:
        assert tok.decode(tok.encode(s)) == s, f"round-trip failed for {s!r}"


def test_determinism():
    assert _trained().merges == _trained().merges


def test_save_load_roundtrip(tmp_path):
    tok = _trained()
    p = str(tmp_path / "tok.json")
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert len(tok2) == len(tok)
    assert tok2.merges == tok.merges
    for s in SAMPLES:
        assert tok2.encode(s) == tok.encode(s)
        assert tok2.decode(tok2.encode(s)) == s


def test_special_token_passthrough():
    tok = _trained()
    ids = tok.encode("hello <|endoftext|> world")
    assert tok.decode(ids) == "hello <|endoftext|> world"
    assert tok.token_to_id[b"<|endoftext|>"] in ids
