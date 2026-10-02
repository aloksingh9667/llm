"""Step 3c quality gates: eval metrics sanity + determinism."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.tokenizer.bpe import BPETokenizer
from myai.tokenizer.evaluate import evaluate_split, evaluate_tokenizer


def _trained() -> BPETokenizer:
    corpus = Path("data/tokenizer_corpus/train.txt")
    assert corpus.exists(), "run scripts/prepare_tokenizer_corpus.py first"
    return BPETokenizer().train(str(corpus), vocab_size=300)


def test_eval_metrics_sane():
    tok = _trained()
    m = evaluate_split(tok, ["hello world", "def f():\n    return 1"])
    assert m["n_docs"] == 2
    assert m["n_tokens"] > 0
    assert m["round_trip_rate"] == 1.0
    assert 0 < m["tokens_per_byte"] <= 4.0
    assert m["chars_per_token"] >= 1.0
    assert m["fertility"] >= 1.0


def _without_timing(report: dict) -> dict:
    import copy

    r = copy.deepcopy(report)
    for m in list(r["splits"].values()) + [r["overall"]]:
        m.pop("tokens_per_sec", None)
    return r


def test_eval_deterministic_and_split_aware():
    tok = _trained()
    splits = {"prose": ["hello world, this is prose."], "code": ["def f():\n    return 1"]}
    r1 = _without_timing(evaluate_tokenizer(tok, splits))
    r2 = _without_timing(evaluate_tokenizer(tok, splits))
    assert r1 == r2
    assert set(r1["splits"]) == {"prose", "code"}
    assert r1["overall"]["n_docs"] == 2
