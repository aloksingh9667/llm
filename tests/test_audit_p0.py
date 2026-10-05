"""Audit follow-ups: config validation, 20M size class, token budget."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from myai.foundation.config import MyAIConfig
from myai.foundation.model import MyAIModel


def test_stream_windows_partition_without_overlap():
    from scripts.stream_fineweb import windowed

    import pytest

    rows = [{"id": i} for i in range(100)]
    assert [r["id"] for r in windowed(rows, 0, 30)] == list(range(30))
    assert [r["id"] for r in windowed(rows, 30, 30)] == list(range(30, 60))
    assert [r["id"] for r in windowed(rows, 90, 30)] == list(range(90, 100))
    assert list(windowed(rows, 200, 10)) == []
    with pytest.raises(ValueError):
        list(windowed(rows, -1, 10))


def test_config_validation_catches_bad_dims():
    good = MyAIConfig()
    good.validate()
    bad_heads = MyAIConfig(hidden_size=255, num_heads=8)
    with pytest.raises(ValueError, match="hidden"):
        bad_heads.validate()
    bad_kv = MyAIConfig(num_heads=8, num_kv_heads=3)
    with pytest.raises(ValueError, match="kv-heads"):
        bad_kv.validate()
    bad_vocab = MyAIConfig(vocab_size=100)
    with pytest.raises(ValueError, match="vocab"):
        bad_vocab.validate()


def test_all_ship_configs_validate():
    for name in ("myai-20m.yaml", "myai-100m.yaml"):
        MyAIConfig.from_yaml(str(Path("configs") / name)).validate()


def test_20m_name_matches_size_class():
    cfg = MyAIConfig.from_yaml("configs/myai-20m.yaml")
    n = MyAIModel(cfg).num_parameters()
    assert 18_000_000 <= n <= 22_000_000, f"MyAI-20M has {n} params"


def test_exact_token_budget_counts(tmp_path):
    """Budget loop: stop at actual tokenizer tokens, not char estimates."""
    from myai.tokenizer.bpe import BPETokenizer

    tok = BPETokenizer().train("data/tokenizer_corpus/train.txt", vocab_size=300)
    docs = ["hello world, this is a test document here"] * 20
    total, kept = 0, 0
    for doc in docs:
        n = len(tok.encode(doc))
        if total + n > 100:
            break
        total += n
        kept += 1
    assert total <= 100 and kept > 0
    # Char heuristic disagrees with exact counts: budgets must be measured,
    # not estimated (audit section 15).
    char_est = sum(len(d) for d in docs[:kept]) // 4
    assert total != char_est
