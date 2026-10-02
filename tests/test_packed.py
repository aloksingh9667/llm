"""Step 10 quality gates: shift correctness, split coverage, model flow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn.functional as F

from myai.data.packed import PackedLMDataset, pack_tokens, train_val_split
from myai.foundation.config import MyAIConfig
from myai.foundation.model import MyAIModel


def test_shift_by_one_and_shapes():
    ids = list(range(200))
    pairs = pack_tokens(ids, seq_len=64)
    assert len(pairs) == 3  # floor((200-64)/64)
    for inp, lab in pairs:
        assert len(inp) == 64 and len(lab) == 64
        assert lab[:-1] == inp[1:], "labels must be input shifted by one"
        assert lab[-1] == inp[-1] + 1, "last label is the true next token"
    assert pairs[0][0][0] == 0 and pairs[1][0][0] == 64, "contiguous blocks"


def test_remainder_dropped_and_empty_guard():
    assert pack_tokens(list(range(65)), 64) == [(list(range(64)), list(range(1, 65)))]
    assert pack_tokens(list(range(64)), 64) == []
    with pytest.raises(ValueError):
        PackedLMDataset([])
    with pytest.raises(ValueError):
        pack_tokens([1, 2, 3], 0)


def test_split_disjoint_cover_all():
    pairs = [( [i], [i + 1]) for i in range(10)]
    train, val = train_val_split(pairs, val_ratio=0.2)
    assert len(train) == 8 and len(val) == 2
    assert train + val == pairs, "split must partition without reorder"
    with pytest.raises(ValueError):
        train_val_split(pairs, val_ratio=0.0)


def test_dataset_tensors_and_batch_flows_through_model():
    torch.manual_seed(0)
    ids = (list(range(60)) * 4)  # 240 ids, vocab 60
    pairs = pack_tokens(ids, seq_len=32)
    train, val = train_val_split(pairs, val_ratio=0.2)
    ds = PackedLMDataset(train)
    # Stack a manual batch of 4:
    batch = {k: torch.stack([ds[i][k] for i in range(4)]) for k in ("input_ids", "labels")}
    assert batch["input_ids"].shape == (4, 32)

    cfg = MyAIConfig(
        vocab_size=60, hidden_size=24, num_layers=1, num_heads=4,
        num_kv_heads=2, intermediate_size=48, max_sequence_length=64,
    )
    model = MyAIModel(cfg).eval()
    logits = model(batch["input_ids"])
    loss = F.cross_entropy(logits.view(-1, 60), batch["labels"].view(-1))
    assert torch.isfinite(loss) and loss > 0
    # Labels really are next tokens: loss on shifted input is the LM objective.
    assert loss.item() < 10.0, "random-init loss should be near ln(60)≈4.1"
