"""P1 gates: eval metrics + SDPA grad equivalence already covered in test_perf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from myai.data.packed import PackedLMDataset, pack_tokens
from myai.evaluation.metrics import evaluate, perplexity
from myai.foundation.config import MyAIConfig
from myai.foundation.model import MyAIModel


def test_perplexity_is_exp_loss():
    import math

    assert perplexity(0.0) == 1.0
    assert abs(perplexity(2.0) - math.e**2) < 1e-9


def test_evaluate_reports_loss_ppl_accuracy():
    torch.manual_seed(0)
    cfg = MyAIConfig(vocab_size=32, hidden_size=24, num_layers=1, num_heads=4,
                     num_kv_heads=2, intermediate_size=48, max_sequence_length=32)
    model = MyAIModel(cfg).eval()
    ids = list(range(30)) * 4
    ds = PackedLMDataset(pack_tokens(ids, seq_len=16))
    out = evaluate(model, DataLoader(ds, batch_size=4))
    assert out["loss"] > 0 and out["perplexity"] > 1.0
    assert 0.0 <= out["accuracy"] <= 1.0
    assert out["batches"] == 2  # 7 pairs / batch 4
