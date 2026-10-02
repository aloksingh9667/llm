"""Step 16 quality gates: accumulation math == full batch, AMP-safe on CPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from myai.data.packed import PackedLMDataset, pack_tokens
from myai.foundation.config import MyAIConfig
from myai.foundation.model import MyAIModel
from myai.training.loop import accum_step, build_optimizer, micro_step, TrainConfig


def _setup(seed=0):
    torch.manual_seed(seed)
    cfg = MyAIConfig(
        vocab_size=32, hidden_size=24, num_layers=1, num_heads=4,
        num_kv_heads=2, intermediate_size=48, max_sequence_length=32,
    )
    ids = list(range(30)) * 6
    ds = PackedLMDataset(pack_tokens(ids, seq_len=16))
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    return MyAIModel(cfg), loader


def test_accumulation_equals_full_batch():
    """2 micro-batches (accum=2) must match 1 double-sized batch update."""
    tcfg = TrainConfig(learning_rate=1e-3)
    m1, loader = _setup()
    m2, _ = _setup()  # identical init
    o1, o2 = build_optimizer(m1, tcfg), build_optimizer(m2, tcfg)
    batches = [next(iter(loader)) for _ in range(2)]
    big = {k: torch.cat([b[k] for b in batches]) for k in ("input_ids", "labels")}

    # Path A: two micro-steps then one update.
    o1.zero_grad()
    for b in batches:
        micro_step(m1, b, grad_clip=0.0, accum_steps=2)
    accum_step(m1, o1, grad_clip=0.0)

    # Path B: single full-batch train_step equivalent (manual, same math).
    from myai.training.loop import lm_loss

    m2.train()
    loss = lm_loss(m2(big["input_ids"]), big["labels"])
    o2.zero_grad()
    loss.backward()
    o2.step()

    for a, b in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_micro_step_cpu_no_amp_by_default():
    m, loader = _setup()
    tcfg = TrainConfig(learning_rate=1e-3)
    opt = build_optimizer(m, tcfg)
    opt.zero_grad()
    batch = next(iter(loader))
    raw = micro_step(m, batch, grad_clip=1.0, scaler=None, accum_steps=1)
    assert raw > 0
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "micro-step must populate grads without stepping"
