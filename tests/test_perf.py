"""P1 perf gates: SDPA==reference, grad-ckpt==plain, dist solo no-op."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from myai.foundation.config import MyAIConfig
from myai.foundation.layers.attention import CausalSelfAttention
from myai.foundation.model import MyAIModel
from myai.training.distributed import cleanup, dist_env, init_dist, is_dist, maybe_ddp, maybe_shard


def test_sdpa_matches_reference():
    torch.manual_seed(0)
    kw = dict(hidden_size=32, num_heads=4, num_kv_heads=2)
    ref = CausalSelfAttention(**kw, use_sdpa=False).eval()
    fast = CausalSelfAttention(**kw, use_sdpa=True).eval()
    fast.load_state_dict(ref.state_dict())
    x = torch.randn(2, 9, 32)
    torch.testing.assert_close(fast(x), ref(x), rtol=1e-4, atol=1e-5)


def test_grad_ckpt_matches_plain_and_flows():
    torch.manual_seed(1)
    cfg = MyAIConfig(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4,
                     num_kv_heads=2, intermediate_size=80, max_sequence_length=32)
    plain = MyAIModel(cfg).eval()
    ckpt = MyAIModel(cfg)
    ckpt.load_state_dict(plain.state_dict())
    ckpt.config.grad_ckpt = True
    ckpt.train()
    ids = torch.randint(0, 64, (2, 12))
    torch.testing.assert_close(ckpt(ids), plain(ids), rtol=1e-4, atol=1e-5)
    ckpt(ids).pow(2).sum().backward()
    assert all(p.grad is not None for p in ckpt.parameters() if p.requires_grad)


def test_dist_solo_is_noop_and_gloo_ok():
    assert dist_env() == (0, 1, 0) and not is_dist()
    model = torch.nn.Linear(4, 4)
    assert maybe_ddp(model) is model and maybe_shard(model) is model
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29511"})
    try:
        assert init_dist() == (0, 1)  # solo: no group, pure passthrough
        import torch.distributed as dist

        dist.init_process_group("gloo", rank=0, world_size=1)  # explicit 1-proc group
        t = torch.ones(2)
        dist.broadcast(t, src=0)
        assert torch.equal(t, torch.ones(2))
    finally:
        cleanup()
        for k in ("MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(k, None)
