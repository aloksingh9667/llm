"""Step 6 quality gates: causality, weights, GQA, KV-cache, gradients."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from myai.foundation.layers.attention import (
    CausalSelfAttention,
    causal_allowed,
    repeat_kv,
)


def test_shapes_and_row_stochastic():
    torch.manual_seed(0)
    attn = CausalSelfAttention(hidden_size=32, num_heads=4).eval()
    x = torch.randn(2, 6, 32)
    out, probs = attn(x, return_weights=True)
    assert out.shape == (2, 6, 32)
    assert probs.shape == (2, 4, 6, 6)
    torch.testing.assert_close(
        probs.sum(-1), torch.ones(2, 4, 6), rtol=1e-5, atol=1e-6
    )


def test_strictly_causal_weights():
    torch.manual_seed(1)
    attn = CausalSelfAttention(hidden_size=16, num_heads=2).eval()
    x = torch.randn(1, 5, 16)
    _, probs = attn(x, return_weights=True)
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert (probs[0, 0][future] == 0).all(), "future positions must get zero weight"
    assert (probs[..., 0, 0] == 1).all(), "first token attends only to itself"


def test_output_prefix_invariant_to_future():
    """Perturbing future tokens must not change earlier outputs."""
    torch.manual_seed(2)
    attn = CausalSelfAttention(hidden_size=24, num_heads=3).eval()
    x = torch.randn(1, 8, 24)
    y1 = attn(x)
    x2 = x.clone()
    x2[:, 5:, :] += 10.0
    y2 = attn(x2)
    torch.testing.assert_close(y1[:, :5, :], y2[:, :5, :], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(y1[:, 5:, :], y2[:, 5:, :]), "sanity: future changed"


def test_kv_cache_matches_full_forward():
    """Incremental forward_with_kv must equal the full-block forward."""
    torch.manual_seed(3)
    attn = CausalSelfAttention(hidden_size=24, num_heads=4, num_kv_heads=2).eval()
    x = torch.randn(1, 6, 24)
    full = attn(x)
    cache = None
    outs = []
    for t in range(6):
        o, cache = attn.forward_with_kv(x[:, t : t + 1, :], cache, start_pos=t)
        outs.append(o)
    incr = torch.cat(outs, dim=1)
    torch.testing.assert_close(incr, full, rtol=1e-4, atol=1e-5)


def test_gqa_grouping_and_mask_offsets():
    torch.manual_seed(4)
    attn = CausalSelfAttention(hidden_size=32, num_heads=8, num_kv_heads=2).eval()
    assert attn.groups == 4
    x = torch.randn(1, 4, 32)
    out = attn(x)
    assert out.shape == (1, 4, 32) and torch.isfinite(out).all()
    # repeat_kv unit check.
    kv = torch.randn(1, 2, 3, 4)
    assert repeat_kv(kv, 1) is kv
    assert repeat_kv(kv, 4).shape == (1, 8, 3, 4)
    # Mask with cache offset: query abs-pos 2..3 over 4 cached keys.
    m = causal_allowed(Tq=2, Tk=4, start_pos=2)
    assert m.tolist() == [[True, True, True, False], [True, True, True, True]]


def test_gradient_flows():
    torch.manual_seed(5)
    attn = CausalSelfAttention(hidden_size=16, num_heads=2)
    x = torch.randn(2, 4, 16, requires_grad=True)
    loss = attn(x).pow(2).sum()
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name, p in attn.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
