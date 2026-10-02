"""Step 8 quality gates: residual identity, causality, KV-cache, gradients."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from myai.foundation.layers.block import TransformerBlock


def _block(**kw) -> TransformerBlock:
    args = dict(hidden_size=24, num_heads=4, num_kv_heads=2, intermediate_size=64)
    args.update(kw)
    return TransformerBlock(**args)


def test_shapes():
    torch.manual_seed(0)
    blk = _block().eval()
    x = torch.randn(2, 7, 24)
    out = blk(x)
    assert out.shape == (2, 7, 24) and torch.isfinite(out).all()


def test_zeroed_sublayers_give_identity():
    """Zero attention-out + MLP-down projs -> both residuals add 0."""
    torch.manual_seed(1)
    blk = _block().eval()
    with torch.no_grad():
        blk.attn.o_proj.weight.zero_()
        blk.mlp.down_proj.weight.zero_()
    x = torch.randn(2, 5, 24)
    torch.testing.assert_close(blk(x), x, rtol=0, atol=1e-6)


def test_prefix_invariant_to_future():
    torch.manual_seed(2)
    blk = _block().eval()
    x = torch.randn(1, 8, 24)
    y1 = blk(x)
    x2 = x.clone()
    x2[:, 5:, :] += 10.0
    y2 = blk(x2)
    torch.testing.assert_close(y1[:, :5, :], y2[:, :5, :], rtol=1e-4, atol=1e-5)


def test_incremental_matches_full():
    torch.manual_seed(3)
    blk = _block().eval()
    x = torch.randn(1, 6, 24)
    full = blk(x)
    cache, outs = None, []
    for t in range(6):
        o, cache = blk.forward_with_kv(x[:, t : t + 1, :], cache, start_pos=t)
        outs.append(o)
    torch.testing.assert_close(torch.cat(outs, dim=1), full, rtol=1e-4, atol=1e-5)


def test_gradient_reaches_input_and_all_params():
    torch.manual_seed(4)
    blk = _block()
    x = torch.randn(2, 4, 24, requires_grad=True)
    blk(x).pow(2).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    missing = [n for n, p in blk.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for {missing}"
    bad = [n for n, p in blk.named_parameters() if not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradient for {bad}"
