"""Step 4 quality gates: RMSNorm numerics + gradient flow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from myai.foundation.layers.rmsnorm import RMSNorm


def test_matches_naive_formula():
    torch.manual_seed(0)
    norm = RMSNorm(16)
    norm.weight.data = torch.randn(16) * 0.5 + 1.0
    x = torch.randn(2, 8, 16)
    got = norm(x)
    # Independent naive computation (no shared code path).
    rms = torch.sqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + norm.eps)
    expected = (x.float() / rms * norm.weight.float()).to(x.dtype)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_unit_weight_gives_unit_rms():
    torch.manual_seed(1)
    norm = RMSNorm(32)  # weight defaults to ones
    x = torch.randn(4, 32) * 3.0
    y = norm(x)
    rms = torch.sqrt(y.float().pow(2).mean(dim=-1))
    torch.testing.assert_close(rms, torch.ones(4), rtol=1e-4, atol=1e-5)


def test_gradient_flows_cleanly():
    torch.manual_seed(2)
    norm = RMSNorm(12)
    x = torch.randn(3, 5, 12, requires_grad=True)
    loss = norm(x).pow(2).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()
    assert x.grad.abs().sum() > 0, "input must receive gradient"
    assert norm.weight.grad.abs().sum() > 0, "weight must receive gradient"


def test_arbitrary_leading_shapes_and_scale_invariance():
    torch.manual_seed(3)
    norm = RMSNorm(10)
    x = torch.randn(2, 3, 4, 10)
    assert norm(x).shape == x.shape
    # Scaling input by k must not change output (RMS cancels the scale).
    k = 7.5
    torch.testing.assert_close(norm(x * k), norm(x), rtol=1e-4, atol=1e-5)
