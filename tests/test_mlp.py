"""Step 7 quality gates: SwiGLU numerics + gate ablation + gradients."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from myai.foundation.layers.mlp import SwiGLU


def test_shapes():
    torch.manual_seed(0)
    mlp = SwiGLU(hidden_size=32, intermediate_size=88).eval()
    x = torch.randn(2, 7, 32)
    out = mlp(x)
    assert out.shape == (2, 7, 32) and torch.isfinite(out).all()


def test_matches_naive_formula():
    torch.manual_seed(1)
    mlp = SwiGLU(hidden_size=16, intermediate_size=40).eval()
    x = torch.randn(3, 5, 16)
    got = mlp(x)
    expected = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    # Independent path: manual matmuls, no module reuse.
    g = x @ mlp.gate_proj.weight.t()
    u = x @ mlp.up_proj.weight.t()
    h = (g / (1 + torch.exp(-g))) * u
    manual = h @ mlp.down_proj.weight.t()
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got, manual, rtol=1e-5, atol=1e-6)


def test_gate_ablation_kills_output():
    """Zero gate -> silu(0)=0 -> gated product 0 -> down_proj(0)=0."""
    torch.manual_seed(2)
    mlp = SwiGLU(hidden_size=16, intermediate_size=40).eval()
    with torch.no_grad():
        mlp.gate_proj.weight.zero_()
    x = torch.randn(2, 4, 16)
    torch.testing.assert_close(
        mlp(x), torch.zeros(2, 4, 16), rtol=0, atol=1e-6
    )


def test_gate_is_nonlinear():
    """Halving pre-activation must NOT halve the gate (silu is nonlinear)."""
    torch.manual_seed(3)
    mlp = SwiGLU(hidden_size=8, intermediate_size=24).eval()
    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        g = mlp.gate_proj(x)
    half_scaled = F.silu(g * 0.5) * 2
    assert not torch.allclose(F.silu(g), half_scaled, atol=1e-3), (
        "gate looks linear — check activation"
    )


def test_gradient_flows_to_all_projections():
    torch.manual_seed(4)
    mlp = SwiGLU(hidden_size=12, intermediate_size=28)
    x = torch.randn(2, 5, 12, requires_grad=True)
    mlp(x).pow(2).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name in ("gate_proj", "up_proj", "down_proj"):
        g = getattr(mlp, name).weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name
