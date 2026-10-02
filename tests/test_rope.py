"""Step 5 quality gates: RoPE rotation properties."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from myai.foundation.layers.rope import (
    RotaryEmbedding,
    apply_rope,
    build_rope_cache,
    rotate_half,
)


def test_cache_shapes_and_determinism():
    cos, sin = build_rope_cache(16, 32)
    assert cos.shape == (16, 32) and sin.shape == (16, 32)
    cos2, sin2 = build_rope_cache(16, 32)
    torch.testing.assert_close(cos, cos2)
    torch.testing.assert_close(sin, sin2)
    # Position 0 rotates by angle 0 -> identity rows.
    torch.testing.assert_close(cos[0], torch.ones(32), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sin[0], torch.zeros(32), rtol=1e-5, atol=1e-6)


def test_rotation_preserves_norm():
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=32, max_seq_len=64)
    x = torch.randn(2, 10, 32)
    y = rope(x, start_pos=5)
    assert y.shape == x.shape
    torch.testing.assert_close(
        y.float().pow(2).sum(-1), x.float().pow(2).sum(-1), rtol=1e-4, atol=1e-5
    )


def test_same_position_preserves_dot_product():
    """<R_m q, R_m k> == <q, k>: relative distance zero cancels rotation."""
    torch.manual_seed(1)
    rope = RotaryEmbedding(head_dim=24, max_seq_len=32)
    q = torch.randn(2, 6, 24)
    k = torch.randn(2, 6, 24)
    qr = rope(q, start_pos=7)
    kr = rope(k, start_pos=7)
    torch.testing.assert_close(
        (qr * kr).sum(-1), (q * k).sum(-1), rtol=1e-4, atol=1e-5
    )


def test_inverse_rotation_recovers_input():
    torch.manual_seed(2)
    cos, sin = build_rope_cache(32, 16)
    x = torch.randn(3, 8, 16)
    y = apply_rope(x, cos, sin, start_pos=4)
    back = apply_rope(y, cos[4:12], -sin[4:12], start_pos=0)
    torch.testing.assert_close(back, x, rtol=1e-4, atol=1e-5)


def test_matches_naive_pair_rotation():
    """Compare against an independent per-pair complex rotation."""
    torch.manual_seed(3)
    head_dim, seq, theta = 8, 5, 10000.0
    cos, sin = build_rope_cache(32, head_dim, theta)
    x = torch.randn(1, seq, head_dim)
    got = apply_rope(x, cos, sin, start_pos=2)

    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half).float() / half))
    expected = torch.zeros_like(x)
    for m in range(seq):
        angles = (2 + m) * inv_freq
        c, s = torch.cos(angles), torch.sin(angles)
        xa, xb = x[0, m, :half], x[0, m, half:]
        expected[0, m, :half] = xa * c - xb * s
        expected[0, m, half:] = xb * c + xa * s
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_start_pos_shifts_encoding():
    torch.manual_seed(4)
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
    x = torch.randn(1, 4, 16)
    a = rope(x, start_pos=0)
    b = rope(x, start_pos=10)
    assert not torch.allclose(a, b), "different positions must encode differently"
    # rotate_half sanity: applied twice negates.
    torch.testing.assert_close(rotate_half(rotate_half(x)), -x, rtol=0, atol=0)
