"""Rotary position embeddings, implemented from scratch (Step 5).

Idea: rotate each consecutive pair of features by an angle proportional
to the token position. Attention scores then depend on *relative*
distance (m - n) rather than absolute positions, with no learned tables.

For head dim d, pair i gets frequency theta_i = 10000^(-2i/d).
Position m rotates pair (x0, x1) by angle m*theta_i:

    [cos  -sin] [x0]
    [sin   cos] [x1]

Implemented in the interleaved "rotate-half" form: split x into first
and second halves (x_a, x_b), output (x_a*cos - x_b*sin,
x_b*cos + x_a*sin) with cos/sin duplicated per half. Norm-preserving
by construction.
"""
import torch
import torch.nn as nn


def build_rope_cache(
    max_seq_len: int, head_dim: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape (max_seq_len, head_dim)."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half).float() / half))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, inv_freq)  # (seq, half)
    emb = torch.cat([angles, angles], dim=-1)  # (seq, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Interleaved 90-degree rotation: (-x_b, x_a)."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0
) -> torch.Tensor:
    """Rotate sequence x (..., seq, head_dim) with tables (cache, head_dim)."""
    seq = x.shape[-2]
    c = cos[start_pos : start_pos + seq].to(x.dtype)
    s = sin[start_pos : start_pos + seq].to(x.dtype)
    # Broadcast over leading dims: view as (1, ..., seq, head_dim).
    while c.dim() < x.dim():
        c = c.unsqueeze(0)
        s = s.unsqueeze(0)
    return x * c + rotate_half(x) * s


class RotaryEmbedding(nn.Module):
    """Cached RoPE tables + application, one per attention layer shape."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        cos, sin = build_rope_cache(max_seq_len, head_dim, theta)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        if x.shape[-1] != self.head_dim:
            raise ValueError(f"last dim {x.shape[-1]} != head_dim {self.head_dim}")
        end = start_pos + x.shape[-2]
        if end > self.max_seq_len:
            raise ValueError(f"sequence [{start_pos}:{end}] exceeds cache {self.max_seq_len}")
        return apply_rope(x, self.cos_cached, self.sin_cached, start_pos)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, theta={self.theta}"
