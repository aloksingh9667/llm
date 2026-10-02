"""Causal self-attention, implemented from scratch (Step 6).

Structure per forward pass (training shape, start_pos=0):

    x -> Q,K,V projections -> reshape heads -> RoPE on Q,K
      -> scores = Q K^T / sqrt(head_dim) -> causal mask -> softmax
      -> weighted sum V -> merge heads -> output projection

Grouped-query attention: num_kv_heads <= num_heads; each KV head is
shared by a group of Q heads (num_heads // num_kv_heads). num_kv_heads
= num_heads is plain multi-head attention.

The causal mask is start_pos-aware so the same module serves KV-cache
inference later: query position m may attend key position n iff n <= m.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rope


def causal_allowed(Tq: int, Tk: int, start_pos: int = 0) -> torch.Tensor:
    """Bool matrix (Tq, Tk): True where query may attend key.

    Query rows correspond to absolute positions [start_pos, start_pos+Tq);
    key columns to [0, Tk). Requires Tk >= start_pos + Tq.
    """
    if Tk < start_pos + Tq:
        raise ValueError(f"key length {Tk} < {start_pos} + {Tq}")
    qpos = torch.arange(start_pos, start_pos + Tq).unsqueeze(1)
    kpos = torch.arange(Tk).unsqueeze(0)
    return kpos <= qpos


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Expand (B, nkv, T, hd) -> (B, nh, T, hd) by repeating head groups."""
    if groups == 1:
        return x
    return x.repeat_interleave(groups, dim=1)


class CausalSelfAttention(nn.Module):
    """Single causal self-attention layer (no cross-attention)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden {hidden_size} not divisible by heads {num_heads}")
        nkv = num_kv_heads if num_kv_heads is not None else num_heads
        if num_heads % nkv != 0:
            raise ValueError(f"heads {num_heads} not divisible by kv-heads {nkv}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = nkv
        self.head_dim = hidden_size // num_heads
        self.groups = num_heads // nkv

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, nkv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, nkv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, theta)
        self.attn_dropout = dropout

    def _project(self, x: torch.Tensor):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
        return_weights: bool,
    ):
        Tq, Tk = q.shape[-2], k.shape[-2]
        # q/k are (B, heads, seq, head_dim): apply_rope broadcasts over the
        # leading dims and reads the sequence from dim -2. (Never route these
        # through RotaryEmbedding.forward — it expects seq at dim -2 of a
        # (..., seq, head_dim) tensor without a head axis.)
        q = apply_rope(q, self.rope.cos_cached, self.rope.sin_cached, start_pos)
        # NOTE: k is passed in post-RoPE (cache stores rotated keys, whose
        # angles already encode absolute position), so no re-rotation here.
        k = repeat_kv(k, self.groups)
        v = repeat_kv(v, self.groups)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = causal_allowed(Tq, Tk, start_pos).to(q.device)
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        if self.training and self.attn_dropout > 0:
            probs = F.dropout(probs, p=self.attn_dropout)
        out = (probs @ v).transpose(1, 2).reshape(
            probs.shape[0], Tq, self.hidden_size
        )
        out = self.o_proj(out)
        if return_weights:
            return out, probs
        return out

    def forward(
        self, x: torch.Tensor, start_pos: int = 0, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Training path: full block, positions [start_pos, start_pos+T)."""
        q, k, v = self._project(x)
        k = apply_rope(k, self.rope.cos_cached, self.rope.sin_cached, start_pos)
        return self._core(q, k, v, start_pos, return_weights)

    def forward_with_kv(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None,
        start_pos: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Incremental path: append one block to the KV cache.

        past_kv holds post-RoPE keys + values shaped (B, nkv, Tp, hd);
        start_pos must equal Tp. Returns (output, updated cache).
        """
        q, k_cur, v_cur = self._project(x)
        k_cur = apply_rope(k_cur, self.rope.cos_cached, self.rope.sin_cached, start_pos)
        if past_kv is None:
            if start_pos != 0:
                raise ValueError(f"no cache but start_pos={start_pos}")
            k_full, v_full = k_cur, v_cur
        else:
            past_k, past_v = past_kv
            if past_k.shape[-2] != start_pos:
                raise ValueError(
                    f"cache length {past_k.shape[-2]} != start_pos {start_pos}"
                )
            k_full = torch.cat([past_k, k_cur], dim=-2)
            v_full = torch.cat([past_v, v_cur], dim=-2)
        out = self._core(q, k_full, v_full, start_pos, False)
        return out, (k_full, v_full)

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, dropout={self.attn_dropout}"
        )
