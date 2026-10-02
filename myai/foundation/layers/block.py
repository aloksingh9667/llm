"""Pre-norm transformer block, assembled from scratch (Step 8).

    h   = x + attention(rmsnorm_1(x))
    out = h + swiglu(rmsnorm_2(h))

Pre-norm (norm inside the residual branch) keeps the gradient highway
clean: with sublayers zeroed the block is the exact identity, and at
init the signal flows unimpeded through depth. KV-cache threads through
to the attention layer for incremental decoding.
"""
import torch.nn as nn

from .attention import CausalSelfAttention
from .mlp import SwiGLU
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """One pre-norm decoder block: attention sublayer + MLP sublayer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        intermediate_size: int | None = None,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = CausalSelfAttention(
            hidden_size, num_heads, num_kv_heads, max_seq_len, theta, dropout
        )
        self.mlp_norm = RMSNorm(hidden_size)
        inter = intermediate_size or 4 * hidden_size
        self.mlp = SwiGLU(hidden_size, inter)

    def forward(self, x, start_pos: int = 0):
        """Training path: full block over (B, T, hidden)."""
        h = x + self.attn(self.attn_norm(x), start_pos=start_pos)
        return h + self.mlp(self.mlp_norm(h))

    def forward_with_kv(self, x, past_kv, start_pos: int):
        """Incremental path: threads the KV cache through attention."""
        a, present = self.attn.forward_with_kv(self.attn_norm(x), past_kv, start_pos)
        h = x + a
        return h + self.mlp(self.mlp_norm(h)), present

    def extra_repr(self) -> str:
        return f"hidden={self.hidden_size}, heads={self.attn.num_heads}"
