"""SwiGLU feed-forward block, implemented from scratch (Step 7).

    y = down_proj(silu(gate_proj(x)) * up_proj(x))

The SiLU-gated branch lets the network modulate each hidden unit
multiplicatively; with gate outputs near zero the unit shuts off
(silu(0) = 0). Standard in modern decoder-only models. No biases —
normalization layers carry the shift terms.
"""
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Gated MLP: hidden -> intermediate (gated) -> hidden."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("sizes must be positive")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def extra_repr(self) -> str:
        return f"hidden={self.hidden_size}, intermediate={self.intermediate_size}"
