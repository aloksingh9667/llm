"""Root-mean-square normalization, implemented from scratch (Step 4).

Formula (norm over last dim, scale by learned weight):

    rms = sqrt(mean(x^2) + eps)
    y = (x / rms) * weight

No mean-centering (unlike LayerNorm) — cheaper and standard in modern
decoder-only models (LLaMA/Mistral family use this form). Weight inits
to ones so the layer starts as identity-up-to-scale.
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"last dim {x.shape[-1]} != norm dim {self.dim}")
        # Compute scale in float32 for stability, cast back afterwards.
        xf = x.float()
        rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        ynorm = xf / rms
        return (ynorm * self.weight.float()).to(x.dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
