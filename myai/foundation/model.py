"""MyAI decoder-only transformer, assembled from scratch (Step 9).

    ids -> token embedding -> N x TransformerBlock (causal)
        -> final RMSNorm -> (tied) LM head -> logits

Weight tying: the LM head reuses the embedding matrix, so the model
learns one token representation for input and output (saves V*H
params). Greedy/sampling generation threads per-layer KV caches for
linear-time decoding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from .config import MyAIConfig
from .layers.block import TransformerBlock
from .layers.rmsnorm import RMSNorm


class MyAIModel(nn.Module):
    """Full decoder-only foundation model (random init, no pretrained weights)."""

    def __init__(self, config: MyAIConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    intermediate_size=config.intermediate_size,
                    max_seq_len=config.max_sequence_length,
                    theta=config.rope_theta,
                    dropout=config.dropout,
                    use_sdpa=config.use_sdpa,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self) -> None:
        """Scaled init (GPT-2 style): small std so initial loss ≈ ln(V).

        Default PyTorch inits are wrong here: nn.Embedding uses N(0,1),
        which through the *tied* head yields huge logits (loss ~23 for
        V=60 instead of ln(60)≈4.1) and destabilizes early training.
        Residual output projections (o_proj, down_proj) get an extra
        1/sqrt(2N) so residual-stream variance stays O(1) with depth.
        """
        import math

        std = 0.02
        res_std = std / math.sqrt(2 * max(1, self.config.num_layers))
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=std)
        if not self.config.tie_embeddings:
            # Untied head must not rely on framework defaults (audit P1):
            # same scale as the embedding it replaces.
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        for blk in self.blocks:
            for proj in (blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj, blk.mlp.gate_proj, blk.mlp.up_proj):
                torch.nn.init.normal_(proj.weight, mean=0.0, std=std)
            for proj in (blk.attn.o_proj, blk.mlp.down_proj):
                torch.nn.init.normal_(proj.weight, mean=0.0, std=res_std)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Training path: (B, T) ids -> (B, T, V) logits."""
        if input_ids.shape[1] + start_pos > self.config.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        h = self.embedding(input_ids)
        for blk in self.blocks:
            if self.config.grad_ckpt and self.training:
                # Closure (not extra args): non-reentrant ckpt takes tensors only.
                h = ckpt.checkpoint(
                    lambda x, _blk=blk, _pos=start_pos: _blk(x, start_pos=_pos),
                    h, use_reentrant=False)
            else:
                h = blk(h, start_pos=start_pos)
        return self.lm_head(self.final_norm(h))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Greedy (temp=0) or sampled decoding with per-layer KV caches."""
        self.eval()
        ids = input_ids.clone()
        caches: list = [None] * len(self.blocks)
        # Prefill the prompt through the cache path (equals forward output).
        h = self.embedding(ids)
        for i, blk in enumerate(self.blocks):
            h, caches[i] = blk.forward_with_kv(h, caches[i], start_pos=0)
        logits = self.lm_head(self.final_norm(h))
        for _ in range(max_new_tokens):
            next_id = self._sample(logits[:, -1, :], temperature)
            ids = torch.cat([ids, next_id], dim=1)
            if ids.shape[1] >= self.config.max_sequence_length:
                break
            h = self.embedding(next_id)
            for i, blk in enumerate(self.blocks):
                h, caches[i] = blk.forward_with_kv(h, caches[i], start_pos=ids.shape[1] - 1)
            logits = self.lm_head(self.final_norm(h))
        return ids

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        return torch.multinomial(
            F.softmax(logits / temperature, dim=-1).view(-1, logits.shape[-1]), 1
        ).view(logits.shape[0], 1)

    def num_parameters(self) -> int:
        """Count trainable params (tied head counted once via shared storage)."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def save_checkpoint(self, path: str, step: int = 0, tokens_seen: int = 0) -> None:
        """Save weights + config + progress (doc section 26 manifest)."""
        import datetime

        torch.save(
            {
                "model": self.state_dict(),
                "config": self.config.__dict__,
                "step": step,
                "tokens_seen": tokens_seen,
                "saved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str) -> tuple["MyAIModel", dict]:
        """Load weights + config; returns (model, metadata)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(MyAIConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        return model, {k: v for k, v in ckpt.items() if k != "model"}
