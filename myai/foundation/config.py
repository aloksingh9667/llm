"""MyAI model config (extended in Step 9 for the full model)."""
from dataclasses import dataclass


@dataclass
class MyAIConfig:
    model_name: str = "MyAI-20M"
    vocab_size: int = 32000
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 8
    intermediate_size: int = 688
    max_sequence_length: int = 512
    normalization: str = "rmsnorm"
    activation: str = "swiglu"
    position_encoding: str = "rope"
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    # Performance switches (audit P1): reference math stays default.
    use_sdpa: bool = False  # fused attention kernel (train-shaped calls)
    grad_ckpt: bool = False  # activation checkpointing (trades compute for memory)

    @classmethod
    def from_yaml(cls, path: str) -> "MyAIConfig":
        """Load model fields from a YAML config (ignores training: section)."""
        import yaml

        raw = yaml.safe_load(open(path, encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def validate(self) -> None:
        """Fail fast on inconsistent dims (audit: config validation)."""
        c = self
        if c.hidden_size % c.num_heads != 0:
            raise ValueError(f"hidden {c.hidden_size} % heads {c.num_heads} != 0")
        if c.num_heads % c.num_kv_heads != 0:
            raise ValueError(f"heads {c.num_heads} % kv-heads {c.num_kv_heads} != 0")
        if c.intermediate_size <= 0 or c.hidden_size <= 0 or c.num_layers <= 0:
            raise ValueError("sizes and layers must be positive")
        if c.vocab_size <= 256:
            raise ValueError(f"vocab {c.vocab_size} leaves no room for merges")
        if c.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
