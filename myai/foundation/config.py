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

    @classmethod
    def from_yaml(cls, path: str) -> "MyAIConfig":
        """Load model fields from a YAML config (ignores training: section)."""
        import yaml

        raw = yaml.safe_load(open(path, encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})
