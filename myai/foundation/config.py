"""MyAI model config placeholder (full dataclass lands in Steps 4-9)."""
from dataclasses import dataclass


@dataclass
class MyAIConfig:
    model_name: str = "MyAI-20M"
    vocab_size: int = 32000
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    max_sequence_length: int = 512
    normalization: str = "rmsnorm"
    activation: str = "swiglu"
    position_encoding: str = "rope"
    tie_embeddings: bool = True
