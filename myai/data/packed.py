"""Packed causal-LM dataset, implemented from scratch (Step 10).

Pipeline: token ids -> contiguous non-overlapping blocks of seq_len+1
-> (input_ids, labels) with labels shifted one ahead:

    input  = block[0 : seq_len]
    labels = block[1 : seq_len+1]

so the model predicts token t+1 from tokens [..t]. Train/val split is a
prefix cut (first 1-val_ratio blocks train), keeping document order and
guaranteeing disjoint coverage. Stdlib + torch only.
"""
import torch
from torch.utils.data import Dataset


def pack_tokens(ids: list[int], seq_len: int) -> list[tuple[list[int], list[int]]]:
    """Cut ids into (input, labels) blocks. Drops the trailing remainder."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    pairs = []
    for start in range(0, len(ids) - seq_len, seq_len):
        block = ids[start : start + seq_len + 1]
        pairs.append((block[:-1], block[1:]))
    return pairs


def train_val_split(
    pairs: list, val_ratio: float = 0.05
) -> tuple[list, list]:
    """Prefix split: first (1-val_ratio) train, rest val. At least 1 val."""
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    n_val = max(1, int(len(pairs) * val_ratio)) if pairs else 0
    cut = len(pairs) - n_val
    return pairs[:cut], pairs[cut:]


class PackedLMDataset(Dataset):
    """torch Dataset of (input_ids, labels) LongTensors."""

    def __init__(self, pairs: list[tuple[list[int], list[int]]]):
        if not pairs:
            raise ValueError("no sequences — corpus too small for seq_len?")
        self.inputs = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        self.labels = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.inputs[idx], "labels": self.labels[idx]}
