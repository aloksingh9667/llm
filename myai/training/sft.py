"""SFT example building: prompt masking from scratch (Step 18).

Format per example:
    <prompt ids> <response ids> <eos>

Labels are IGNORE (-100) over the prompt so cross-entropy trains only
on the response (instruction-following signal, not prompt modeling).
Pairs are packed greedily into seq_len blocks; a response split across
a boundary keeps its mask per token.
"""
import torch

IGNORE = -100


def build_sft_pair(prompt_ids: list[int], response_ids: list[int], eos_id: int) -> tuple[list[int], list[int]]:
    """One (input, labels) pair with prompt positions masked."""
    inp = prompt_ids + response_ids + [eos_id]
    labels = [IGNORE] * len(prompt_ids) + response_ids + [eos_id]
    return inp, labels


def pack_sft_pairs(
    pairs: list[tuple[list[int], list[int]]], seq_len: int
) -> tuple[list[tuple[list[int], list[int]]], int]:
    """Greedy-concatenate pairs into seq_len blocks (labels travel along).

    Drops blocks with zero trainable tokens (all IGNORE): cross-entropy
    over a fully-masked block is NaN and would poison training/eval.
    Returns (blocks, dropped_count).
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    flat_inp, flat_lab, blocks = [], [], []
    dropped = 0
    for inp, lab in pairs:
        assert len(inp) == len(lab)
        flat_inp.extend(inp)
        flat_lab.extend(lab)
    for s in range(0, len(flat_inp) - seq_len + 1, seq_len):
        block_lab = flat_lab[s : s + seq_len]
        if not any(t != IGNORE for t in block_lab):
            dropped += 1
            continue
        blocks.append((flat_inp[s : s + seq_len], block_lab))
    return blocks, dropped


def has_trainable_tokens(labels, ignore_index: int = IGNORE) -> bool:
    """False when a batch carries no learning signal (skip it, don't NaN)."""
    import torch

    if isinstance(labels, torch.Tensor):
        return bool((labels != ignore_index).any().item())
    return any(t != ignore_index for t in labels)


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over unmasked positions only."""
    import torch.nn.functional as F

    return F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=IGNORE)
