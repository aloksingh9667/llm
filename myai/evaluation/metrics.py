"""Evaluation metrics from scratch (audit P1, section 28 foundation).

Loss/perplexity/token-accuracy over a loader; frozen eval sets should be
versioned under data/eval/ so runs stay comparable (next step).
"""
import math

import torch
import torch.nn.functional as F


def perplexity(loss: float) -> float:
    return math.exp(loss) if loss == loss else float("nan")


@torch.no_grad()
def evaluate(model, loader, device: str | None = None,
             ignore_index: int = -100) -> dict:
    """Mean loss, perplexity, and next-token accuracy (unmasked positions)."""
    model.eval()
    if device is None:
        try:
            device = str(next(model.parameters()).device)
        except StopIteration:
            device = "cpu"
    total_loss, total_correct, total_count = 0.0, 0, 0
    n_batches = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(ids)
        total_loss += F.cross_entropy(logits.view(-1, logits.shape[-1]),
                                      labels.view(-1), ignore_index=ignore_index).item()
        mask = labels.view(-1) != ignore_index
        if mask.any():
            pred = logits.view(-1, logits.shape[-1]).argmax(-1)[mask]
            total_correct += (pred == labels.view(-1)[mask]).sum().item()
            total_count += mask.sum().item()
        n_batches += 1
    loss = total_loss / n_batches if n_batches else float("nan")
    return {"loss": loss, "perplexity": perplexity(loss),
            "accuracy": total_correct / total_count if total_count else 0.0,
            "batches": n_batches}
