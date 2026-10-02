"""Training loop from scratch (Step 11).

Causal-LM step: logits = model(input_ids); loss = CE(logits, labels).
Schedule: linear warmup -> cosine decay to min_lr. Stabilizers:
gradient clipping, AdamW, checkpoint save/resume with step +
tokens_seen + optimizer/scheduler state + RNG states (doc §26).
"""
import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 1000
    grad_clip: float = 1.0
    batch_size: int = 4
    accum_steps: int = 1  # micro-batches per optimizer step (GPU memory saver)
    amp: bool = False  # mixed precision (needs CUDA; auto-disabled on CPU)
    seed: int = 1337
    log_every: int = 10
    eval_every: int = 100
    ckpt_every: int = 500


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Warmup then cosine decay. Pure function — unit tested."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * cosine


def lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy (doc section 20 objective)."""
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))


def train_step(model, optimizer, batch, grad_clip: float) -> float:
    """One forward/backward/update. Returns detached loss."""
    model.train()
    logits = model(batch["input_ids"])
    loss = lm_loss(logits, batch["labels"])
    optimizer.zero_grad()
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


def micro_step(model, batch, grad_clip: float, scaler=None, accum_steps: int = 1) -> float:
    """One accumulation micro-step: forward+backward, no optimizer step.

    Loss is divided by accum_steps so that summed micro-gradients equal
    the full-batch gradient. With AMP, the scaler scales the loss.
    Returns the *unscaled* micro loss for logging.
    """
    model.train()
    device_type = "cuda" if next(model.parameters()).is_cuda else "cpu"
    use_amp = scaler is not None and device_type == "cuda"
    with torch.autocast(device_type=device_type, enabled=use_amp):
        logits = model(batch["input_ids"])
        loss = lm_loss(logits, batch["labels"]) / accum_steps
    raw = loss.item() * accum_steps
    if scaler is not None and use_amp:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    return raw


def accum_step(model, optimizer, grad_clip: float, scaler=None) -> None:
    """Apply accumulated gradients: unscale, clip, step, update scaler."""
    if scaler is not None and next(model.parameters()).is_cuda:
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


@torch.no_grad()
def eval_loss(model, loader: DataLoader) -> float:
    """Mean LM loss over a loader. Empty loader -> NaN (caller decides)."""
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        total += lm_loss(model(batch["input_ids"]), batch["labels"]).item()
        count += 1
    return total / count if count else float("nan")


def build_optimizer(model, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay only on matrices (no decay on norms/biases)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def save_state(path: str, model, optimizer, step: int, tokens_seen: int, cfg: TrainConfig) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": model.config.__dict__,
            "step": step,
            "tokens_seen": tokens_seen,
            "train_cfg": cfg.__dict__,
            "rng": torch.get_rng_state(),
        },
        path,
    )


def load_state(path: str, model, optimizer=None) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
