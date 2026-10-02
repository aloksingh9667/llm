"""Training loop from scratch (Step 11).

Causal-LM step: logits = model(input_ids); loss = CE(logits, labels).
Schedule: linear warmup -> cosine decay to min_lr. Stabilizers:
gradient clipping, AdamW, checkpoint save/resume with step +
tokens_seen + optimizer/scheduler state + RNG states (doc §26).
"""
import math
import random
import sys
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


def micro_step(model, batch, grad_clip: float, scaler=None, accum_steps: int = 1,
               amp_dtype: str = "fp16") -> float:
    """One accumulation micro-step: forward+backward, no optimizer step.

    Loss is divided by accum_steps so that summed micro-gradients equal
    the full-batch gradient. With AMP, the scaler scales the loss.
    fp16 needs a GradScaler; bf16 does not. Returns unscaled micro loss.
    """
    model.train()
    want = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    device_type = "cuda" if next(model.parameters()).is_cuda else "cpu"
    use_amp = scaler is not None and device_type == "cuda"
    if scaler is None and amp_dtype == "bf16" and device_type == "cuda":
        use_amp = True  # bf16 needs no loss scaler
    with torch.autocast(device_type=device_type, dtype=want, enabled=use_amp):
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
def eval_loss(model, loader: DataLoader, device: str | None = None) -> float:
    """Mean LM loss over a loader. Moves batches to the model's device
    by default (eval on GPU was crashing on CPU batches)."""
    model.eval()
    if device is None:
        try:
            device = str(next(model.parameters()).device)
        except StopIteration:
            device = "cpu"
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
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


def gpu_stats() -> dict:
    """Allocated/reserved MB + utilization where available; {} on CPU."""
    if not torch.cuda.is_available():
        return {}
    try:
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev) / 1e6
        reserved = torch.cuda.memory_reserved(dev) / 1e6
        util = torch.cuda.utilization(dev)
        return {"gpu_mem_alloc_mb": round(alloc, 1),
                "gpu_mem_reserved_mb": round(reserved, 1),
                "gpu_util_pct": util}
    except Exception:
        return {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _git_commit() -> str | None:
    try:
        import subprocess

        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _env_meta() -> dict:
    meta = {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        try:
            meta["gpu"] = torch.cuda.get_device_name(0)
            meta["cuda_version"] = torch.version.cuda
        except Exception:
            pass
    return meta


def save_state(path: str, model, optimizer, step: int, tokens_seen: int, cfg: TrainConfig,
               scaler=None, data_pos: dict | None = None, extra_meta: dict | None = None) -> None:
    """Full-fidelity checkpoint (audit P0): weights, optimizer, RNG states,
    AMP scaler, scheduler snapshot (step + config reproduces warmup-cosine),
    data position, git commit, and environment metadata."""
    import datetime
    import random

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model.config.__dict__,
        "step": step,
        "tokens_seen": tokens_seen,
        "train_cfg": cfg.__dict__,
        "scheduler": {"kind": "warmup-cosine", "step": step,
                      "learning_rate": cfg.learning_rate, "min_lr": cfg.min_lr,
                      "warmup_steps": cfg.warmup_steps, "max_steps": cfg.max_steps},
        "rng": {"python": random.getstate(), "torch_cpu": torch.get_rng_state()},
        "git_commit": _git_commit(),
        "env": _env_meta(),
        "saved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if torch.cuda.is_available():
        try:
            payload["rng"]["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    if scaler is not None and hasattr(scaler, "state_dict"):
        try:
            payload["scaler"] = scaler.state_dict()
        except Exception:
            pass
    if data_pos:
        payload["data_pos"] = data_pos
    if extra_meta:
        payload["meta"] = extra_meta
    torch.save(payload, path)


def load_state(path: str, model, optimizer=None, scaler=None, restore_rng: bool = True) -> dict:
    """Load checkpoint; optionally restores optimizer, scaler, and RNG states
    for exact-resume. Missing keys (old checkpoints) default gracefully."""
    import random

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass
    if restore_rng and "rng" in ckpt:
        try:
            random.setstate(ckpt["rng"]["python"])
            torch.set_rng_state(ckpt["rng"]["torch_cpu"])
            if "torch_cuda" in ckpt["rng"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ckpt["rng"]["torch_cuda"])
        except Exception:
            pass
    return ckpt
