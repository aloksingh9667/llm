"""Single-GPU + torchrun-ready distributed helpers (audit P1).

Kaggle reality is one GPU per session (sessions are NOT a cluster), so
the default path stays single-process. These helpers make `train.py`
torchrun-ready for multi-GPU machines without changing its single-GPU
behavior: when WORLD_SIZE==1 everything is a no-op passthrough.
FSDP2 sharding follows the same pattern via `maybe_shard`.
"""
import os

import torch
import torch.distributed as dist


def dist_env() -> tuple[int, int, int]:
    """(rank, world_size, local_rank) from torchrun env, defaulting to solo."""
    return (int(os.environ.get("RANK", 0)),
            int(os.environ.get("WORLD_SIZE", 1)),
            int(os.environ.get("LOCAL_RANK", 0)))


def is_dist() -> bool:
    return dist_env()[1] > 1


def init_dist(backend: str | None = None) -> tuple[int, int]:
    """Init process group if distributed; returns (rank, world). Idempotent."""
    rank, world, _ = dist_env()
    if world == 1 or (dist.is_available() and dist.is_initialized()):
        return rank, world
    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=backend)
    return rank, world


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def maybe_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap in DDP when distributed, else return untouched."""
    if not is_dist():
        return model
    _, _, local_rank = dist_env()
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank] if torch.cuda.is_available() else None
    )


def maybe_shard(model: torch.nn.Module) -> torch.nn.Module:
    """FSDP2 full sharding when distributed, else return untouched."""
    if not is_dist():
        return model
    from torch.distributed._composable.fsdp import fully_shard

    return fully_shard(model)
