"""Train MyAI on packed sequences (Steps 13/15).

Usage:
    python scripts/train.py --model-config configs/myai-20m.yaml \\
        --data-dir data/processed/seed16 --max-steps 60 --out-dir checkpoints/seed20m

Model shape comes from the YAML; vocab_size is set from the data
manifest (seed tokenizer is smaller than the 32k target). Supports
--resume CKPT for fault tolerance (doc section 26). Writes a JSON
report with loss history + val perplexity.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.data.packed import PackedLMDataset
from myai.foundation.config import MyAIConfig
from myai.foundation.model import MyAIModel
from myai.training.loop import (
    TrainConfig,
    build_optimizer,
    eval_loss,
    load_state,
    lr_at,
    save_state,
    set_seed,
    train_step,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/myai-20m.yaml")
    parser.add_argument("--data-dir", default="data/processed/seed16")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out-dir", default="checkpoints/seed20m")
    parser.add_argument("--report", default="reports/train-seed20m.json")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    model_yaml = yaml.safe_load(Path(args.model_config).read_text(encoding="utf-8"))
    data = Path(args.data_dir)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    t = model_yaml.get("training", {})

    cfg = MyAIConfig.from_yaml(args.model_config)
    cfg.vocab_size = manifest["vocab_size"]  # seed vocab, not the 32k target
    tcfg = TrainConfig(
        learning_rate=t.get("learning_rate", 6e-4),
        weight_decay=t.get("weight_decay", 0.1),
        warmup_steps=t.get("warmup_steps", 100),
        max_steps=args.max_steps or t.get("max_steps", 1000),
        grad_clip=t.get("grad_clip", 1.0),
        batch_size=args.batch_size,
        seed=t.get("seed", 1337),
    )
    set_seed(tcfg.seed)
    model = MyAIModel(cfg)
    opt = build_optimizer(model, tcfg)

    train_ds = PackedLMDataset(_pairs(torch.load(data / "train.pt", weights_only=True)))
    val_ds = PackedLMDataset(_pairs(torch.load(data / "val.pt", weights_only=True)))
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size)

    start_step, tokens_seen = 0, 0
    if args.resume:
        meta = load_state(args.resume, model, opt)
        start_step, tokens_seen = meta["step"], meta.get("tokens_seen", 0)
        print(f"resumed: step={start_step} tokens={tokens_seen}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.time()
    step = start_step
    import itertools

    cycle = itertools.cycle(train_loader)
    first_loss = None
    while step < tcfg.max_steps:
        batch = next(cycle)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        loss = train_step(model, opt, batch, tcfg.grad_clip)
        first_loss = loss if first_loss is None else first_loss
        tokens_seen += batch["input_ids"].numel()
        if (step + 1) % 10 == 0:
            history.append({"step": step + 1, "loss": loss})
            print(f"step={step + 1} loss={loss:.4f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        step += 1
    val = eval_loss(model, val_loader)
    dt = time.time() - t0
    save_state(str(out / "final.pt"), model, opt, step, tokens_seen, tcfg)
    report = {
        "model": cfg.model_name,
        "params": model.num_parameters(),
        "vocab_size": cfg.vocab_size,
        "steps": tcfg.max_steps - start_step,
        "first_loss": first_loss,
        "final_train_loss": history[-1]["loss"] if history else None,
        "val_loss": val,
        "val_perplexity": float(torch.exp(torch.tensor(val)).item()) if val == val else None,
        "tokens_seen": tokens_seen,
        "seconds": round(dt, 1),
        "history": history,
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"done: {first_loss:.4f} -> {history[-1]['loss']:.4f} train, val={val:.4f} ppl={report['val_perplexity']:.2f}")
    print(f"checkpoint -> {out / 'final.pt'}  report -> {rp}")


def _pairs(blob: dict) -> list[tuple[list[int], list[int]]]:
    return list(zip(blob["input_ids"].tolist(), blob["labels"].tolist()))


if __name__ == "__main__":
    main()
