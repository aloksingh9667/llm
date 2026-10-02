"""Train MyAI on packed sequences (Steps 13/15, GPU-ready).

Usage (Kaggle, single GPU):
    python scripts/train.py --model-config configs/myai-100m.yaml \\
        --data-dir data/processed/fw10M-32k --batch-size 4 --accum 8 \\
        --amp --max-steps 5000 --out-dir checkpoints/myai-100m-v01

CPU smoke (this host):
    python scripts/train.py --model-config configs/myai-20m.yaml \\
        --data-dir data/processed/seed16 --max-steps 60

Model shape from YAML; vocab_size from the data manifest. --resume
continues step/tokens/optimizer. Reports JSON with loss history.
"""
import argparse
import itertools
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
    accum_step,
    build_optimizer,
    eval_loss,
    load_state,
    lr_at,
    micro_step,
    save_state,
    set_seed,
)


def _pairs(blob: dict):
    return list(zip(blob["input_ids"].tolist(), blob["labels"].tolist()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/myai-20m.yaml")
    parser.add_argument("--data-dir", default="data/processed/seed16")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--ckpt-every", type=int, default=0)
    parser.add_argument("--out-dir", default="checkpoints/seed20m")
    parser.add_argument("--report", default="reports/train-seed20m.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp and device == "cuda"
    if args.amp and device == "cpu":
        print("note: --amp ignored (no CUDA here); Kaggle GPU will use it")

    model_yaml = yaml.safe_load(Path(args.model_config).read_text(encoding="utf-8"))
    data = Path(args.data_dir)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    t = model_yaml.get("training", {})

    cfg = MyAIConfig.from_yaml(args.model_config)
    cfg.vocab_size = manifest["vocab_size"]
    tcfg = TrainConfig(
        learning_rate=t.get("learning_rate", 6e-4),
        weight_decay=t.get("weight_decay", 0.1),
        warmup_steps=t.get("warmup_steps", 100),
        max_steps=args.max_steps or t.get("max_steps", 1000),
        grad_clip=t.get("grad_clip", 1.0),
        batch_size=args.batch_size,
        accum_steps=args.accum,
        amp=use_amp,
        seed=t.get("seed", 1337),
    )
    set_seed(tcfg.seed)
    model = MyAIModel(cfg).to(device)
    opt = build_optimizer(model, tcfg)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    kw = dict(batch_size=tcfg.batch_size, shuffle=True)
    if device == "cuda":
        kw.update(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(
        PackedLMDataset(_pairs(torch.load(data / "train.pt", weights_only=True))), **kw
    )
    val_loader = DataLoader(PackedLMDataset(_pairs(torch.load(data / "val.pt", weights_only=True))),
                            batch_size=tcfg.batch_size)

    start_step, tokens_seen = 0, 0
    if args.resume:
        meta = load_state(args.resume, model, opt)
        start_step, tokens_seen = meta["step"], meta.get("tokens_seen", 0)
        print(f"resumed: step={start_step} tokens={tokens_seen}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history, first_loss, tok0, t0 = [], None, tokens_seen, time.time()
    step = start_step
    cycle = itertools.cycle(train_loader)
    while step < tcfg.max_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        opt.zero_grad()
        micro_loss, micro_toks = 0.0, 0
        for _ in range(tcfg.accum_steps):
            batch = {k: v.to(device, non_blocking=True) for k, v in next(cycle).items()}
            micro_loss += micro_step(model, batch, tcfg.grad_clip, scaler, tcfg.accum_steps)
            micro_toks += batch["input_ids"].numel()
        accum_step(model, opt, tcfg.grad_clip, scaler)
        step += 1
        tokens_seen += micro_toks
        first_loss = micro_loss if first_loss is None else first_loss
        if step % 10 == 0:
            dt = time.time() - t0
            print(f"step={step} loss={micro_loss:.4f} lr={opt.param_groups[0]['lr']:.2e} "
                  f"{(tokens_seen - tok0) / dt:.0f} tok/s [{device}]", flush=True)
            history.append({"step": step, "loss": micro_loss})
        if args.eval_every and step % args.eval_every == 0:
            v = eval_loss(model, val_loader)
            print(f"  eval step={step} loss={v:.4f}", flush=True)
            history.append({"step": step, "val_loss": v})
        if args.ckpt_every and step % args.ckpt_every == 0:
            save_state(str(out / f"step-{step:07d}.pt"), model, opt, step, tokens_seen, tcfg)
    val = eval_loss(model, val_loader)
    save_state(str(out / "final.pt"), model, opt, step, tokens_seen, tcfg)
    report = {
        "model": cfg.model_name, "params": model.num_parameters(),
        "vocab_size": cfg.vocab_size, "device": device, "amp": use_amp,
        "accum_steps": tcfg.accum_steps, "micro_batch": tcfg.batch_size,
        "steps": tcfg.max_steps - start_step, "first_loss": first_loss,
        "final_train_loss": history[-1].get("loss"),
        "val_loss": val,
        "val_perplexity": float(torch.exp(torch.tensor(val)).item()) if val == val else None,
        "tokens_seen": tokens_seen, "seconds": round(time.time() - t0, 1),
        "history": history,
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"done: {first_loss:.4f} -> {report['final_train_loss']:.4f} train, val={val:.4f}")
    print(f"checkpoint -> {out / 'final.pt'}  report -> {rp}")


if __name__ == "__main__":
    main()
