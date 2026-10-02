"""Evaluate a trained checkpoint: val loss/perplexity + samples (Step 14).

Usage:
    python scripts/eval_model.py --checkpoint checkpoints/seed20m/final.pt \\
        --data-dir data/processed/seed16
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.data.packed import PackedLMDataset
from myai.foundation.model import MyAIModel
from myai.training.loop import eval_loss, load_state


def _pairs(blob: dict):
    return list(zip(blob["input_ids"].tolist(), blob["labels"].tolist()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed/seed16")
    parser.add_argument("--report", default="reports/eval-seed20m.json")
    parser.add_argument("--device", default=None,
                        help="cuda/cpu; defaults to the model's device")
    args = parser.parse_args()

    data = Path(args.data_dir)
    val_ds = PackedLMDataset(_pairs(torch.load(data / "val.pt", weights_only=True)))
    val_loader = DataLoader(val_ds, batch_size=4)

    model, meta = MyAIModel.load_checkpoint(args.checkpoint)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    val = eval_loss(model, val_loader)
    ppl = float(torch.exp(torch.tensor(val)).item())

    # Sample: greedy continuation of the first val input ids.
    prompt = val_ds.inputs[:1, :8].to(device)
    gen = model.generate(prompt, max_new_tokens=8).cpu()
    result = {
        "checkpoint": args.checkpoint,
        "step": meta.get("step"),
        "tokens_seen": meta.get("tokens_seen"),
        "val_loss": val,
        "val_perplexity": ppl,
        "prompt_ids": prompt[0].tolist(),
        "generated_ids": gen[0].tolist(),
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"val_loss={val:.4f} ppl={ppl:.2f} step={meta.get('step')}")
    print(f"prompt={result['prompt_ids']}")
    print(f"generated={result['generated_ids']}")
    print(f"report -> {rp}")


if __name__ == "__main__":
    main()
