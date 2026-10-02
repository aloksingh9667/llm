"""End-to-end proof (doc section 43 items 7-14): random model -> generate.

Usage:
    python scripts/generate_sample.py [--config configs/myai-20m.yaml] [--prompt "hello world"]

Random weights => gibberish output. That is expected: this proves the
pipeline (init -> forward -> KV-cache generate -> checkpoint) works
before any training. Uses a tiny override config for CPU speed unless
--config points at a real one.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from myai.foundation.config import MyAIConfig
from myai.foundation.model import MyAIModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--prompt", default="hello world")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.config:
        cfg = MyAIConfig.from_yaml(args.config)
        # Cap context for the CPU demo; real training uses full length.
        cfg.max_sequence_length = min(cfg.max_sequence_length, 128)
    else:
        cfg = MyAIConfig(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=176,
            max_sequence_length=128,
        )
    model = MyAIModel(cfg).eval()
    toks = [ord(c) % cfg.vocab_size for c in args.prompt] or [0]
    prompt = torch.tensor([toks], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=args.max_new_tokens)
    print(f"params={model.num_parameters()} prompt_tokens={toks}")
    print(f"generated_ids={out[0].tolist()}")
    ckpt = Path("checkpoints") / "sample-model.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(ckpt), step=0, tokens_seen=0)
    print(f"checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
