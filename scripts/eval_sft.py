"""Frozen-set SFT eval: generate answers for versioned prompts (Phase 1).

Usage:
    python scripts/eval_sft.py --checkpoint ckpts/xxx/final.pt \\
        --tokenizer data/tokenizer_corpus/tokenizer-32k.json \\
        --eval-set data/eval/sft-v01.json --report reports/sft-eval.json

Judgment is human/LLM review of the saved JSON (no fake auto-grading);
the script guarantees reproducibility (greedy, fixed format).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.foundation.model import MyAIModel
from myai.tokenizer.bpe import BPETokenizer
from scripts.pack_sft import format_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--eval-set", default="data/eval/sft-v01.json")
    parser.add_argument("--report", default="reports/sft-eval.json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer.load(args.tokenizer)
    model, meta = MyAIModel.load_checkpoint(args.checkpoint)
    model = model.to(device).eval()
    spec = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))

    results = []
    for item in spec["items"]:
        prompt, _ = format_pair(item)
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long).to(device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=args.max_new_tokens)
        new_ids = gen[0, ids.shape[1]:].tolist()
        results.append({"instruction": item["instruction"],
                        "reference": item["output"],
                        "generated": tok.decode(new_ids)})
        print(f"Q: {item['instruction'][:80]}")
        print(f"A: {results[-1]['generated'][:200]}")
        print("---")
    report = {"eval_set": args.eval_set, "eval_version": spec["version"],
              "checkpoint": args.checkpoint, "step": meta.get("step"),
              "results": results}
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report -> {rp} ({len(results)} answers, review manually)")


if __name__ == "__main__":
    main()
