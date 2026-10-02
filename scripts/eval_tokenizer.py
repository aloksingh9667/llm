"""Evaluate the Step 3b tokenizer and record the freeze decision (Step 3c).

Usage:
    python scripts/eval_tokenizer.py [--config configs/tokenizer-train.yaml]

Splits: prose vs code (doc section 8 wants compression/coverage per
domain). Writes reports/tokenizer-3c.json and prints a markdown table.
If the trained tokenizer file is missing, trains it first from config.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.tokenizer.bpe import BPETokenizer
from myai.tokenizer.evaluate import evaluate_tokenizer

PROSE_HELDOUT = [
    "Evaluation tracks validation loss and downstream benchmarks, not just perplexity.",
    "The model reads tokens from left to right with causal masking.",
]

CODE_HELDOUT = [
    "def train_step(batch):\n    logits = model(batch['input_ids'])\n    return loss",
    "for batch in dataloader:\n    optimizer.zero_grad()\n    loss.backward()\n    optimizer.step()",
]

# Informational probes (audit section 11): fertility recorded, not gated.
# English-first policy stands; these quantify future multilingual work.
MULTILINGUAL_PROBES = {
    "hindi": ["यह एक परीक्षण वाक्य है जो टोकनाइज़र की क्षमता को मापता है"],
    "hinglish": ["yeh model bahut achha kaam kar raha hai bhai"],
}


def load_corpus_splits(corpus_path: str) -> dict[str, list[str]]:
    prose, code = list(PROSE_HELDOUT), list(CODE_HELDOUT)
    p = Path(corpus_path)
    if p.exists():
        for doc in p.read_text(encoding="utf-8").split("\n\n"):
            doc = doc.strip()
            if not doc:
                continue
            if "def " in doc or "import " in doc or "(" in doc and ")" in doc:
                code.append(doc)
            else:
                prose.append(doc)
    return {"prose": prose, "code": code}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer-train.yaml")
    parser.add_argument("--report", default="reports/tokenizer-3c.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tok_path = Path(cfg["output_path"])
    if not tok_path.exists():
        print(f"tokenizer missing at {tok_path}, training first...")
        tok = BPETokenizer().train(
            cfg["corpus_path"],
            vocab_size=cfg["vocab_size"],
            special_tokens=cfg.get("special_tokens"),
            min_frequency=cfg.get("min_frequency", 1),
        )
        tok.save(str(tok_path))
    tok = BPETokenizer.load(str(tok_path))

    splits = load_corpus_splits(cfg["corpus_path"])
    report = evaluate_tokenizer(tok, splits)
    report["multilingual_probes"] = {
        name: evaluate_tokenizer(tok, {name: docs})["splits"][name]
        for name, docs in MULTILINGUAL_PROBES.items()
    }
    report["tokenizer_path"] = str(tok_path)
    report["freeze_decision"] = (
        "SEED-DEMO ONLY — 798-char corpus supports 313 merges (vocab 570); "
        "the 32k freeze needs the 10M+ token FineWeb/Stack mix (Step 3c-target). "
        "Procedure (metrics + report format) is frozen; vocab is NOT."
    )
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"vocab={report['vocab_size']} merges={report['n_merges']}")
    print("| split | docs | tok/byte | chars/tok | fertility | tok/sec | round-trip |")
    print("|---|---|---|---|---|---|---|")
    for name, m in {**report["splits"], "overall": report["overall"]}.items():
        print(
            f"| {name} | {m['n_docs']} | {m['tokens_per_byte']:.3f} | "
            f"{m['chars_per_token']:.2f} | {m['fertility']:.2f} | "
            f"{m['tokens_per_sec']:.0f} | {m['round_trip_rate']:.2f} |"
        )
    print(f"report -> {rp}")
    print(f"freeze: {report['freeze_decision']}")


if __name__ == "__main__":
    main()
