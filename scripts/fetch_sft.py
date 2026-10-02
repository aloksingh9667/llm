"""Fetch licensed SFT sources with manifests (doc section 9).

Sources (both permissive, both require attribution — recorded):
  - OpenAssistant OASST1 (Apache-2.0): https://huggingface.co/datasets/OpenAssistant/oasst1
  - Databricks Dolly 15k (CC-BY-SA-3.0): https://huggingface.co/datasets/databricks/databricks-dolly-15k

Usage:
    python scripts/fetch_sft.py --source oasst1 --max-pairs 5000
    python scripts/fetch_sft.py --source dolly --max-pairs 15000
    python scripts/fetch_sft.py --source seed --max-pairs 50   # offline hand-written mini-set

Writes data/sft/<source>.jsonl ({instruction, input, output}) + .manifest.json.
Keeps only first-turn English pairs; drops empties and >2k-char outliers.
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SOURCES = {
    "oasst1": {"hf": "OpenAssistant/oasst1", "license": "Apache-2.0",
               "url": "https://huggingface.co/datasets/OpenAssistant/oasst1"},
    "dolly": {"hf": "databricks/databricks-dolly-15k", "license": "CC-BY-SA-3.0",
              "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k"},
}

SEED_PAIRS = [
    ("Write a Python function that adds two numbers.",
     "", "def add(a, b):\n    return a + b"),
    ("What is the capital of France?", "", "The capital of France is Paris."),
    ("Explain what a verb is in one sentence.", "",
     "A verb is a word that describes an action or state."),
    ("Write a for loop in Python that prints numbers 0 to 4.",
     "", "for i in range(5):\n    print(i)"),
    ("What does CPU stand for?", "", "CPU stands for Central Processing Unit."),
    ("Translate 'good morning' to French.", "", "'Good morning' in French is 'bonjour'."),
    ("What is 12 times 8?", "", "12 times 8 is 96."),
    ("Name one use of a dictionary in Python.", "",
     "A Python dictionary stores key-value pairs for fast lookup by key."),
    ("What planet is known as the Red Planet?", "", "Mars is known as the Red Planet."),
    ("Write the first line of a bash script.", "", "#!/bin/bash"),
    ("What is the opposite of 'hot'?", "", "The opposite of 'hot' is 'cold'."),
    ("How do you print text in Python?", "", "Use print, for example: print('hello')."),
]


def today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def clean(text: str) -> str:
    return " ".join((text or "").split())


def pair_oasst_rows(rows, max_pairs: int) -> list[dict]:
    """Join assistant replies to their prompter parents (pure fn, tested offline)."""
    prompts: dict[str, str] = {}
    pairs, seen = [], set()
    for row in rows:
        if len(pairs) >= max_pairs:
            break
        if row.get("lang") != "en":
            continue
        if row.get("role") == "prompter":
            text = clean(row.get("text", ""))
            if text and len(text) <= 2000 and row.get("message_id"):
                prompts[row["message_id"]] = text
        elif row.get("role") == "assistant":
            prompt = prompts.get(row.get("parent_id", ""))
            output = clean(row.get("text", ""))
            if not prompt or not output or len(output) > 2000:
                continue
            h = hashlib.md5(prompt.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            pairs.append({"instruction": prompt, "input": "", "output": output})
    return pairs


def from_oasst1(max_pairs: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(SOURCES["oasst1"]["hf"], split="train", streaming=True)
    return pair_oasst_rows(ds, max_pairs)


def from_dolly(max_pairs: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(SOURCES["dolly"]["hf"], split="train", streaming=True)
    pairs = []
    for row in ds:
        if len(pairs) >= max_pairs:
            break
        instruction = clean(row.get("instruction", ""))
        output = clean(row.get("response", ""))
        if not instruction or not output or len(output) > 2000:
            continue
        pairs.append({"instruction": instruction, "input": clean(row.get("context", "")),
                      "output": output})
    return pairs


def from_seed(max_pairs: int) -> list[dict]:
    return [{"instruction": i, "input": x, "output": o}
            for i, x, o in SEED_PAIRS[:max_pairs]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["oasst1", "dolly", "seed"], default="seed")
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--out-dir", default="data/sft")
    args = parser.parse_args()

    fetch = {"oasst1": from_oasst1, "dolly": from_dolly, "seed": from_seed}[args.source]
    pairs = fetch(args.max_pairs)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.source}.jsonl").write_text(
        "\n".join(json.dumps(p) for p in pairs) + "\n", encoding="utf-8")
    info = SOURCES.get(args.source, {"hf": "hand-written", "license": "internal-seed",
                                     "url": "local"})
    manifest = {"name": f"sft-{args.source}", "version": "0.1.0",
                "source": info["url"], "hf_dataset": info["hf"],
                "license": info["license"], "download_date": today(),
                "allowed_use": "research with attribution per license",
                "pairs": len(pairs),
                "sha256": hashlib.sha256((out / f"{args.source}.jsonl").read_bytes()).hexdigest()}
    (out / f"{args.source}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{args.source}: {len(pairs)} pairs -> {out} [{info['license']}]")


if __name__ == "__main__":
    main()
