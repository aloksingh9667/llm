"""Stream a token-budgeted FineWeb sample to disk + manifest (Step 12).

Usage (Kaggle GPU):
    python scripts/stream_fineweb.py --max-tokens 10000000 \\
        --out data/raw/fineweb-10M.txt --min-chars 200 --lang en

Streams without downloading the corpus (HuggingFaceFW/fineweb,
ODC-By-1.0 — see doc section 10: provenance recorded, underlying page
terms apply). English-score filter + length gate keep quality; exact
SHA-256 dedup runs here so training never sees repeats.

Small-CPU validation:
    python scripts/stream_fineweb.py --max-tokens 20000 --max-docs 200 \\
        --out /tmp/fw-mini.txt
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASET = "HuggingFaceFW/fineweb"
LICENSE = "ODC-By-1.0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=10_000_000)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data/raw/fineweb-10M.txt")
    parser.add_argument("--char-per-token", type=float, default=4.0,
                        help="stop when chars >= max_tokens * ratio")
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer JSON for EXACT token budgets (audit P0): "
                             "counts actual tokens per doc, stops at max_tokens")
    args = parser.parse_args()

    from datasets import load_dataset

    tok = None
    if args.tokenizer:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from myai.tokenizer.bpe import BPETokenizer
        tok = BPETokenizer.load(args.tokenizer)

    ds = load_dataset(DATASET, split=args.split, streaming=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    n_docs = kept = 0
    n_chars = 0
    actual_tokens = 0
    target_chars = int(args.max_tokens * args.char_per_token)
    t0 = time.time()
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            if args.max_docs and n_docs >= args.max_docs:
                break
            if tok is None and n_chars >= target_chars:
                break
            if tok is not None and actual_tokens >= args.max_tokens:
                break
            n_docs += 1
            text = (row.get("text") or "").strip()
            if len(text) < args.min_chars:
                continue
            if args.lang and row.get("language") not in (None, args.lang):
                continue
            if float(row.get("language_score") or 1.0) < 0.5:
                continue
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            f.write(text + "\n\n")
            kept += 1
            n_chars += len(text)
            if tok is not None:
                actual_tokens += len(tok.encode(text))
            if kept % 2000 == 0:
                print(f"...{kept} docs {n_chars / 1e6:.1f}M chars", flush=True)

    manifest = {
        "name": out.stem,
        "version": "1.0.0",
        "source": f"https://huggingface.co/datasets/{DATASET}",
        "license": LICENSE,
        "allowed_use": "research — verify per dataset card + underlying page terms",
        "provenance": f"streamed sample, ~{args.max_tokens} token budget",
        "filters": ["min_chars", "language", "language_score>=0.5", "exact_dedup_sha256"],
        "docs_scanned": n_docs,
        "docs_kept": kept,
        "chars": n_chars,
        "token_budget": args.max_tokens,
        "actual_tokens": actual_tokens if tok is not None else None,
        "counting": "exact-tokenizer" if tok is not None else "char-estimate",
        "seconds": round(time.time() - t0, 1),
    }
    mp = out.with_suffix(".manifest.json")
    mp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"kept {kept}/{n_docs} docs, {n_chars} chars -> {out} + {mp.name}")


if __name__ == "__main__":
    main()
