"""Build the Step 3a tokenizer corpus (offline-capable).

Usage:
    python scripts/prepare_tokenizer_corpus.py [--config configs/tokenizer-corpus.yaml]

If data/raw is empty, writes a small built-in English+code seed so the
pipeline (collect -> normalize -> dedup -> manifest) is testable offline.
Real FineWeb/Stack streaming replaces the seed in Step 3b.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.tokenizer.corpus import build_tokenizer_corpus

SEED_DOCS = {
    "seed_en_prose_001.txt": """Language modeling predicts the next token given previous tokens. \
A decoder-only Transformer reads tokens from left to right with causal masking, \
so each position only attends to earlier positions. Training minimizes \
next-token cross-entropy over a large corpus.
""",
    "seed_en_prose_002.txt": """Data quality decides model quality. Provenance records the source, \
license, and download date of every dataset. Deduplication removes exact and \
near-duplicate documents so frequent text does not dominate training. \
Evaluation tracks validation loss and downstream benchmarks, not just perplexity.
""",
    "seed_en_code_001.py": '''def causal_loss(logits, labels):
    """Next-token cross-entropy, ignoring padded positions."""
    import torch
    import torch.nn.functional as F
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss
''',
}


def ensure_seed(raw_dir: Path) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in raw_dir.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    if existing:
        return 0
    for name, text in SEED_DOCS.items():
        (raw_dir / name).write_text(text, encoding="utf-8")
    return len(SEED_DOCS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer-corpus.yaml")
    parser.add_argument("--raw-dir", default=None,
                        help="override: single source dir (replaces config sources)")
    parser.add_argument("--corpus", default=None, help="override output corpus path")
    parser.add_argument("--manifest", default=None, help="override output manifest path")
    parser.add_argument("--name", default=None, help="override corpus name")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    norm = cfg.get("normalization", {})
    norm_kwargs = {
        "unicode_form": norm.get("unicode_form", "NFKC"),
        "collapse_whitespace": norm.get("collapse_whitespace", True),
        "strip_controls": norm.get("strip_controls", True),
        "max_newlines": norm.get("max_newlines", 2),
    }
    seeded = ensure_seed(Path("data/raw"))
    if seeded:
        print(f"seeded {seeded} files into data/raw (offline bootstrap)")

    out = cfg.get("output", {})
    sources = cfg.get("sources", [])
    if args.raw_dir:
        # Halves mode: exactly one file in its own dir (no cross-half mixing).
        sources = [{"path": args.raw_dir, "max_files": 10,
                    "max_bytes_per_file": 1 << 31,
                    "license": "ODC-By-1.0-fineweb"}]
    else:
        seeded = ensure_seed(Path("data/raw"))
        if seeded:
            print(f"seeded {seeded} files into data/raw (offline bootstrap)")

    stats = build_tokenizer_corpus(
        source_dirs=sources,
        corpus_path=args.corpus or out.get("corpus_path", "data/tokenizer_corpus/train.txt"),
        manifest_path=args.manifest or out.get("manifest_path", "data/tokenizer_corpus/manifest.json"),
        norm_kwargs=norm_kwargs,
        min_doc_chars=out.get("min_doc_chars", 20),
        doc_separator=out.get("doc_separator", "\n\n"),
        corpus_name=args.name or cfg.get("name", "myai-tokenizer-corpus-en-v0.1"),
        corpus_version=cfg.get("version", "0.1.0"),
        eos_text=out.get("eos_text"),
    )
    print(
        "corpus built: files_read={files_read} docs_kept={docs_kept} "
        "duplicates={duplicates_removed} chars={total_chars} sha={sha}...".format(
            sha=stats["corpus_sha256"][:16], **stats
        )
    )


if __name__ == "__main__":
    main()
