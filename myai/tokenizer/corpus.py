"""Corpus builder: local files -> normalized train.txt + manifest (Step 3a).

Implements the doc section 9 rule: every source gets a manifest with
name/version/source/license/provenance/filters, plus SHA-256 dedup.
Stdlib only so the base env stays lean.
"""
import datetime
import hashlib
import json
from pathlib import Path

from .normalization import normalize_text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_files(source_dir: str, max_files: int, max_bytes_per_file: int) -> list[Path]:
    root = Path(source_dir)
    if not root.exists():
        return []
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep"
        and p.suffix.lower() in {".txt", ".md", ".py"}
    )
    out = []
    for p in files:
        if len(out) >= max_files:
            break
        try:
            if p.stat().st_size <= max_bytes_per_file:
                out.append(p)
        except OSError:
            continue
    return out


def build_tokenizer_corpus(
    source_dirs: list[dict],
    corpus_path: str,
    manifest_path: str,
    norm_kwargs: dict,
    min_doc_chars: int = 20,
    doc_separator: str = "\n\n",
    corpus_name: str = "myai-tokenizer-corpus-en-v0.1",
    corpus_version: str = "0.1.0",
    eos_text: str | None = None,
) -> dict:
    """Read, normalize, exact-dedup, write corpus + manifest. Returns stats."""
    seen: set[str] = set()
    docs: list[str] = []
    files_read = 0
    duplicates = 0
    too_short = 0

    for src in source_dirs:
        for path in collect_files(
            src["path"], src.get("max_files", 100), src.get("max_bytes_per_file", 1 << 20)
        ):
            try:
                raw = path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeError, OSError):
                continue
            files_read += 1
            # Streamed dumps (.txt/.md) hold many docs separated by blank
            # lines — split so dedup/manifest stay per-document. Code files
            # stay whole (blank lines are not doc boundaries there).
            chunks = (
                raw.split("\n\n") if path.suffix.lower() in {".txt", ".md"} else [raw]
            )
            for chunk in chunks:
                doc = normalize_text(chunk, **norm_kwargs)
                if len(doc) < min_doc_chars:
                    too_short += 1
                    continue
                h = _sha256(doc)
                if h in seen:
                    duplicates += 1
                    continue
                seen.add(h)
                docs.append(doc)

    if not docs:
        raise ValueError(
            f"no documents survived (files_read={files_read}, "
            f"too_short={too_short}) — check source caps and min_doc_chars"
        )

    # Explicit document boundaries (audit P0): each doc ends with the EOS
    # marker, which doubles as the split point for downstream packing.
    # The marker must survive normalization, so it is appended after.
    boundary = f"\n{eos_text}\n" if eos_text else doc_separator
    corpus_file = Path(corpus_path)
    corpus_file.parent.mkdir(parents=True, exist_ok=True)
    body = boundary.join(docs)
    if eos_text:
        body += boundary
    else:
        body += "\n"
    corpus_file.write_text(body, encoding="utf-8")

    corpus_hash = hashlib.sha256(corpus_file.read_bytes()).hexdigest()
    total_chars = sum(len(d) for d in docs)
    stats = {
        "name": corpus_name,
        "version": corpus_version,
        "source": [s["path"] for s in source_dirs],
        "license": "seed-internal",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "allowed_use": "research/bootstrap only",
        "provenance": "local seed files, normalized + sha256 exact-dedup",
        "filters": ["unicode_NFKC", "control_strip", "whitespace_collapse", "exact_dedup_sha256"],
        "eos_text": eos_text,
        "files_read": files_read,
        "docs_kept": len(docs),
        "duplicates_removed": duplicates,
        "too_short_removed": too_short,
        "total_chars": total_chars,
        "corpus_sha256": corpus_hash,
    }
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats
