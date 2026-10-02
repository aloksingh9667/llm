"""Step 3a quality gates: normalization determinism + corpus manifest."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myai.tokenizer.corpus import build_tokenizer_corpus
from myai.tokenizer.normalization import normalize_text


def test_normalization_deterministic():
    raw = "Hello,\u00a0\u00a0world!\x00\n\n\nNext  line.\n"
    a = normalize_text(raw)
    b = normalize_text(raw)
    assert a == b
    assert "\x00" not in a
    assert "  " not in a.replace("\n\n", "")
    assert a.startswith("Hello, world!")


def test_normalization_keeps_code_indent():
    raw = "def f():\n    return 1\n"
    out = normalize_text(raw)
    assert "    return 1" in out


def test_multidoc_txt_split_and_empty_guard(tmp_path):
    import pytest

    from myai.tokenizer.corpus import build_tokenizer_corpus

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    doc = "This streamed dump holds several documents in one text file for testing."
    (raw_dir / "dump.txt").write_text(f"{doc}\n\n{doc}\n\nA third distinct document here.", encoding="utf-8")
    (raw_dir / "code.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n", encoding="utf-8")
    stats = build_tokenizer_corpus(
        source_dirs=[{"path": str(raw_dir), "max_files": 10, "max_bytes_per_file": 1 << 26}],
        corpus_path=str(tmp_path / "train.txt"),
        manifest_path=str(tmp_path / "manifest.json"),
        norm_kwargs={},
        min_doc_chars=20,
    )
    # dump.txt -> 2 unique docs (one duplicate removed); code.py stays whole.
    assert stats["docs_kept"] == 3
    assert stats["duplicates_removed"] == 1

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no documents survived"):
        build_tokenizer_corpus(
            source_dirs=[{"path": str(empty), "max_files": 10, "max_bytes_per_file": 1 << 20}],
            corpus_path=str(tmp_path / "t2.txt"),
            manifest_path=str(tmp_path / "m2.json"),
            norm_kwargs={},
        )


def test_corpus_dedup_and_manifest(tmp_path):
    raw_dir = tmp_path / "raw2"
    raw_dir.mkdir()
    (raw_dir / "a.txt").write_text("Hello world, this is a test document.", encoding="utf-8")
    (raw_dir / "b.txt").write_text("Hello world, this is a test document.", encoding="utf-8")
    (raw_dir / "c.txt").write_text("short", encoding="utf-8")

    stats = build_tokenizer_corpus(
        source_dirs=[{"path": str(raw_dir), "max_files": 10, "max_bytes_per_file": 1 << 20}],
        corpus_path=str(tmp_path / "train.txt"),
        manifest_path=str(tmp_path / "manifest.json"),
        norm_kwargs={},
        min_doc_chars=20,
    )
    assert stats["docs_kept"] == 1
    assert stats["duplicates_removed"] == 1
    assert stats["too_short_removed"] == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for key in ("name", "version", "source", "license", "provenance", "filters", "corpus_sha256"):
        assert key in manifest
