"""Merged K6 data packs: identical tokenizer, stacked rows, flag guardrails."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def test_merge_packs_stacks_rows_and_validates(tmp_path):
    from scripts.merge_packs import merge_packs

    seq, vocab = 4, 8
    def half(tok_sha, tag):
        d = tmp_path / tag
        d.mkdir()
        torch.save({"input_ids": torch.arange(seq * vocab).reshape(-1, seq) % vocab,
                    "labels": torch.arange(seq * vocab).reshape(-1, seq) % vocab},
                   d / "train.pt")
        torch.save({"input_ids": torch.zeros(2, seq, dtype=torch.long),
                    "labels": torch.zeros(2, seq, dtype=torch.long)}, d / "val.pt")
        (d / "manifest.json").write_text(json.dumps(
            {"tokenizer_sha256": tok_sha, "vocab_size": vocab, "seq_len": seq,
             "eos_text": "<|endoftext|>", "train_docs": 3, "val_docs": 1,
             "total_tokens": seq * vocab}), encoding="utf-8")
        return d

    a, b = half("samesha" + str(vocab), "halfA"), half("samesha" + str(vocab), "halfB")
    out = tmp_path / "merged"
    m = merge_packs([str(a), str(b)], str(out), expect_sha="samesha8")
    rows = torch.load(out / "train.pt", weights_only=True)["input_ids"].shape[0]
    assert rows == 2 * (seq * vocab) // seq
    assert m["train_sequences"] == rows and m["total_tokens_real"] == 2 * seq * vocab
    assert len(m["halves"]) == 2

    disagree = half("OTHER", "halfC")
    try:
        merge_packs([str(disagree)], str(out / "bad"), expect_sha="samesha8")
        raise AssertionError("mismatched tokenizers must be rejected")
    except ValueError as e:
        assert "does not match" in str(e)
