"""P0 pipeline gates: EOS boundaries, hash split, sharded round-trip."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from myai.data.sharded import load_sharded, verify_index, write_index, write_shards
from myai.tokenizer.bpe import BPETokenizer
from scripts.prepare_sequences import hash_split, split_docs


def test_eos_becomes_boundary_id():
    tok = BPETokenizer()
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("hello world\nfoo bar\n")
        corpus = f.name
    try:
        tok.train(corpus, vocab_size=300)
    finally:
        os.unlink(corpus)
    eos_id = tok.token_to_id[b"<|endoftext|>"]
    ids = tok.encode("hello world<|endoftext|>foo bar<|endoftext|>")
    assert ids.count(eos_id) == 2
    assert tok.decode(ids) == "hello world<|endoftext|>foo bar<|endoftext|>"


def test_split_docs_on_eos():
    text = "doc one\n<|endoftext|>\ndoc two\n<|endoftext|>\n"
    assert split_docs(text, "<|endoftext|>") == ["doc one", "doc two"]
    assert split_docs("a\n\nb", None) == ["a", "b"]


def test_hash_split_no_leakage_and_deterministic():
    docs = [f"document number {i} with distinct content here" for i in range(50)]
    docs += [docs[0], docs[1]]  # exact dupes must share a split
    t1, v1 = hash_split(docs, 0.2)
    t2, v2 = hash_split(list(reversed(docs)), 0.2)
    assert sorted(t1) == sorted(t2) and sorted(v1) == sorted(v2), "order-independent"
    assert not (set(t1) & set(v1)), "no document in both splits"


def test_sharded_roundtrip_and_tamper(tmp_path):
    pairs = [([i, i + 1], [i + 1, i + 2]) for i in range(10)]
    e = write_shards(pairs, tmp_path, "train", rows_per_shard=4)
    write_index(tmp_path, [e], {"seq_len": 2})
    assert [s["rows"] for s in e["shards"]] == [4, 4, 2]
    ds, entry = load_sharded(tmp_path, "train")
    assert len(ds) == 10
    assert ds[0]["input_ids"].tolist() == [0, 1]
    assert all(verify_index(tmp_path).values())
    # Tamper with a shard -> verification fails.
    blob = torch.load(tmp_path / "train" / "shard-00000.pt", weights_only=True)
    blob["input_ids"][0, 0] += 99
    torch.save(blob, tmp_path / "train" / "shard-00000.pt")
    assert not verify_index(tmp_path)["train/shard-00000.pt"]
