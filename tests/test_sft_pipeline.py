"""Phase 1 gates: OASST pairing, SFT masking end-to-end, ignore-index loss."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from myai.tokenizer.bpe import BPETokenizer
from myai.training.sft import IGNORE, sft_loss
from scripts.fetch_sft import pair_oasst_rows
from scripts.pack_sft import format_pair


def test_oasst_pairing_joins_parent_prompt():
    rows = [
        {"message_id": "m1", "parent_id": None, "role": "prompter", "lang": "en", "text": "Add two numbers in Python"},
        {"message_id": "m2", "parent_id": "m1", "role": "assistant", "lang": "en", "text": "def add(a, b): return a + b"},
        {"message_id": "m3", "parent_id": "mX", "role": "assistant", "lang": "en", "text": "orphan, no parent"},
        {"message_id": "m4", "parent_id": None, "role": "prompter", "lang": "de", "text": "Deutsch wird ignoriert"},
    ]
    pairs = pair_oasst_rows(rows, max_pairs=10)
    assert len(pairs) == 1
    assert pairs[0]["instruction"] == "Add two numbers in Python"
    assert pairs[0]["output"] == "def add(a, b): return a + b"


def test_sft_end_to_end_masked_loss(tmp_path):
    tok = BPETokenizer().train("data/tokenizer_corpus/train.txt", vocab_size=300)
    prompt, response = format_pair({"instruction": "Add.", "input": "", "output": "1+1=2"})
    from myai.training.sft import build_sft_pair

    eos_id = tok.encode("<|endoftext|>")[0]
    inp, lab = build_sft_pair(tok.encode(prompt), tok.encode(response), eos_id)
    assert lab[: len(tok.encode(prompt))] == [IGNORE] * len(tok.encode(prompt))
    assert lab[-1] == eos_id
    # Masked positions truly ignored: randomizing prompt logits is a no-op.
    torch.manual_seed(0)
    logits = torch.randn(1, len(inp), len(tok))
    base = sft_loss(logits, torch.tensor([lab]))
    noisy = logits.clone()
    noisy[:, : len(tok.encode(prompt)), :] += 50.0
    torch.testing.assert_close(sft_loss(noisy, torch.tensor([lab])), base, rtol=1e-5, atol=1e-6)
