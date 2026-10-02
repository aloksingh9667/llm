"""Step 18 quality gates: prompt masking correctness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from myai.training.sft import IGNORE, build_sft_pair, pack_sft_pairs, sft_loss


def test_prompt_masked_response_kept():
    inp, lab = build_sft_pair([1, 2, 3], [4, 5], eos_id=0)
    assert inp == [1, 2, 3, 4, 5, 0]
    assert lab == [IGNORE, IGNORE, IGNORE, 4, 5, 0]


def test_loss_ignores_prompt():
    torch.manual_seed(0)
    vocab = 10
    logits = torch.randn(1, 6, vocab)
    _, lab = build_sft_pair([1, 2, 3], [4, 5], eos_id=0)
    labels = torch.tensor([lab])
    got = sft_loss(logits, labels)
    import torch.nn.functional as F

    want = F.cross_entropy(logits.view(-1, vocab)[3:], labels.view(-1)[3:])
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    # Corrupting prompt logits must not change the loss.
    bad = logits.clone()
    bad[:, :3, :] += 100.0
    torch.testing.assert_close(sft_loss(bad, labels), got, rtol=1e-5, atol=1e-6)


def test_packing_keeps_masks_aligned():
    pairs = [build_sft_pair([1, 2], [3], eos_id=0), build_sft_pair([4], [5, 6], eos_id=0)]
    blocks = pack_sft_pairs(pairs, seq_len=4)
    assert len(blocks) == 2
    (b0_in, b0_lab), (b1_in, b1_lab) = blocks
    assert b0_in == [1, 2, 3, 0] and b0_lab == [IGNORE, IGNORE, 3, 0]
    for inp, lab in blocks:
        assert len(inp) == len(lab) == 4
