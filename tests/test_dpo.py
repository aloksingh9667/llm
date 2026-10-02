"""Phase 1 gates: DPO loss behavior + SFT training smoke."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from myai.training.dpo import dpo_loss, sequence_logps


def test_sequence_logps_skips_masked():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 8)
    labels = torch.tensor([[1, 2, -100, -100, 3], [4, -100, -100, -100, -100]])
    got = sequence_logps(logits, labels)
    logp = F.log_softmax(logits.float(), -1)
    want0 = logp[0, 0, 1] + logp[0, 1, 2] + logp[0, 4, 3]
    want1 = logp[1, 0, 4]
    torch.testing.assert_close(got[0], want0, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got[1], want1, rtol=1e-5, atol=1e-6)


def test_dpo_prefers_chosen_margin():
    # Large chosen margin -> near-zero loss; inverted margin -> large loss.
    good = dpo_loss(torch.tensor([32.0]), torch.tensor([2.0]),
                    torch.tensor([2.0]), torch.tensor([2.0]))
    bad = dpo_loss(torch.tensor([2.0]), torch.tensor([32.0]),
                   torch.tensor([2.0]), torch.tensor([2.0]))
    assert good.item() < 0.1 and bad.item() > 1.0
    # Reference cancels out when policy matches it: loss = log(2).
    tied = dpo_loss(torch.tensor([3.0]), torch.tensor([1.0]),
                    torch.tensor([3.0]), torch.tensor([1.0]))
    torch.testing.assert_close(tied, torch.tensor(0.6931), rtol=1e-3, atol=1e-3)
