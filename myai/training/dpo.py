"""DPO preference loss from scratch (Phase 1, audit Stage 5).

Given policy logps and reference logps for chosen/rejected responses:

    loss = -log sigmoid(beta * ((lp_c - lp_r) - (ref_c - ref_r)))

`sequence_logps` sums token log-probs over unmasked positions so prompt
tokens never count. Reference logps come from the frozen base model.
"""
import torch
import torch.nn.functional as F


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor,
                   ignore_index: int = -100) -> torch.Tensor:
    """Sum log-prob of target tokens per sequence, skipping masked spots."""
    logp = F.log_softmax(logits.float(), dim=-1)
    tok = labels.clamp_min(0).unsqueeze(-1)
    gathered = logp.gather(-1, tok).squeeze(-1)
    mask = (labels != ignore_index).float()
    return (gathered * mask).sum(-1)


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float = 0.1) -> torch.Tensor:
    """Mean DPO loss over the batch. Higher chosen margin -> lower loss."""
    margins = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return (-F.logsigmoid(beta * margins)).mean()
