import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Mean cross-entropy for (batch, vocab) logits and integer class indices.
    Stable via log-sum-exp: -log softmax(logits)[target] = logsumexp(logits) - logits[target].
    """
    max_logits = logits.max(dim=-1).values
    shifted_logits = logits - max_logits.unsqueeze(-1)
    log_norm = max_logits + torch.log(torch.exp(shifted_logits).sum(dim=-1))
    idx = targets.unsqueeze(-1)
    chosen = logits.gather(-1, idx).squeeze(-1)
    loss = log_norm - chosen
    return loss.mean()
