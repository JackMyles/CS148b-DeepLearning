"""
Autoregressive decoding with temperature scaling and top-p (nucleus) sampling.
"""

import torch

from eecs148b_hw1.softmax import softmax


@torch.no_grad()
def decode(
    model: torch.nn.Module,
    prompt: list[int],
    max_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> list[int]:
    """
    Generate a token sequence from a language model given a prompt.

    Args:
        model:        A TransformerLM that maps (1, seq_len) int ids → (1, seq_len, vocab) logits.
        prompt:       List of integer token IDs for the initial context.
        max_tokens:   Maximum number of new tokens to generate.
        temperature:  Softmax temperature (τ). Lower → sharper; 0 → greedy argmax.
        top_p:        Nucleus sampling threshold in (0, 1]. 1.0 disables it.
        eos_token_id: If not None, stop generating when this token is produced.

    Returns:
        The full token sequence (prompt + generated tokens).
    """
    device = next(model.parameters()).device
    context_length = getattr(model, "context_length", None)
    tokens = list(prompt)

    model.eval()
    for _ in range(max_tokens):
        # Truncate to context window if the model has one
        if context_length is not None:
            input_ids = tokens[-context_length:]
        else:
            input_ids = tokens
        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        logits = model(x)                       # (1, seq_len, vocab)
        next_logits = logits[0, -1, :]          # (vocab,)

        # --- temperature scaling ---
        if temperature == 0.0:
            next_id = int(next_logits.argmax())
        else:
            scaled = next_logits / temperature
            probs = softmax(scaled, dim=-1)     # (vocab,)

            # --- top-p (nucleus) sampling ---
            if top_p < 1.0:
                probs = _nucleus(probs, top_p)

            next_id = int(torch.multinomial(probs, num_samples=1))

        tokens.append(next_id)
        if eos_token_id is not None and next_id == eos_token_id:
            break

    return tokens


def _nucleus(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out tokens outside the smallest set whose cumulative prob >= p, then renormalize."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Mask: True for tokens *beyond* the nucleus (cumulative already >= p before this token)
    mask = (cumulative - sorted_probs) >= p
    sorted_probs[mask] = 0.0

    # Scatter back to original ordering and renormalize
    probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    return probs / probs.sum()
