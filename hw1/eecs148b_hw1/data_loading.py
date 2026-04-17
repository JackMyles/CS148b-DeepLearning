import numpy as np
import torch


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random length-m windows from a 1D token stream: inputs x[s:s+m],
    next-token targets x[s+1:s+m+1]. Works with ordinary arrays and np.memmap.
    """
    n = x.shape[0]
    # Valid start indices s satisfy s + context_length < n (need one more token for targets)
    hi = n - context_length
    starts = np.random.randint(0, hi, size=batch_size)

    # Row r uses start starts[r]; columns add 0..m-1
    offsets = np.arange(context_length, dtype=np.int64)
    idx = starts[:, None] + offsets[None, :]

    inputs_np = x[idx]
    targets_np = x[idx + 1]

    inputs = torch.as_tensor(inputs_np, dtype=torch.long, device=device)
    targets = torch.as_tensor(targets_np, dtype=torch.long, device=device)
    return inputs, targets
