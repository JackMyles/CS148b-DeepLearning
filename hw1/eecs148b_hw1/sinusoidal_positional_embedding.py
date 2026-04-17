import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        pe_dtype = dtype or torch.float32
        pos = torch.arange(max_seq_len, device=device, dtype=pe_dtype).unsqueeze(1)
        # Angle for pair (2i, 2i+1): p / 10000^(2i/d_model)
        div = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=pe_dtype)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_len, d_model, device=device, dtype=pe_dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, token_positions: torch.Tensor) -> torch.Tensor:
        return self.pe[token_positions]