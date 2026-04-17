import torch
import torch.nn as nn
from eecs148b_hw1.scaled_dot_product_attention import scaled_dot_product_attention as sdpa
from eecs148b_hw1.linear import Linear

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        # (..., seq_len, d_model) -> (..., num_heads, seq_len, head_dim)
        q = self.q_proj(x).unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)
        k = self.k_proj(x).unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)
        v = self.v_proj(x).unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)

        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attn_out = sdpa(q, k, v, causal)
        merged = attn_out.transpose(-3, -2).contiguous().flatten(-2, -1)
        return self.output_proj(merged)