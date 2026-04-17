import torch
import torch.nn as nn
from eecs148b_hw1.multihead_self_attention import MultiheadSelfAttention
from eecs148b_hw1.positionwise_feedforward import PositionwiseFeedForward
from eecs148b_hw1.layernorm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        no_layernorm: bool = False,
    ) -> None:
        super().__init__()
        
        norm = (lambda: nn.Identity()) if no_layernorm else (lambda: LayerNorm(d_model, device=device, dtype=dtype))
        self.ln1 = norm()
        self.attn = MultiheadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = norm()
        self.ffn = PositionwiseFeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x