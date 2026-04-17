import torch
import torch.nn as nn
from eecs148b_hw1.linear import Linear

class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.fc2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.fc1(x)
        relu_out = torch.where(out1 > 0, out1, 0)
        out2 = self.fc2(relu_out)
        return out2