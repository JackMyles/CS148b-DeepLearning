import torch
import torch.nn as nn

from eecs148b_hw1.embedding import Embedding
from eecs148b_hw1.layernorm import LayerNorm
from eecs148b_hw1.linear import Linear
from eecs148b_hw1.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from eecs148b_hw1.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """Pre-norm Transformer LM: token + sinusoidal PE → blocks → LayerNorm → LM head."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        no_layernorm: bool = False,
        no_pos_emb: bool = False,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.no_pos_emb = no_pos_emb

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.pos_encoding = None if no_pos_emb else SinusoidalPositionalEmbedding(
            d_model, context_length, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype, no_layernorm=no_layernorm)
            for _ in range(num_layers)
        )
        self.ln_final = nn.Identity() if no_layernorm else LayerNorm(d_model, device=device, dtype=dtype)
        # Maps each position's hidden state to vocab logits (weight shape vocab × d_model).
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # (batch, seq) → (batch, seq, d_model)
        x = self.token_embeddings(token_ids)

        if self.pos_encoding is not None:
            _, seq_len = token_ids.shape
            positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long).unsqueeze(0)
            positions = positions.expand(token_ids.shape[0], -1)
            x = x + self.pos_encoding(positions)

        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)
