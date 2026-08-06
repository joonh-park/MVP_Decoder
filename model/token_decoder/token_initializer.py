import torch
from torch import nn

from model.token_decoder.attention import CrossAttentionBlock


class TokenInitializer(nn.Module):
    def __init__(
        self,
        num_queries: int,
        dim: int,
        num_heads: int,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.query_bank = nn.Parameter(torch.empty(1, num_queries, dim))
        nn.init.normal_(self.query_bank, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            CrossAttentionBlock(dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        z = self.query_bank.expand(evidence.shape[0], -1, -1)
        for layer in self.layers:
            z = layer(z, evidence)
        return self.output_norm(z)
