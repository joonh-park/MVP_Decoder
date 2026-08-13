import torch
from torch import nn

from model.token_decoder.attention import TokenLayerStack


class TokenInitializer(nn.Module):
    def __init__(
        self,
        num_queries: int,
        dim: int,
        num_heads: int,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        layer_specs=None,
        dropout: float = 0.0,
        query_chunk_size: int = 0,
        evidence_chunk_size: int = 1024,
        slot_epsilon: float = 1e-8,
        slot_null: bool = True,
    ):
        super().__init__()
        self.query_bank = nn.Parameter(torch.empty(1, num_queries, dim))
        nn.init.normal_(self.query_bank, mean=0.0, std=0.02)
        if layer_specs is None:
            layer_specs = ["cross"] * num_layers
        self.stack = TokenLayerStack(
            dim=dim,
            layer_specs=layer_specs,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            query_chunk_size=query_chunk_size,
            evidence_chunk_size=evidence_chunk_size,
            slot_epsilon=slot_epsilon,
            slot_null=slot_null,
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        z = self.query_bank.expand(evidence.shape[0], -1, -1)
        return self.stack(z, evidence)
