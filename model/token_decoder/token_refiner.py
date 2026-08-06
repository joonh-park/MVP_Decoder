from torch import nn

from model.token_decoder.attention import CrossAttentionBlock


class TokenRefiner(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            CrossAttentionBlock(dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, z, evidence):
        for layer in self.layers:
            z = layer(z, evidence)
        return self.output_norm(z)
