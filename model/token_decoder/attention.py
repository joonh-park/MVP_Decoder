import torch
import torch.nn.functional as F
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        q = self._heads(self.q_proj(query))
        k = self._heads(self.k_proj(evidence))
        v = self._heads(self.v_proj(evidence))
        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).contiguous().flatten(2)
        return self.out_proj(output)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention Transformer block with internal residuals."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.query_norm = nn.LayerNorm(dim)
        self.evidence_norm = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        z = z + self.cross_attn(self.query_norm(z), self.evidence_norm(evidence))
        z = z + self.ffn(self.ffn_norm(z))
        return z
