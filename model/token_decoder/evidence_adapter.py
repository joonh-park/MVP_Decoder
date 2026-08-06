import torch
from torch import nn


class EvidenceAdapter(nn.Module):
    """Fuse frozen appearance features with explicit patch-center geometry."""

    def __init__(self, feature_dim: int, token_dim: int):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, token_dim, bias=False)
        self.ray_encoder = nn.Sequential(
            nn.Linear(9, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim, bias=False),
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, feature: torch.Tensor, center_ray: torch.Tensor) -> torch.Tensor:
        if feature.shape[:-1] != center_ray.shape[:-1]:
            raise ValueError(
                f"Evidence grids do not align: {feature.shape} vs {center_ray.shape}"
            )
        ray_feature = self.ray_encoder(center_ray.to(dtype=feature.dtype))
        evidence = self.norm(self.feature_proj(feature) + ray_feature)
        return evidence.flatten(1, 3)
