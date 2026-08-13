import torch
from torch import nn


class EvidenceAdapter(nn.Module):
    """Fuse frozen appearance features with explicit patch-center geometry."""

    def __init__(self, feature_dim: int, token_dim: int, geometry_dim: int = 128):
        super().__init__()
        if geometry_dim <= 0:
            raise ValueError("geometry_dim must be positive")
        self.feature_proj = nn.Linear(feature_dim, token_dim, bias=False)
        self.appearance_norm = nn.LayerNorm(token_dim)
        self.ray_encoder = nn.Sequential(
            nn.Linear(9, geometry_dim),
            nn.GELU(),
            nn.Linear(geometry_dim, geometry_dim, bias=False),
        )
        self.geometry_norm = nn.LayerNorm(geometry_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(token_dim + geometry_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, feature: torch.Tensor, center_ray: torch.Tensor) -> torch.Tensor:
        if feature.shape[:-1] != center_ray.shape[:-1]:
            raise ValueError(
                f"Evidence grids do not align: {feature.shape} vs {center_ray.shape}"
            )
        appearance = self.appearance_norm(self.feature_proj(feature))
        geometry = self.geometry_norm(
            self.ray_encoder(center_ray.to(dtype=feature.dtype))
        )
        fusion_delta = self.fusion_mlp(torch.cat((appearance, geometry), dim=-1))
        evidence = self.norm(appearance + fusion_delta)
        return evidence.flatten(1, 3)
