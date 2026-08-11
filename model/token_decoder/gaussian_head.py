import torch
import torch.nn.functional as F
from torch import nn

from model.token_decoder.types import GaussianParams


class SharedGaussianHead(nn.Module):
    """Shared latent readout used before and after densification/refinement."""

    def __init__(
        self,
        dim: int,
        sh_degree: int,
        position_range: float,
        scale_bias: float,
        scale_max: float,
        opacity_bias: float,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.position_range = position_range
        self.scale_bias = scale_bias
        self.scale_max = scale_max
        self.opacity_bias = opacity_bias
        self.color_dim = 3 * (sh_degree + 1) ** 2
        output_dim = 3 + self.color_dim + 3 + 4 + 1
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, output_dim)

    def forward(self, z: torch.Tensor) -> GaussianParams:
        raw = self.proj(self.norm(z)).float()
        xyz, feature, scale, rotation, opacity = torch.split(
            raw,
            [3, self.color_dim, 3, 4, 1],
            dim=-1,
        )
        xyz = torch.tanh(xyz) * self.position_range
        scale = (scale + self.scale_bias).clamp(max=self.scale_max)

        identity = rotation.new_tensor([1.0, 0.0, 0.0, 0.0])
        rotation = F.normalize(rotation + identity, p=2, dim=-1)
        opacity = torch.sigmoid(opacity + self.opacity_bias)

        feature = feature.view(
            *feature.shape[:-1], (self.sh_degree + 1) ** 2, 3
        ).contiguous()
        return GaussianParams(
            xyz=xyz,
            feature=feature,
            scale=scale,
            rotation=rotation,
            opacity=opacity,
        )
