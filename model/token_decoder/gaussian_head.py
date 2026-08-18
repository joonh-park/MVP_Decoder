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
        position_anchor: tuple[float, float, float] | list[float],
        scale_weight: float,
        opacity_bias: float,
        opacity_mapping_initial: float,
        opacity_mapping_final: float,
        opacity_mapping_warm_up: int,
    ):
        super().__init__()
        if len(position_anchor) != 3:
            raise ValueError(
                f"position_anchor must contain 3 values, got {position_anchor}"
            )
        if scale_weight <= 0.0:
            raise ValueError("scale_weight must be positive")
        if opacity_mapping_warm_up < 0:
            raise ValueError("opacity_mapping_warm_up must be non-negative")
        self.sh_degree = sh_degree
        self.register_buffer(
            "position_anchor",
            torch.tensor(position_anchor, dtype=torch.float32),
            persistent=False,
        )
        self.scale_weight = scale_weight
        self.opacity_bias = opacity_bias
        self.opacity_mapping_initial = opacity_mapping_initial
        self.opacity_mapping_final = opacity_mapping_final
        self.opacity_mapping_warm_up = opacity_mapping_warm_up
        self.color_dim = 3 * (sh_degree + 1) ** 2
        output_dim = 3 + self.color_dim + 3 + 4 + 1
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, output_dim)

    def _map_opacity(self, probability: torch.Tensor, global_step: int) -> torch.Tensor:
        if self.opacity_mapping_warm_up == 0:
            progress = 1.0
        else:
            progress = min(max(global_step, 0) / self.opacity_mapping_warm_up, 1.0)
        mapping = self.opacity_mapping_initial + progress * (
            self.opacity_mapping_final - self.opacity_mapping_initial
        )
        exponent = 2.0**mapping
        return 0.5 * (
            1.0
            - (1.0 - probability).pow(exponent)
            + probability.pow(1.0 / exponent)
        )

    def forward(self, z: torch.Tensor, global_step: int = 0) -> GaussianParams:
        raw = self.proj(self.norm(z)).float()
        xyz, feature, scale, rotation, opacity = torch.split(
            raw,
            [3, self.color_dim, 3, 4, 1],
            dim=-1,
        )
        xyz = xyz + self.position_anchor.to(device=xyz.device, dtype=xyz.dtype)
        scale = self.scale_weight * F.softplus(scale)
        scale = scale.log()

        identity = rotation.new_tensor([1.0, 0.0, 0.0, 0.0])
        rotation = F.normalize(rotation + identity, p=2, dim=-1)
        opacity = self._map_opacity(
            torch.sigmoid(opacity + self.opacity_bias),
            global_step,
        )

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
