import math

import torch
import torch.nn.functional as F
from torch import nn

from model.token_decoder.types import GaussianParams


class RayAnchorSelector(nn.Module):
    """Select a soft GT-pose ray anchor for every learned 3D token."""

    def __init__(
        self,
        dim: int,
        anchor_dim: int = 64,
        query_chunk_size: int = 256,
        temperature: float = 0.1,
    ):
        super().__init__()
        if anchor_dim <= 0:
            raise ValueError("anchor_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.scale = anchor_dim**-0.5 / temperature
        self.query_chunk_size = query_chunk_size
        self.query_norm = nn.LayerNorm(dim)
        self.evidence_norm = nn.LayerNorm(dim)
        self.query_proj = nn.Linear(dim, anchor_dim, bias=False)
        self.key_proj = nn.Linear(dim, anchor_dim, bias=False)
        with torch.no_grad():
            self.key_proj.weight.copy_(self.query_proj.weight)

    def forward(
        self,
        z: torch.Tensor,
        evidence: torch.Tensor,
        center_ray: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if center_ray.shape[-1] != 9:
            raise ValueError(f"center_ray must have 9 channels, got {center_ray.shape}")
        rays = center_ray.flatten(1, -2).float()
        if evidence.shape[:2] != rays.shape[:2]:
            raise ValueError(
                "Evidence and GT-pose ray counts must match, got "
                f"{evidence.shape} and {rays.shape}"
            )

        query = self.query_proj(self.query_norm(z)).float()
        key = self.key_proj(self.evidence_norm(evidence)).float()
        key_t = key.transpose(-1, -2)
        origins = rays[..., :3]
        directions = rays[..., 3:6]

        chunk_size = self.query_chunk_size
        if chunk_size <= 0:
            chunk_size = query.shape[1]
        selected_origins = []
        selected_directions = []
        for query_chunk in query.split(chunk_size, dim=1):
            weights = torch.softmax(
                torch.matmul(query_chunk, key_t) * self.scale,
                dim=-1,
            )
            selected_origins.append(torch.matmul(weights, origins))
            selected_directions.append(torch.matmul(weights, directions))

        ray_origin = torch.cat(selected_origins, dim=1)
        ray_direction = F.normalize(
            torch.cat(selected_directions, dim=1),
            p=2,
            dim=-1,
            eps=1e-8,
        )
        return ray_origin, ray_direction


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
        position_mode: str = "free",
        depth_min: float = 0.01,
        depth_max: float = 500.0,
        depth_init: float = 2.0,
        anchor_dim: int = 64,
        anchor_query_chunk_size: int = 256,
        anchor_temperature: float = 0.1,
    ):
        super().__init__()
        if position_mode not in {"free", "gt_ray"}:
            raise ValueError(
                f"Unknown position_mode '{position_mode}'; expected 'free' or 'gt_ray'"
            )
        if not depth_min < depth_init < depth_max:
            raise ValueError(
                "GT-ray depth bounds must satisfy "
                f"depth_min < depth_init < depth_max, got "
                f"{depth_min}, {depth_init}, {depth_max}"
            )
        self.sh_degree = sh_degree
        self.position_range = position_range
        self.scale_bias = scale_bias
        self.scale_max = scale_max
        self.opacity_bias = opacity_bias
        self.position_mode = position_mode
        self.depth_min = depth_min
        self.depth_max = depth_max
        initial_probability = (depth_init - depth_min) / (depth_max - depth_min)
        self.depth_logit_bias = math.log(
            initial_probability / (1.0 - initial_probability)
        )
        self.ray_selector = (
            RayAnchorSelector(
                dim=dim,
                anchor_dim=anchor_dim,
                query_chunk_size=anchor_query_chunk_size,
                temperature=anchor_temperature,
            )
            if position_mode == "gt_ray"
            else None
        )
        self.color_dim = 3 * (sh_degree + 1) ** 2
        output_dim = 3 + self.color_dim + 3 + 4 + 1
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, output_dim)

    def forward(
        self,
        z: torch.Tensor,
        evidence: torch.Tensor | None = None,
        center_ray: torch.Tensor | None = None,
    ) -> GaussianParams:
        raw = self.proj(self.norm(z)).float()
        position, feature, scale, rotation, opacity = torch.split(
            raw,
            [3, self.color_dim, 3, 4, 1],
            dim=-1,
        )
        if self.position_mode == "gt_ray":
            if evidence is None or center_ray is None:
                raise ValueError(
                    "GT-ray positioning requires decoder evidence and center_ray"
                )
            ray_origin, ray_direction = self.ray_selector(z, evidence, center_ray)
            depth = self.depth_min + torch.sigmoid(
                position.mean(dim=-1, keepdim=True) + self.depth_logit_bias
            ) * (self.depth_max - self.depth_min)
            xyz = ray_origin + ray_direction * depth
        else:
            xyz = torch.tanh(position) * self.position_range
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
