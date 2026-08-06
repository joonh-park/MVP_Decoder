from dataclasses import dataclass

import torch


@dataclass
class EvidenceOutput:
    """Frozen multi-view evidence consumed by a 3D-token decoder."""

    feature: torch.Tensor
    center_ray: torch.Tensor
    grid_size: tuple[int, int]
    input_c2w: torch.Tensor
    input_intrinsics: torch.Tensor

    def __post_init__(self) -> None:
        if self.feature.ndim != 5:
            raise ValueError(f"feature must be [B,V,H,W,C], got {self.feature.shape}")
        if self.center_ray.shape[:-1] != self.feature.shape[:-1]:
            raise ValueError(
                "center_ray and feature grids must align, got "
                f"{self.center_ray.shape} and {self.feature.shape}"
            )
        if self.center_ray.shape[-1] != 9:
            raise ValueError(f"center_ray must have 9 channels, got {self.center_ray.shape}")

    @property
    def batch_size(self) -> int:
        return self.feature.shape[0]

    @property
    def num_views(self) -> int:
        return self.feature.shape[1]
