from dataclasses import dataclass

import torch


@dataclass
class GaussianParams:
    xyz: torch.Tensor
    feature: torch.Tensor
    scale: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor

    @property
    def num_gaussians(self) -> int:
        return self.xyz.shape[-2]

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "xyz": self.xyz,
            "feature": self.feature,
            "scale": self.scale,
            "rotation": self.rotation,
            "opacity": self.opacity,
        }


@dataclass
class TokenDecoderOutput:
    z_initial: torch.Tensor
    gaussians_initial: GaussianParams
    render_initial: torch.Tensor | None = None
    z_final: torch.Tensor | None = None
    gaussians_final: GaussianParams | None = None
    render_final: torch.Tensor | None = None
    input_error: torch.Tensor | None = None
    split_scores: torch.Tensor | None = None
    split_target: torch.Tensor | None = None


@dataclass
class SplitOutput:
    latent: torch.Tensor
    scores: torch.Tensor
    parent_index: torch.Tensor
