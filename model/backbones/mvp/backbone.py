from contextlib import nullcontext
from pathlib import Path

import torch

from model.backbones.base import EvidenceBackbone
from model.backbones.evidence_output import EvidenceOutput
from model.backbones.mvp.model import MVPModel
from model.backbones.mvp.utils import compute_plucmap


def _clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("_checkpoint_wrapped_module.", "")
        key = key.replace("_orig_mod.", "")
        while key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def compute_patch_center_plucker(
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
) -> torch.Tensor:
    """Compute one exact center ray for every DPT patch cell."""

    image_h, image_w = image_size
    grid_h, grid_w = grid_size
    scaled_intrinsics = intrinsics.clone()
    scaled_intrinsics[..., 0] *= grid_w / image_w
    scaled_intrinsics[..., 1] *= grid_h / image_h
    scaled_intrinsics[..., 2] *= grid_w / image_w
    scaled_intrinsics[..., 3] *= grid_h / image_h

    ray_o, ray_d = compute_plucmap(
        scaled_intrinsics.float(), c2w.float(), grid_h, grid_w
    )
    moment = torch.cross(ray_o, ray_d, dim=2)
    return torch.cat([ray_o, ray_d, moment], dim=2).permute(0, 1, 3, 4, 2).contiguous()


class MVPBackbone(EvidenceBackbone):
    """MVP through DPT/PFA exposed through the evidence-backbone interface."""

    def __init__(
        self,
        backbone_config,
        checkpoint_path: str | None = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.freeze = freeze
        self.mvp = MVPModel(backbone_config)
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        if self.freeze:
            self.requires_grad_(False)
            self.train(False)

    @property
    def output_dim(self) -> int:
        return self.mvp.dim3

    def load_checkpoint(self, checkpoint_path: str):
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"MVP checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("ema", checkpoint.get("model", checkpoint))
        return self.mvp.load_state_dict(_clean_state_dict(state_dict), strict=False)

    def train(self, mode: bool = True):
        return super().train(False if self.freeze else mode)

    def forward(self, images, intrinsics, c2w) -> EvidenceOutput:
        context = torch.no_grad() if self.freeze else nullcontext()
        with context:
            feature = self.mvp.encode_dpt_features(
                {"image": images, "fxfycxcy": intrinsics, "c2w": c2w}
            )
            grid_size = (feature.shape[2], feature.shape[3])
            center_ray = compute_patch_center_plucker(
                intrinsics,
                c2w,
                image_size=(images.shape[-2], images.shape[-1]),
                grid_size=grid_size,
            )
        if self.freeze:
            feature = feature.detach()
            center_ray = center_ray.detach()
            c2w = c2w.detach()
            intrinsics = intrinsics.detach()
        return EvidenceOutput(
            feature=feature,
            center_ray=center_ray,
            grid_size=grid_size,
            input_c2w=c2w,
            input_intrinsics=intrinsics,
        )
