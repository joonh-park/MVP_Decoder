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


class FrozenMVPBackbone(EvidenceBackbone):
    """MVP through DPT/PFA, permanently frozen and detached."""

    def __init__(self, config, checkpoint_path: str | None = None):
        super().__init__()
        self.mvp = MVPModel(config)
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        self.requires_grad_(False)
        self.eval()

    def load_checkpoint(self, checkpoint_path: str):
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"MVP checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("ema", checkpoint.get("model", checkpoint))
        return self.mvp.load_state_dict(_clean_state_dict(state_dict), strict=False)

    def train(self, mode: bool = True):
        super().train(False)
        self.mvp.eval()
        return self

    @torch.no_grad()
    def forward(self, images, intrinsics, c2w) -> EvidenceOutput:
        feature = self.mvp.encode_dpt_features(
            {"image": images, "fxfycxcy": intrinsics, "c2w": c2w}
        ).detach()
        grid_size = (feature.shape[2], feature.shape[3])
        center_ray = compute_patch_center_plucker(
            intrinsics,
            c2w,
            image_size=(images.shape[-2], images.shape[-1]),
            grid_size=grid_size,
        ).detach()
        return EvidenceOutput(
            feature=feature,
            center_ray=center_ray,
            grid_size=grid_size,
            input_c2w=c2w.detach(),
            input_intrinsics=intrinsics.detach(),
        )
