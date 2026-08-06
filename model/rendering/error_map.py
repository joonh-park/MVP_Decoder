import torch
import torch.nn.functional as F
from torch import nn


def compute_input_error(rendering: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    if rendering.shape != image.shape:
        raise ValueError(f"Rendering and image shapes differ: {rendering.shape} vs {image.shape}")
    return (rendering - image).abs().mean(dim=2, keepdim=True)


@torch.no_grad()
def sample_token_error(
    xyz: torch.Tensor,
    error: torch.Tensor,
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Project token centers and average their visible input-view reconstruction error."""

    batch, views, _, height, width = error.shape
    num_tokens = xyz.shape[1]
    w2c = torch.inverse(c2w.float())
    xyz_h = torch.cat([xyz.float(), torch.ones_like(xyz[..., :1])], dim=-1)
    camera_xyz = torch.einsum("bvij,bnj->bvni", w2c, xyz_h)[..., :3]
    depth = camera_xyz[..., 2]
    safe_depth = depth.clamp_min(1e-6)
    pixel_x = intrinsics[..., 0, None] * camera_xyz[..., 0] / safe_depth
    pixel_y = intrinsics[..., 1, None] * camera_xyz[..., 1] / safe_depth
    pixel_x = pixel_x + intrinsics[..., 2, None]
    pixel_y = pixel_y + intrinsics[..., 3, None]

    grid_x = (2.0 * pixel_x + 1.0) / width - 1.0
    grid_y = (2.0 * pixel_y + 1.0) / height - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(batch * views, num_tokens, 1, 2)
    sampled = F.grid_sample(
        error.view(batch * views, 1, height, width).float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).view(batch, views, num_tokens)
    visible = (
        (depth > 0)
        & (grid_x >= -1)
        & (grid_x <= 1)
        & (grid_y >= -1)
        & (grid_y <= 1)
    )
    normalizer = visible.sum(dim=1).clamp_min(1)
    return (sampled * visible).sum(dim=1) / normalizer


class ErrorEvidenceEncoder(nn.Module):
    """Turn an input-view RGB error map into patch-aligned decoder evidence."""

    def __init__(self, token_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim, bias=False),
        )

    def forward(
        self,
        error: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, views, _, height, width = error.shape
        pooled = F.adaptive_avg_pool2d(
            error.view(batch * views, 1, height, width), grid_size
        )
        pooled = pooled.view(batch, views, 1, *grid_size).permute(0, 1, 3, 4, 2)
        return self.proj(pooled).flatten(1, 3)
