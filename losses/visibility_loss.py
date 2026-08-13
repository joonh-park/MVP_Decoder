import torch
import torch.nn.functional as F
from torch import nn


class VisibilityLoss(nn.Module):
    """TokenGS visibility regularizer for directly regressed Gaussian means."""

    def __init__(self, clip: float = 1.0, epsilon: float = 1.0e-6):
        super().__init__()
        if clip <= 0:
            raise ValueError("visibility clip must be positive")
        if epsilon <= 0:
            raise ValueError("visibility epsilon must be positive")
        self.clip = float(clip)
        self.epsilon = float(epsilon)

    def forward(
        self,
        xyz: torch.Tensor,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must have shape [B, N, 3], got {xyz.shape}")
        if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4):
            raise ValueError(f"c2w must have shape [B, V, 4, 4], got {c2w.shape}")
        if intrinsics.shape != c2w.shape[:2] + (4,):
            raise ValueError(
                "intrinsics must align with c2w and have 4 channels, got "
                f"{intrinsics.shape} and {c2w.shape}"
            )
        if xyz.shape[0] != c2w.shape[0]:
            raise ValueError(
                f"xyz and cameras must share a batch size, got {xyz.shape[0]} "
                f"and {c2w.shape[0]}"
            )

        height, width = image_size
        w2c = torch.linalg.inv(c2w.float())
        camera_xyz = torch.einsum(
            "bvij,bnj->bvni", w2c[..., :3, :3], xyz.float()
        ) + w2c[..., :3, 3].unsqueeze(-2)
        x, y, z = camera_xyz.unbind(dim=-1)
        safe_z = torch.where(
            z >= 0,
            z.clamp_min(self.epsilon),
            z.clamp_max(-self.epsilon),
        )

        fx, fy, cx, cy = intrinsics.float().unbind(dim=-1)
        u = fx.unsqueeze(-1) * x / safe_z + cx.unsqueeze(-1)
        v = fy.unsqueeze(-1) * y / safe_z + cy.unsqueeze(-1)
        normalized_u = 2.0 * (u / width) - 1.0
        normalized_v = 2.0 * (v / height) - 1.0

        distance = F.relu(normalized_u.abs() - 1.0) + F.relu(
            normalized_v.abs() - 1.0
        )
        nearest_view_distance = distance.amin(dim=1)
        return nearest_view_distance.clamp(max=self.clip).mean()
