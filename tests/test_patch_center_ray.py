import torch

from model.backbones.mvp.frozen import compute_patch_center_plucker


def test_patch_center_ray_matches_original_patch_center():
    intrinsics = torch.tensor([[[80.0, 80.0, 40.0, 32.0]]])
    c2w = torch.eye(4).view(1, 1, 4, 4)
    ray = compute_patch_center_plucker(
        intrinsics,
        c2w,
        image_size=(64, 80),
        grid_size=(8, 10),
    )
    expected = torch.tensor([(4.0 - 40.0) / 80.0, (4.0 - 32.0) / 80.0, 1.0])
    expected = expected / expected.norm()
    assert torch.allclose(ray[0, 0, 0, 0, 3:6], expected, atol=1e-6)
