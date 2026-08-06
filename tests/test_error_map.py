import torch

from model.rendering.error_map import sample_token_error


def test_sample_token_error_projects_world_center_to_image_center():
    xyz = torch.tensor([[[0.0, 0.0, 2.0]]])
    error = torch.zeros(1, 1, 1, 8, 8)
    error[..., 4, 4] = 1.0
    c2w = torch.eye(4).view(1, 1, 4, 4)
    intrinsics = torch.tensor([[[4.0, 4.0, 4.0, 4.0]]])
    sampled = sample_token_error(xyz, error, c2w, intrinsics)
    assert sampled.shape == (1, 1)
    assert torch.allclose(sampled, torch.ones_like(sampled))
