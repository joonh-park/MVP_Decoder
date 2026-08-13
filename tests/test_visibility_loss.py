import torch

from losses.visibility_loss import VisibilityLoss


def _camera_batch(num_views=1):
    c2w = torch.eye(4).view(1, 1, 4, 4).repeat(1, num_views, 1, 1)
    intrinsics = torch.tensor([10.0, 10.0, 5.0, 5.0]).view(1, 1, 4)
    return c2w, intrinsics.repeat(1, num_views, 1)


def test_visibility_is_zero_inside_any_supervision_view():
    loss = VisibilityLoss()
    c2w, intrinsics = _camera_batch(num_views=2)
    c2w[:, 1, 0, 3] = 10.0
    xyz = torch.tensor([[[0.0, 0.0, 1.0]]])

    value = loss(xyz, c2w, intrinsics, image_size=(10, 10))

    assert torch.allclose(value, torch.tensor(0.0))


def test_visibility_penalizes_and_pulls_outside_gaussian_toward_image():
    loss = VisibilityLoss()
    c2w, intrinsics = _camera_batch()
    xyz = torch.tensor([[[0.75, 0.0, 1.0]]], requires_grad=True)

    value = loss(xyz, c2w, intrinsics, image_size=(10, 10))
    value.backward()

    assert 0.0 < value < 1.0
    assert xyz.grad[0, 0, 0] > 0


def test_visibility_clips_each_gaussian_penalty():
    loss = VisibilityLoss(clip=1.0)
    c2w, intrinsics = _camera_batch()
    xyz = torch.tensor([[[100.0, 100.0, 1.0]]])

    value = loss(xyz, c2w, intrinsics, image_size=(10, 10))

    assert torch.allclose(value, torch.tensor(1.0))
