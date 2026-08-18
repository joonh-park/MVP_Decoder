import torch

from data.batch import concatenate_camera_batches


def _camera_batch(value, views):
    return {
        "image": torch.full((1, views, 3, 8, 8), value),
        "fxfycxcy": torch.full((1, views, 4), value),
        "c2w": torch.full((1, views, 4, 4), value),
        "index": torch.full((1, views, 1), int(value)),
        "scene_name": ["scene"],
    }


def test_camera_batches_concatenate_target_before_context():
    target = _camera_batch(1.0, 4)
    context = _camera_batch(2.0, 2)

    combined = concatenate_camera_batches(target, context)

    assert combined["image"].shape == (1, 6, 3, 8, 8)
    assert torch.all(combined["image"][:, :4] == 1.0)
    assert torch.all(combined["image"][:, 4:] == 2.0)
    assert combined["scene_name"] == ["scene"]
