from io import BytesIO

import numpy as np
from easydict import EasyDict as edict
from PIL import Image
import torch

from data.re10k_dataset import RE10KDataset


def _encoded_image(value):
    image = Image.fromarray(np.full((360, 640, 3), value, dtype=np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8)


def _config(root):
    return edict(
        data=edict(
            root=str(root),
            stage="train",
            image_size=[224, 224],
            original_image_size=[360, 640],
            num_context_views=2,
            num_target_views=4,
            min_context_gap=2,
            max_context_gap=4,
            min_target_distance=0,
            make_baseline_one=True,
            relative_pose=True,
            augment=False,
            skip_bad_shape=True,
            seed=42,
        )
    )


def test_re10k_chunk_to_mvp_batch(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    cameras = []
    images = []
    for index in range(6):
        w2c = torch.eye(4)
        w2c[0, 3] = -float(index)
        camera = torch.cat(
            (
                torch.tensor([0.8, 0.8, 0.5, 0.5, 0.0, 0.0]),
                w2c[:3].flatten(),
            )
        )
        cameras.append(camera)
        images.append(_encoded_image(index * 20))
    torch.save(
        [{"key": "scene", "cameras": torch.stack(cameras), "images": images}],
        train_dir / "000000.torch",
    )

    dataset = RE10KDataset(_config(tmp_path))
    example = next(iter(dataset))
    assert example["input_image"].shape == (2, 3, 224, 224)
    assert example["target_image"].shape == (4, 3, 224, 224)
    assert example["input_fxfycxcy"].shape == (2, 4)
    assert example["input_c2w"].shape == (2, 4, 4)
    assert torch.allclose(example["input_c2w"][0], torch.eye(4), atol=1e-5)
    baseline = (
        example["input_c2w"][0, :3, 3]
        - example["input_c2w"][1, :3, 3]
    ).norm()
    assert torch.allclose(baseline, torch.tensor(1.0), atol=1e-5)


def test_re10k_four_view_sampling_and_mvp_pose_normalization(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    cameras = []
    images = []
    for index in range(8):
        w2c = torch.eye(4)
        w2c[0, 3] = -float(index)
        cameras.append(
            torch.cat(
                (
                    torch.tensor([0.8, 0.8, 0.5, 0.5, 0.0, 0.0]),
                    w2c[:3].flatten(),
                )
            )
        )
        images.append(_encoded_image(index * 20))
    torch.save(
        [{"key": "scene", "cameras": torch.stack(cameras), "images": images}],
        train_dir / "000000.torch",
    )
    config = _config(tmp_path)
    config.data.num_context_views = 4
    config.data.min_context_gap = 4
    config.data.max_context_gap = 6
    config.data.pose_normalization = "mvp"

    example = next(iter(RE10KDataset(config)))

    assert example["input_image"].shape == (4, 3, 224, 224)
    assert example["input_c2w"].shape == (4, 4, 4)
    assert torch.allclose(
        example["input_c2w"][:, :3, 3].mean(dim=0),
        torch.zeros(3),
        atol=1e-5,
    )
    assert torch.allclose(
        example["input_c2w"][:, :3, 3].abs().max(),
        torch.tensor(1.0),
        atol=1e-5,
    )
