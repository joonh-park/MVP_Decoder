from types import SimpleNamespace

import pytest
import torch

from model.rendering.gaussian_renderer import scheduled_low_pass_filter
from training_script import wandb_visualization


def test_c3g_low_pass_schedule():
    kwargs = {
        "initial": 10.0,
        "minimum": 0.3,
        "decrease_factor": 3.0,
        "decrease_every": 1000,
    }
    assert scheduled_low_pass_filter(global_step=999, **kwargs) == 10.0
    assert scheduled_low_pass_filter(global_step=1000, **kwargs) == pytest.approx(
        10.0 / 3.0
    )
    assert scheduled_low_pass_filter(global_step=3000, **kwargs) == pytest.approx(
        10.0 / 27.0
    )
    assert scheduled_low_pass_filter(global_step=4000, **kwargs) == 0.3


def test_make_rendering_view_contains_all_context_and_target_views():
    input_data = {"image": torch.zeros(1, 2, 3, 8, 8)}
    target_data = {"image": torch.ones(1, 4, 3, 8, 8)}
    prediction = torch.full((1, 4, 3, 8, 8), 0.25)

    view = wandb_visualization.make_rendering_view(
        input_data,
        target_data,
        prediction,
    )

    assert view.shape == (3, 32, 32)
    assert torch.allclose(view[:, :16, :8], torch.zeros(3, 16, 8))
    assert torch.allclose(view[:, 16:, :8], torch.ones(3, 16, 8))
    assert torch.allclose(view[:, :, 8:16], torch.ones(3, 32, 8))
    assert torch.allclose(view[:, :, 16:24], torch.full((3, 32, 8), 0.25))
    assert torch.allclose(view[:, :, 24:], torch.full((3, 32, 8), 0.75))


def test_make_xyz_projection_view_builds_three_axis_cameras(monkeypatch):
    captured = {}

    def fake_render(
        xyz,
        feature,
        scale,
        rotation,
        opacity,
        c2w,
        intrinsics,
        width,
        height,
        sh_degree,
        near_plane,
        far_plane,
        low_pass_filter,
        background_color,
    ):
        captured["c2w"] = c2w
        captured["intrinsics"] = intrinsics
        captured["background_color"] = background_color
        return torch.zeros(3, height, width, 3)

    monkeypatch.setattr(
        wandb_visualization.GaussianRenderer,
        "render",
        staticmethod(fake_render),
    )
    gaussians = SimpleNamespace(
        xyz=torch.tensor([[[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]]),
        feature=torch.zeros(1, 2, 4, 3),
        scale=torch.zeros(1, 2, 3),
        rotation=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(1, 2, 1),
        opacity=torch.ones(1, 2, 1),
    )

    view = wandb_visualization.make_xyz_projection_view(
        gaussians,
        sh_degree=1,
        near_plane=0.01,
        far_plane=1.0e6,
        low_pass_filter=0.3,
        background_color=[0.0, 0.0, 0.0],
        resolution=32,
    )

    assert view.shape == (3, 32, 96)
    assert captured["c2w"].shape == (3, 4, 4)
    assert captured["intrinsics"].shape == (3, 4)
    assert torch.all(captured["intrinsics"][:, :2] > 0)
    assert captured["background_color"] == [0.0, 0.0, 0.0]
