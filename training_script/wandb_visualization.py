from math import radians, tan

import torch
import torch.nn.functional as F

from model.rendering.gaussian_renderer import GaussianRenderer


def _vertical_strip(images):
    return torch.cat(tuple(images), dim=1)


def _pad_height(image, height):
    return F.pad(image, (0, 0, 0, height - image.shape[1]), value=1.0)


@torch.no_grad()
def make_rendering_view(input_data, target_data, prediction):
    """C3G-style context | target | prediction | error comparison."""

    context = _vertical_strip(input_data["image"][0].detach().float().cpu())
    target = _vertical_strip(target_data["image"][0].detach().float().cpu())
    prediction = _vertical_strip(
        prediction[0].detach().float().clamp(0.0, 1.0).cpu()
    )
    error = (prediction - target).abs()
    columns = (context, target, prediction, error)
    height = max(column.shape[1] for column in columns)
    return torch.cat(
        tuple(_pad_height(column, height) for column in columns),
        dim=2,
    )


@torch.no_grad()
def make_xyz_projection_view(
    gaussians,
    sh_degree,
    near_plane,
    far_plane,
    low_pass_filter,
    resolution=256,
    margin=0.1,
    fov_degrees=10.0,
):
    """Render C3G-style YZ, ZX, and XY near-orthographic projections."""

    xyz = gaussians.xyz[:1].detach().float()
    minima = xyz.amin(dim=1)
    maxima = xyz.amax(dim=1)
    center = 0.5 * (minima + maxima)
    extent = (maxima - minima).amax(dim=-1).clamp_min(1.0e-3)
    extent = extent * (1.0 + 2.0 * margin)

    half_fov_tangent = tan(0.5 * radians(fov_degrees))
    camera_distance = 0.5 * extent / half_fov_tangent
    focal = 0.5 * resolution / half_fov_tangent

    c2w = torch.zeros((1, 3, 4, 4), device=xyz.device, dtype=xyz.dtype)
    intrinsics = torch.zeros((1, 3, 4), device=xyz.device, dtype=xyz.dtype)
    for look_axis in range(3):
        right_axis = (look_axis + 1) % 3
        down_axis = (look_axis + 2) % 3
        c2w[0, look_axis, right_axis, 0] = 1.0
        c2w[0, look_axis, down_axis, 1] = 1.0
        c2w[0, look_axis, look_axis, 2] = 1.0
        c2w[0, look_axis, :3, 3] = center[0]
        c2w[0, look_axis, look_axis, 3] -= (
            camera_distance[0] + 0.5 * extent[0]
        )
        c2w[0, look_axis, 3, 3] = 1.0
        intrinsics[0, look_axis] = intrinsics.new_tensor(
            [focal, focal, 0.5 * resolution, 0.5 * resolution]
        )

    projections = GaussianRenderer.render(
        xyz[0],
        gaussians.feature[:1].detach().float()[0],
        gaussians.scale[:1].detach().float()[0],
        gaussians.rotation[:1].detach().float()[0],
        gaussians.opacity[:1].detach().float()[0],
        c2w[0],
        intrinsics[0],
        resolution,
        resolution,
        sh_degree,
        near_plane,
        far_plane,
        low_pass_filter,
    )
    projections = projections.permute(0, 3, 1, 2).clamp(0.0, 1.0).cpu()
    return torch.cat(tuple(projections), dim=2)
