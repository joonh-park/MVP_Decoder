import torch
import torch.nn.functional as F
from gsplat import rasterization

from model.token_decoder.types import GaussianParams


def scheduled_low_pass_filter(
    initial,
    minimum,
    decrease_factor,
    decrease_every,
    global_step,
):
    """Exponentially decay the renderer low-pass filter without steps."""

    value = float(initial)
    minimum = float(minimum)
    if decrease_every <= 0 or decrease_factor <= 1.0:
        return value
    progress = max(float(global_step), 0.0) / float(decrease_every)
    return max(minimum, value * float(decrease_factor) ** (-progress))


class GaussianRenderer(torch.autograd.Function):
    """Chunked differentiable renderer for scalar-opacity 3D tokens."""

    CHUNK_SIZE = 1

    @staticmethod
    def render(
        xyz,
        feature,
        scale,
        rotation,
        opacity,
        test_c2w,
        test_intr,
        width,
        height,
        sh_degree,
        near_plane,
        far_plane,
        low_pass_filter,
        background_color,
    ):
        batch_dims = xyz.shape[:-2]
        num_gaussians = xyz.shape[-2]
        num_cameras = test_c2w.shape[-3]
        color_basis = (sh_degree + 1) ** 2

        assert xyz.shape == batch_dims + (num_gaussians, 3)
        assert feature.shape == batch_dims + (num_gaussians, color_basis, 3)
        assert scale.shape == batch_dims + (num_gaussians, 3)
        assert rotation.shape == batch_dims + (num_gaussians, 4)
        assert opacity.shape == batch_dims + (num_gaussians, 1)
        assert test_c2w.shape == batch_dims + (num_cameras, 4, 4)
        assert test_intr.shape == batch_dims + (num_cameras, 4)

        scale = scale.exp()
        rotation = F.normalize(rotation, p=2, dim=-1)
        opacity = opacity.squeeze(-1)
        test_w2c = test_c2w.float().inverse()
        intrinsics = torch.zeros(
            batch_dims + (num_cameras, 3, 3),
            device=test_w2c.device,
            dtype=test_w2c.dtype,
        )
        intrinsics[..., 0, 0] = test_intr[..., 0]
        intrinsics[..., 1, 1] = test_intr[..., 1]
        intrinsics[..., 0, 2] = test_intr[..., 2]
        intrinsics[..., 1, 2] = test_intr[..., 3]
        intrinsics[..., 2, 2] = 1.0
        background = test_intr.new_tensor(background_color).expand(
            batch_dims + (num_cameras, 3)
        )
        rendering, _, _ = rasterization(
            xyz,
            rotation,
            scale,
            opacity,
            feature,
            test_w2c,
            intrinsics,
            width,
            height,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            render_mode="RGB",
            backgrounds=background,
            rasterize_mode="classic",
            eps2d=low_pass_filter,
        )
        return rendering

    @staticmethod
    def forward(
        ctx,
        xyz,
        feature,
        scale,
        rotation,
        opacity,
        test_c2ws,
        test_intr,
        width,
        height,
        sh_degree,
        near_plane,
        far_plane,
        low_pass_filter,
        background_color,
    ):
        ctx.save_for_backward(
            xyz, feature, scale, rotation, opacity, test_c2ws, test_intr
        )
        ctx.width = width
        ctx.height = height
        ctx.sh_degree = sh_degree
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        ctx.low_pass_filter = low_pass_filter
        ctx.background_color = background_color

        batch, views, _ = test_intr.shape
        renderings = torch.zeros(
            batch, views, height, width, 3, device=xyz.device, dtype=xyz.dtype
        )
        with torch.no_grad():
            for batch_index in range(batch):
                for view_index in range(0, views, GaussianRenderer.CHUNK_SIZE):
                    view_end = min(view_index + GaussianRenderer.CHUNK_SIZE, views)
                    renderings[batch_index, view_index:view_end] = GaussianRenderer.render(
                        xyz[batch_index],
                        feature[batch_index],
                        scale[batch_index],
                        rotation[batch_index],
                        opacity[batch_index],
                        test_c2ws[batch_index, view_index:view_end],
                        test_intr[batch_index, view_index:view_end],
                        width,
                        height,
                        sh_degree,
                        near_plane,
                        far_plane,
                        low_pass_filter,
                        background_color,
                    )
        return renderings.requires_grad_()

    @staticmethod
    def backward(ctx, grad_output):
        xyz, feature, scale, rotation, opacity, test_c2ws, test_intr = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()

        batch, views, _ = test_intr.shape
        with torch.enable_grad():
            for batch_index in range(batch):
                for view_index in range(0, views, GaussianRenderer.CHUNK_SIZE):
                    view_end = min(view_index + GaussianRenderer.CHUNK_SIZE, views)
                    rendering = GaussianRenderer.render(
                        xyz[batch_index],
                        feature[batch_index],
                        scale[batch_index],
                        rotation[batch_index],
                        opacity[batch_index],
                        test_c2ws[batch_index, view_index:view_end],
                        test_intr[batch_index, view_index:view_end],
                        ctx.width,
                        ctx.height,
                        ctx.sh_degree,
                        ctx.near_plane,
                        ctx.far_plane,
                        ctx.low_pass_filter,
                        ctx.background_color,
                    )
                    rendering.backward(grad_output[batch_index, view_index:view_end])

        return (
            xyz.grad,
            feature.grad,
            scale.grad,
            rotation.grad,
            opacity.grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def render_gaussians(
    gaussians: GaussianParams,
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    sh_degree: int,
    near_plane: float,
    far_plane: float,
    low_pass_filter: float,
    background_color: tuple[float, float, float] | list[float],
) -> torch.Tensor:
    if len(background_color) != 3:
        raise ValueError(
            f"background_color must contain 3 values, got {background_color}"
        )
    height, width = image_size
    rendering = GaussianRenderer.apply(
        gaussians.xyz,
        gaussians.feature,
        gaussians.scale,
        gaussians.rotation,
        gaussians.opacity,
        c2w,
        intrinsics,
        width,
        height,
        sh_degree,
        near_plane,
        far_plane,
        low_pass_filter,
        background_color,
    )
    return rendering.permute(0, 1, 4, 2, 3).contiguous()
