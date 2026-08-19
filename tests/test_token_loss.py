from easydict import EasyDict as edict
import torch

from losses.token_render_loss import TokenRenderLoss


def _config():
    return edict(
        model=edict(gaussians=edict(near_plane=0.01)),
        loss=edict(
            mse=edict(weight=1.0),
            lpips=edict(weight=0.0, apply_after_step=0),
            visibility=edict(weight=1.0, clip=1.0),
            initial_render_weight=1.0,
            split_score_weight=0.0,
        )
    )


def test_init_loss_uses_only_initial_render():
    loss_fn = TokenRenderLoss(_config())
    assert loss_fn.visibility_loss.epsilon == 0.01
    target = torch.ones(1, 2, 3, 4, 4)
    initial = torch.zeros_like(target)
    metrics = loss_fn(
        initial,
        None,
        target,
        visibility_initial=torch.tensor(0.25),
    )
    assert torch.allclose(metrics.loss, torch.tensor(1.25))
    assert torch.allclose(metrics.mse, torch.tensor(1.0))
    assert torch.allclose(metrics.visibility, torch.tensor(0.25))


def test_full_loss_combines_initial_and_final_render():
    loss_fn = TokenRenderLoss(_config())
    target = torch.ones(1, 2, 3, 4, 4)
    initial = torch.zeros_like(target)
    final = torch.full_like(target, 0.5)
    metrics = loss_fn(
        initial,
        final,
        target,
        visibility_initial=torch.tensor(0.25),
        visibility_final=torch.tensor(0.5),
    )
    assert torch.allclose(metrics.loss, torch.tensor(2.0))
    assert torch.allclose(metrics.initial_mse, torch.tensor(1.0))
    assert torch.allclose(metrics.final_mse, torch.tensor(0.25))
    assert torch.allclose(metrics.initial_visibility, torch.tensor(0.25))
    assert torch.allclose(metrics.final_visibility, torch.tensor(0.5))
