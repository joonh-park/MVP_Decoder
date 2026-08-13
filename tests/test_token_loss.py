from easydict import EasyDict as edict
import torch

from losses.token_render_loss import TokenRenderLoss


def _config():
    return edict(
        loss=edict(
            mse=edict(weight=1.0),
            lpips=edict(weight=0.0, apply_after_step=0),
            visibility=edict(weight=1.0, clip=1.0),
            scale_regularization=edict(weight=1.0, threshold=0.1),
            initial_render_weight=1.0,
            split_score_weight=0.0,
        )
    )


def test_init_loss_uses_only_initial_render():
    loss_fn = TokenRenderLoss(_config())
    target = torch.ones(1, 2, 3, 4, 4)
    initial = torch.zeros_like(target)
    metrics = loss_fn(
        initial,
        None,
        target,
        visibility_initial=torch.tensor(0.25),
        scale_initial=torch.tensor([0.05, 0.2]),
    )
    assert torch.allclose(metrics.loss, torch.tensor(1.255))
    assert torch.allclose(metrics.mse, torch.tensor(1.0))
    assert torch.allclose(metrics.visibility, torch.tensor(0.25))
    assert torch.allclose(metrics.scale_regularization, torch.tensor(0.005))


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
        scale_initial=torch.tensor([0.05, 0.2]),
        scale_final=torch.tensor([0.1, 0.3]),
    )
    assert torch.allclose(metrics.loss, torch.tensor(2.025))
    assert torch.allclose(metrics.initial_mse, torch.tensor(1.0))
    assert torch.allclose(metrics.final_mse, torch.tensor(0.25))
    assert torch.allclose(metrics.initial_visibility, torch.tensor(0.25))
    assert torch.allclose(metrics.final_visibility, torch.tensor(0.5))
    assert torch.allclose(
        metrics.initial_scale_regularization,
        torch.tensor(0.005),
    )
    assert torch.allclose(
        metrics.final_scale_regularization,
        torch.tensor(0.02),
    )
