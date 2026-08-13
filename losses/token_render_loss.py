from easydict import EasyDict as edict
import torch
import torch.nn.functional as F
from torch import nn


class RenderedImageLoss(nn.Module):
    """C3G-style independently weighted MSE and LPIPS rendering losses."""

    def __init__(self, config):
        super().__init__()
        loss_config = config.loss
        self.mse_weight = loss_config.mse.weight
        self.lpips_weight = loss_config.lpips.weight
        self.lpips_apply_after_step = loss_config.lpips.get("apply_after_step", 0)
        self.lpips = None
        if self.lpips_weight > 0:
            import lpips

            self.lpips = lpips.LPIPS(net="vgg").eval()
            for parameter in self.lpips.parameters():
                parameter.requires_grad_(False)

    def forward(self, rendering, target, global_step):
        batch, views, _, height, width = rendering.shape
        rendering = rendering.reshape(batch * views, 3, height, width)
        target = target[:, :, :3].reshape(batch * views, 3, height, width)

        mse_raw = F.mse_loss(rendering, target)
        mse = self.mse_weight * mse_raw
        lpips_loss = mse.new_zeros(())
        if self.lpips is not None and global_step >= self.lpips_apply_after_step:
            lpips_loss = self.lpips_weight * self.lpips(
                rendering,
                target,
                normalize=True,
            ).mean()

        return edict(
            total=mse + lpips_loss,
            mse=mse,
            lpips=lpips_loss,
            psnr=-10.0 * torch.log10(mse_raw.clamp_min(1e-8)),
        )


class TokenRenderLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.initial_weight = config.loss.get("initial_render_weight", 1.0)
        self.split_score_weight = config.loss.get("split_score_weight", 0.0)
        self.image_loss = RenderedImageLoss(config)

    def forward(
        self,
        render_initial,
        render_final,
        target,
        split_scores=None,
        split_target=None,
        global_step=0,
    ):
        initial = self.image_loss(render_initial, target, global_step)
        if render_final is None:
            return edict(
                loss=initial.total,
                mse=initial.mse,
                lpips=initial.lpips,
                psnr=initial.psnr,
            )

        final = self.image_loss(render_final, target, global_step)
        split_score = final.total.new_zeros(())
        if split_scores is not None and split_target is not None:
            split_score = self.split_score_weight * F.binary_cross_entropy_with_logits(
                split_scores.float(), split_target.float().clamp(0.0, 1.0)
            )
        return edict(
            loss=final.total + self.initial_weight * initial.total + split_score,
            initial_mse=self.initial_weight * initial.mse,
            initial_lpips=self.initial_weight * initial.lpips,
            initial_psnr=initial.psnr,
            final_mse=final.mse,
            final_lpips=final.lpips,
            final_psnr=final.psnr,
            split_score=split_score,
        )
