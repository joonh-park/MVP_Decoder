from easydict import EasyDict as edict
import torch.nn.functional as F
from torch import nn

from losses.image_loss import LossComputer


class TokenRenderLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.initial_weight = config.training.get("initial_render_loss_weight", 1.0)
        self.split_score_weight = config.training.get("split_score_loss_weight", 0.0)
        self.image_loss = LossComputer(config)

    def forward(
        self,
        render_initial,
        render_final,
        target,
        split_scores=None,
        split_target=None,
    ):
        initial = self.image_loss(render_initial, target)
        if render_final is None:
            return edict(
                loss=initial.loss,
                initial_loss=initial.loss,
                initial_psnr=initial.psnr,
                final_loss=initial.loss,
                final_psnr=initial.psnr,
            )

        final = self.image_loss(render_final, target)
        split_score_loss = final.loss.new_zeros(())
        if split_scores is not None and split_target is not None:
            split_score_loss = F.binary_cross_entropy_with_logits(
                split_scores.float(), split_target.float().clamp(0.0, 1.0)
            )
        return edict(
            loss=(
                final.loss
                + self.initial_weight * initial.loss
                + self.split_score_weight * split_score_loss
            ),
            initial_loss=initial.loss,
            initial_psnr=initial.psnr,
            final_loss=final.loss,
            final_psnr=final.psnr,
            split_score_loss=split_score_loss,
        )
