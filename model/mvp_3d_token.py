import importlib
from pathlib import Path

import torch
from easydict import EasyDict as edict
from torch import nn

from losses.token_render_loss import TokenRenderLoss
from model.rendering.error_map import (
    ErrorEvidenceEncoder,
    compute_input_error,
    sample_token_error,
)
from model.rendering.gaussian_renderer import (
    GaussianRenderer,
    render_gaussians,
    scheduled_low_pass_filter,
)
from model.token_decoder.evidence_adapter import EvidenceAdapter
from model.token_decoder.gaussian_head import SharedGaussianHead
from model.token_decoder.latent_split import LatentSplitter
from model.token_decoder.token_initializer import TokenInitializer
from model.token_decoder.token_refiner import TokenRefiner
from model.token_decoder.types import TokenDecoderOutput
from utils.config import load_config


class MVP3DTokenModel(nn.Module):
    """Posed MVP evidence followed by an emergent latent 3D-token decoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        decoder_config = config.model.decoder
        gaussian_config = config.model.gaussians
        attention_config = decoder_config.attention
        initializer_config = decoder_config.initializer
        attention_kwargs = {
            "num_heads": attention_config.num_heads,
            "mlp_ratio": attention_config.get("mlp_ratio", 4.0),
            "dropout": attention_config.get("dropout", 0.0),
            "query_chunk_size": attention_config.get("query_chunk_size", 0),
            "evidence_chunk_size": attention_config.get(
                "evidence_chunk_size", 1024
            ),
            "slot_epsilon": attention_config.get("slot_epsilon", 1e-8),
            "slot_null": attention_config.get("slot_null", True),
        }

        backbone_settings = config.model.backbone
        backbone_module, backbone_name = backbone_settings.class_name.rsplit(".", 1)
        backbone_class = importlib.import_module(backbone_module).__dict__[backbone_name]
        backbone_config = load_config(backbone_settings.config_path)
        checkpoint_path = backbone_settings.get("checkpoint_path")
        if checkpoint_path is None:
            checkpoint_path = backbone_config.get("inference", {}).get("ckpt_path")
        if checkpoint_path is None:
            raise ValueError(
                "Backbone checkpoint is missing: set inference.ckpt_path in "
                f"{backbone_settings.config_path} or model.backbone.checkpoint_path"
            )
        self.backbone_checkpoint_path = checkpoint_path
        self.backbone = backbone_class(
            backbone_config,
            checkpoint_path=checkpoint_path,
            freeze=backbone_settings.get("freeze", True),
        )
        self.evidence_adapter = EvidenceAdapter(
            feature_dim=self.backbone.output_dim,
            token_dim=decoder_config.token_dim,
        )
        self.initializer = TokenInitializer(
            num_queries=initializer_config.num_queries,
            dim=decoder_config.token_dim,
            layer_specs=initializer_config.layers,
            **attention_kwargs,
        )
        self.gaussian_head = SharedGaussianHead(
            dim=decoder_config.token_dim,
            sh_degree=gaussian_config.sh_degree,
            position_range=decoder_config.get("position_range", 4.0),
            scale_bias=gaussian_config.scale_bias,
            scale_max=gaussian_config.scale_max,
            opacity_bias=gaussian_config.opacity_bias,
        )

        split_config = decoder_config.split
        self.split_enabled = split_config.enabled
        self.split_dense = split_config.get("dense", True)
        self.split_threshold = split_config.get("threshold", 0.5)

        refinement_config = decoder_config.refinement
        refinement_layers = list(refinement_config.get("layers", []))
        self.refinement_enabled = len(refinement_layers) > 0
        needs_error = self.split_enabled or self.refinement_enabled
        self.error_encoder = (
            ErrorEvidenceEncoder(decoder_config.token_dim) if needs_error else None
        )
        self.splitter = (
            LatentSplitter(
                dim=decoder_config.token_dim,
                num_heads=attention_config.num_heads,
                mlp_ratio=attention_config.get("mlp_ratio", 4.0),
                dropout=attention_config.get("dropout", 0.0),
                query_chunk_size=attention_config.get("query_chunk_size", 0),
            )
            if self.split_enabled
            else None
        )
        self.refiner = (
            TokenRefiner(
                dim=decoder_config.token_dim,
                layer_specs=refinement_layers,
                **attention_kwargs,
            )
            if self.refinement_enabled
            else None
        )

        GaussianRenderer.CHUNK_SIZE = config.training.get("chunk_size", 1)
        self.loss_computer = TokenRenderLoss(config)

    def train(self, mode: bool = True):
        super().train(mode)
        self.loss_computer.eval()
        return self

    def _render(self, gaussians, camera_data, low_pass_filter):
        image = camera_data["image"]
        return render_gaussians(
            gaussians,
            c2w=camera_data["c2w"],
            intrinsics=camera_data["fxfycxcy"],
            image_size=(image.shape[-2], image.shape[-1]),
            sh_degree=self.config.model.gaussians.sh_degree,
            near_plane=self.config.model.gaussians.near_plane,
            far_plane=self.config.model.gaussians.far_plane,
            low_pass_filter=low_pass_filter,
        )

    def forward(self, input_data_dict, target_data_dict=None, global_step=0):
        gaussian_config = self.config.model.gaussians
        low_pass_filter = scheduled_low_pass_filter(
            initial=gaussian_config.get("low_pass_filter", 0.3),
            minimum=gaussian_config.get("low_pass_filter_min", 0.3),
            decrease_factor=gaussian_config.get("decrease_lpf_factor", 3.0),
            decrease_every=gaussian_config.get("decrease_lpf_step", 0),
            global_step=global_step,
        )
        frozen = self.backbone(
            input_data_dict["image"],
            input_data_dict["fxfycxcy"],
            input_data_dict["c2w"],
        )
        evidence = self.evidence_adapter(frozen.feature, frozen.center_ray)
        z_initial = self.initializer(evidence)
        gaussians_initial = self.gaussian_head(z_initial)

        render_initial = None
        if target_data_dict is not None and target_data_dict.get("image") is not None:
            render_initial = self._render(
                gaussians_initial,
                target_data_dict,
                low_pass_filter,
            )

        z_final = None
        gaussians_final = None
        render_final = None
        input_error = None
        split_scores = None
        split_target = None

        if self.split_enabled or self.refinement_enabled:
            input_render = self._render(
                gaussians_initial,
                input_data_dict,
                low_pass_filter,
            )
            input_error = compute_input_error(input_render, input_data_dict["image"])
            split_target = sample_token_error(
                gaussians_initial.xyz,
                input_error,
                input_data_dict["c2w"],
                input_data_dict["fxfycxcy"],
            )
            error_evidence = self.error_encoder(input_error, frozen.grid_size)
            conditioned_evidence = evidence + error_evidence

            if self.split_enabled:
                split = self.splitter(
                    z_initial,
                    conditioned_evidence,
                    dense=self.split_dense,
                    threshold=self.split_threshold,
                )
                z_final = split.latent
                split_scores = split.scores
            else:
                z_final = z_initial

            if self.refinement_enabled:
                z_final = self.refiner(z_final, conditioned_evidence)

            gaussians_final = self.gaussian_head(z_final)
            if target_data_dict is not None and target_data_dict.get("image") is not None:
                render_final = self._render(
                    gaussians_final,
                    target_data_dict,
                    low_pass_filter,
                )

        decoder_output = TokenDecoderOutput(
            z_initial=z_initial,
            gaussians_initial=gaussians_initial,
            render_initial=render_initial,
            z_final=z_final,
            gaussians_final=gaussians_final,
            render_final=render_final,
            input_error=input_error,
            split_scores=split_scores,
            split_target=split_target,
        )

        result = edict(
            input=input_data_dict,
            target=target_data_dict,
            decoder_output=decoder_output,
            gaussians=(
                gaussians_final if gaussians_final is not None else gaussians_initial
            ),
            render=render_final if render_final is not None else render_initial,
            render_initial=render_initial,
            low_pass_filter=low_pass_filter,
        )
        if render_initial is not None:
            result.loss_metrics = self.loss_computer(
                render_initial,
                render_final,
                target_data_dict["image"],
                split_scores=split_scores,
                split_target=split_target,
                global_step=global_step,
            )
        return result

    @torch.no_grad()
    def load_ckpt(self, load_path):
        path = Path(load_path)
        if path.is_dir():
            checkpoints = sorted(path.glob("*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {path}")
            path = checkpoints[-1]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)
        cleaned = {}
        for key, value in state_dict.items():
            key = key.replace("_checkpoint_wrapped_module.", "")
            key = key.replace("_orig_mod.", "")
            while key.startswith("module."):
                key = key[len("module."):]
            cleaned[key] = value
        return self.load_state_dict(cleaned, strict=False)
