from torch import nn

from training_script.training_utils import (
    configure_3d_token_training_stage,
    create_optimizer,
)


class _Backbone(nn.Linear):
    freeze = True


class _TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone(4, 4)
        self.evidence_adapter = nn.Linear(4, 4)
        self.initializer = nn.Linear(4, 4)
        self.gaussian_head = nn.Linear(4, 4)
        self.error_encoder = nn.Linear(4, 4)
        self.splitter = nn.Linear(4, 4)
        self.refiner = nn.Linear(4, 4)
        self.split_enabled = False
        self.refinement_enabled = False


def test_init_stage_selects_only_initial_decoder_modules():
    model = _TokenModel()
    trainable_names = configure_3d_token_training_stage(model, "init")
    trainable_prefixes = {name.split(".", 1)[0] for name in trainable_names}
    assert trainable_prefixes == {
        "evidence_adapter",
        "initializer",
        "gaussian_head",
    }
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(not parameter.requires_grad for parameter in model.splitter.parameters())


def test_init_stage_can_train_unfrozen_backbone():
    model = _TokenModel()
    model.backbone.freeze = False

    trainable_names = configure_3d_token_training_stage(model, "init")

    assert any(name.startswith("backbone.") for name in trainable_names)
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(not parameter.requires_grad for parameter in model.splitter.parameters())


def test_full_stage_trains_every_decoder_module():
    model = _TokenModel()
    model.split_enabled = True
    model.refinement_enabled = True
    trainable_names = configure_3d_token_training_stage(model, "full")
    assert trainable_names
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.initializer.parameters())
    assert all(parameter.requires_grad for parameter in model.splitter.parameters())
    assert all(parameter.requires_grad for parameter in model.refiner.parameters())


def test_full_stage_can_train_unfrozen_backbone():
    model = _TokenModel()
    model.backbone.freeze = False
    model.split_enabled = True

    trainable_names = configure_3d_token_training_stage(model, "full")

    assert any(name.startswith("backbone.") for name in trainable_names)
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_optimizer_applies_backbone_lr_multiplier():
    model = _TokenModel()
    model.backbone.freeze = False
    configure_3d_token_training_stage(model, "init")

    optimizer, _, _ = create_optimizer(
        model,
        weight_decay=0.05,
        learning_rate=2.0e-4,
        betas=(0.9, 0.95),
        backbone_lr_multiplier=0.01,
    )
    group_lrs = {
        group["group_name"]: group["lr"] for group in optimizer.param_groups
    }

    assert group_lrs["decoder"] == 2.0e-4
    assert group_lrs["backbone"] == 2.0e-6
