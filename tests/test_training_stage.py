from torch import nn

from training_script.training_utils import configure_3d_token_training_stage


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
