from pathlib import Path

from utils.config import load_config_with_bases


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_multiple_config_bases_merge_in_order(tmp_path):
    (tmp_path / "model.yaml").write_text("model:\n  dim: 32\n  depth: 2\n")
    (tmp_path / "data.yaml").write_text("data:\n  name: re10k\nmodel:\n  dim: 64\n")
    (tmp_path / "experiment.yaml").write_text(
        "extends:\n"
        "  - model.yaml\n"
        "  - data.yaml\n"
        "model:\n"
        "  depth: 4\n"
    )

    config = load_config_with_bases(tmp_path / "experiment.yaml")

    assert config.model.dim == 64
    assert config.model.depth == 4
    assert config.data.name == "re10k"


def test_3d_token_training_selects_dataset_specific_loader():
    dl3dv = load_config_with_bases(
        PROJECT_ROOT / "configs/3d_token/train_init.yaml"
    )
    re10k = load_config_with_bases(
        PROJECT_ROOT / "configs/3d_token/re10k_2view/train_init.yaml"
    )

    assert dl3dv.training.dataset_name == "data.dataset.Dataset"
    assert dl3dv.training.num_views == [32]
    assert re10k.training.dataset_name == "data.re10k_dataset.RE10KDataset"
    assert re10k.training.num_views == [2]
    assert re10k.training.data_loader_seed == 1234
    assert re10k.data.initial_min_context_gap == 25
    assert re10k.data.initial_max_context_gap == 25
    assert re10k.data.context_gap_warmup_steps == 37500
    assert re10k.validation.enabled is True
    assert re10k.validation.data_stage == "val"
    assert re10k.validation.every == 1000
    assert re10k.validation.seed == 3456


def test_mvp_re10k_inference_uses_re10k_data_base():
    config = load_config_with_bases(
        PROJECT_ROOT / "configs/mvp/inference_re10k_224.yaml"
    )

    assert config.inference.dataset_name == "data.re10k_dataset.RE10KDataset"
    assert config.data.stage == "test"
    assert config.data.num_context_views == 2
    assert config.data.pose_normalization == "mvp"
    assert "data_path" not in config.data
