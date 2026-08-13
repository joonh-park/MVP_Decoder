from utils.config import load_config_with_bases


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
