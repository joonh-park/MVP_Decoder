from pathlib import Path

from easydict import EasyDict as edict
from omegaconf import OmegaConf


def load_config_with_bases(config_path):
    config_path = Path(config_path)
    config = OmegaConf.load(config_path)
    if "extends" not in config:
        return config

    base_path = config_path.parent / config.extends
    del config["extends"]
    return OmegaConf.merge(load_config_with_bases(base_path), config)


def load_config(config_path):
    config = load_config_with_bases(config_path)
    return edict(OmegaConf.to_container(config, resolve=True))
