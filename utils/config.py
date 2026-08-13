from pathlib import Path

from easydict import EasyDict as edict
from omegaconf import OmegaConf


def load_config_with_bases(config_path):
    config_path = Path(config_path)
    config = OmegaConf.load(config_path)
    if "extends" not in config:
        return config

    extends = config.extends
    del config["extends"]
    if isinstance(extends, str):
        extends = [extends]

    bases = [
        load_config_with_bases(config_path.parent / base_path)
        for base_path in extends
    ]
    return OmegaConf.merge(*bases, config)


def load_config(config_path):
    config = load_config_with_bases(config_path)
    return edict(OmegaConf.to_container(config, resolve=True))
