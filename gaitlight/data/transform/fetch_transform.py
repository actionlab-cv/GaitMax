from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn

from gaitlight import data


def fetch_transform(config: DictConfig | ListConfig) -> ([nn.Module], [nn.Module]):
    """
    Fetch train & val transform from config.
    :param config:
        config.train: train transforms
        config.val: val transforms
    :return:
        list: train transforms
        list: val transforms
    """

    t_transform = OmegaConf.to_container(config.train) if 'train' in config else []
    t_transform = [
        getattr(data, x)() if isinstance(x, str)
        else getattr(data, x[0])(*x[1:])
        for x in t_transform
    ]
    v_transform = OmegaConf.to_container(config.val) if 'val' in config else []
    v_transform = [
        getattr(data, x)() if isinstance(x, str)
        else getattr(data, x[0])(*x[1:])
        for x in v_transform
    ]

    return t_transform, v_transform
