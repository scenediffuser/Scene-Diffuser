from typing import Dict
from torch.utils.data import Dataset
from utils.registry import Registry
DATASET = Registry('Dataset')

def create_dataset(cfg: dict, phase: str, slurm: bool, **kwargs: Dict) -> Dataset:
    """ Create a `torch.utils.data.Dataset` object from configuration.

    Args:
        cfg: configuration object, dataset configuration
        phase: phase string, can be 'train' and 'test'
        slurm: on slurm platform or not. This field is used to specify the data path
    
    Return:
        A Dataset object that has loaded the designated dataset.
    """
    return DATASET.get(cfg.name)(cfg, phase, slurm, **kwargs)
