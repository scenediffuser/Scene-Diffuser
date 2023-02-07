from typing import Dict, List
import torch.nn as nn
from omegaconf import DictConfig
from utils.registry import Registry
from models.optimizer.optimizer import Optimizer
from models.planner.planner import Planner

MODEL = Registry('Model')
DIFFUSER = Registry('Diffuser')
OPTIMIZER = Registry('Optimizer')
PLANNER = Registry('Planner')

def create_model(cfg: DictConfig, *args: List, **kwargs: Dict) -> nn.Module:
    """ Create a generative model and return it.
    If 'diffuser' in cfg, this function will call `create_diffuser` function to create a diffusion model.
    Otherwise, this function will create other generative models, e.g., cvae.

    Args:
        cfg: configuration object, the global configuration
    
    Return:
        A generative model
    """
    if 'diffuser' in cfg:
        return create_diffuser(cfg, *args, **kwargs)

    return MODEL.get(cfg.model.name)(cfg.model, *args, **kwargs)

def create_diffuser(cfg: DictConfig, *args: List, **kwargs: Dict) -> nn.Module:
    """ Create a diffuser model, first create a eps_model from model config,
    then create a diffusion model and use the eps_model as input.

    Args:
        cfg: configuration object
    
    Return:
        A diffusion model
    """
    ## use diffusion model, the model is a eps model
    eps_model = MODEL.get(cfg.model.name)(cfg.model, *args, **kwargs)

    ## if the task has observation, then pass it to diffuser
    has_obser = cfg.task.has_observation if 'has_observation' in cfg.task else False
    diffuser  = DIFFUSER.get(cfg.diffuser.name)(eps_model, cfg.diffuser, has_obser, *args, **kwargs)

    ## if optimizer is in cfg, then load it and pass it to diffuser
    if 'optimizer' in cfg:
        optimizer = create_optimizer(cfg.optimizer, *args, **kwargs)
        diffuser.set_optimizer(optimizer)
    
    ## if planner is in cfg, then load it and pass it to diffuser
    if 'planner' in cfg:
        planner = create_planner(cfg.planner, *args, **kwargs)
        diffuser.set_planner(planner)

    return diffuser

def create_optimizer(cfg: DictConfig, *args: List, **kwargs: Dict) -> Optimizer:
    """ Create a optimizer for constrained sampling

    Args:
        cfg: configuration object
    
    Return:
        A optimizer used for guided sampling
    """
    if cfg is None:
        return None
    
    return OPTIMIZER.get(cfg.name)(cfg, *args, **kwargs)

def create_planner(cfg: DictConfig, *args: List, **kwargs: Dict) -> Planner:
    """ Create a planner for constrained sampling

    Args:
        cfg: configuration object
        
    Return:
        A planner used for guided sampling
    """
    if cfg is None:
        return None
    
    return PLANNER.get(cfg.name)(cfg, *args, **kwargs)
