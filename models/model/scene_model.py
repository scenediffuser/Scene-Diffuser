import torch.nn as nn
from omegaconf import DictConfig

from utils.registry import Registry
from models.model.pointtransformer import pointtransformer_enc_repro
from models.model.pointnet import pointnet_enc_repro


SCENEMODEL = Registry('SceneModel')

def create_scene_model(name: str, **kwargs) -> nn.Module:
    """ Create scene model for extract scene feature

    Args:
        name: scene model name
        
    Return:
        A 3D scene model
    """
    return SCENEMODEL.get(name)(**kwargs)

@SCENEMODEL.register()
def PointTransformer(**kwargs):
    return pointtransformer_enc_repro(**kwargs)

@SCENEMODEL.register()
def PointNet(**kwargs):
    return pointnet_enc_repro(**kwargs)
