from typing import Dict
import torch
from omegaconf import DictConfig
from utils.handmodel import get_handmodel
from models.optimizer.optimizer import Optimizer
from models.base import OPTIMIZER


@OPTIMIZER.register()
class GraspWithObject(Optimizer):

    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cpu'
        self.slurm = slurm
        self.scale = cfg.scale
        self.palm_alignment = cfg.palm_alignment
        self.palm_alignment_weight = cfg.palm_alignment_weight
        self.clip_grad_by_value = cfg.clip_grad_by_value

        self.modeling_keys = cfg.modeling_keys
        self.batch_size = cfg.batch_size

        self.hand_model = get_handmodel(batch_size=self.batch_size, device=self.device)

    def optimize(self, x: torch.Tensor, data: Dict) -> torch.Tensor:
        """ Compute gradient for optimizer constraint

        Args:
            x: the denosied signal at current step, which is detached and is required grad
            data: data dict that provides original data

        Return:
            The optimizer objective value of current step
        """
        loss = 0.

        ## compute palm alignment optimization
        if self.palm_alignment:
            # palm_alignment = 1. - toward_of_palm(qpos) (dot prodoct) toward_of_object(qpos)
            # loss += self.palm_alignment_weight * palm_alignment
            pass

        raise NotImplementedError

    def gradient(self, x: torch.Tensor, data: Dict) -> torch.Tensor:
        """ Compute gradient for optimizer constraint

        Args:
            x: the denosied signal at current step
            data: data dict that provides original data

        Return:
            Commputed gradient
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            obj = self.optimize(x_in, data)
            grad = torch.autograd.grad(obj, x_in)[0]
            ## clip gradient by value
            grad = torch.clip(grad, **self.clip_grad_by_value)
        raise NotImplementedError
