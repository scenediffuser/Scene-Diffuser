from typing import Dict
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig

from models.optimizer.optimizer import Optimizer
from models.base import PLANNER

@PLANNER.register()
class GreedyPathPlanner(Optimizer):

    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        self.scale = cfg.scale
        self.scale_type = cfg.scale_type
        self.greedy_type = cfg.greedy_type
        self.robot_radius = cfg.robot_radius

        self.clip_grad_by_value = cfg.clip_grad_by_value

    def objective(self, x: torch.Tensor, data: Dict):
        """ Compute gradient for planner guidance

        Args:
            x: the denosied signal at current step, which is detached and is required grad
            data: data dict that provides original data
        
        Return:
            The optimizer objective value of current step
        """
        loss = 0.

        ## important!!!
        ## normalize x and convert it to absolute representation
        if 'normalizer' in data and data['normalizer'] is not None:
            x = data['normalizer'].unnormalize(x)
        
        _, O, _ = data['start'].shape
        x[:, 0:O, :] = data['start'].clone() # copy start observation to x after unnormalize
        if 'repr_type' in data:
            if data['repr_type'] == 'absolute':
                pass
            elif data['repr_type'] == 'relative':
                x[:, O-1:, :] = torch.cumsum(x[:, O-1:, :], dim=1)
            else:
                raise Exception('Unsupported repr type.')
    
        ## compute objective
        target = data['target'] # <B, 2>
        if self.greedy_type == 'last_frame_l1':
            loss = torch.norm(x[:, -1, :] - target, dim=-1, p=1).sum()
        elif self.greedy_type == 'last_frame_exp':
            dist = torch.norm(x[:, -1, :] - target, dim=-1, p=1) # <B>
            loss += (-1.0) * torch.exp(1 / dist.clamp(min=0.1)).sum()
        elif self.greedy_type == 'all_frame_l1':
            loss = torch.norm(x - target.unsqueeze(1), dim=-1, p=1).sum()
        elif self.greedy_type == 'all_frame_exp':
            dist = torch.norm(x - target.unsqueeze(1), dim=-1, p=1) # <B, T>
            loss += (-1.0) * torch.exp(1 / dist.clamp(min=0.1)).sum()
        elif self.greedy_type == 'closest_l1':
            dist = torch.norm(x - target.unsqueeze(1), dim=-1, p=1) # <B, T>
            closest_dist = dist.min(-1)[0]
            loss += closest_dist.sum()
        else:
            raise Exception('Unsupported greedy type')

        return (-1.0) * loss

    def gradient(self, x: torch.Tensor, data: Dict, variance: torch.Tensor) -> torch.Tensor:
        """ Compute gradient for planner guidance
        Args:
            x: the denosied signal at current step
            data: data dict that provides original data
            variance: variance at current step
        
        Return:
            Commputed gradient
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            obj = self.objective(x_in, data)
            grad = torch.autograd.grad(obj, x_in)[0]
            ## clip gradient by value
            grad = torch.clip(grad, **self.clip_grad_by_value)
            ## TODO clip gradient by norm

            if self.scale_type == 'normal':
                grad = self.scale * grad * variance
            elif self.scale_type == 'div_var':
                grad = self.scale * grad
            else:
                raise Exception('Unsupported scale type!')

            return grad
