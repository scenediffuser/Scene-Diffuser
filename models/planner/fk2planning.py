from typing import Dict
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig

from models.optimizer.optimizer import Optimizer
from models.base import PLANNER

@PLANNER.register()
class GreedyFK2Planner(Optimizer):

    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        self.scale = cfg.scale

        self.greedy_type = cfg.greedy_type

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

        ## important!!! scale x with region_area
        ## convert x to reality psoition
        target = data['target']  # <B, 7>

        if self.greedy_type == 'last_frame':
            loss += F.l1_loss(x[:, -1, :], target, reduction='mean')
        elif self.greedy_type == 'all_frame':
            loss += F.l1_loss(x, target.unsqueeze(1), reduction='mean')
        elif self.greedy_type == 'all_frame_exp':
            traj_dist = torch.norm(x - target.unsqueeze(1), dim=-1, p=1)
            loss += (-1.0) * torch.exp(1 / traj_dist.clamp(min=0.01)).sum()
        else:
            raise Exception('Unsupported greedy type')

        return (-1.0) * loss

    def gradient(self, x: torch.Tensor, data: Dict, variance: torch.Tensor) -> torch.Tensor:
        """ Compute gradient for planner guidance
        Args:
            x: the denosied signal at current step
            data: data dict that provides original data

        Return:
            Commputed gradient
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            obj = self.objective(x_in, data)
            grad = torch.autograd.grad(obj, x_in)[0]

            # print(f'obj: {-obj.detach().cpu()}')

            ## clip gradient by value
            grad = grad * self.scale
            grad = torch.clip(grad, **self.clip_grad_by_value)
            ## TODO clip gradient by norm

            return grad