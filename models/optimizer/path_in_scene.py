from typing import Dict
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig

from models.optimizer.optimizer import Optimizer
from models.optimizer.utils import transform_verts
from models.base import OPTIMIZER

@OPTIMIZER.register()
class PathInSceneOptimizer(Optimizer):

    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cpu'
        
        self.scale = cfg.scale
        self.scale_type = cfg.scale_type
        self.collision = cfg.collision
        self.collision_weight = cfg.collision_weight
        self.continuity = cfg.continuity
        self.continuity_weight = cfg.continuity_weight
        self.continuity_step = cfg.continuity_step

        self.robot_radius = cfg.robot_radius
        self.robot_top = cfg.robot_top
        self.robot_bottom = cfg.robot_bottom

        self.clip_grad_by_value = cfg.clip_grad_by_value
    
    def optimize(self, x: torch.Tensor, data: Dict) -> torch.Tensor:
        """ Compute gradient for optimizer constraint
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

        ## compute continuity optimization, the serveral start position should be close to the first start frame
        if self.continuity:
            continuity_loss = 0.0
            for t in range(self.continuity_step):
                traj_dist = torch.norm(x[:, t, :] - data['start'][:, 0, :], dim=-1)
                continuity_loss += F.relu(traj_dist - t * self.robot_radius).sum()
            
            loss += self.continuity_weight * continuity_loss
        
        ## compute collision optimization
        if self.collision:
            B, T, D = x.shape

            s_grid_map = data['s_grid_map']
            s_grid_min = data['s_grid_min']
            s_grid_max = data['s_grid_max']
            s_grid_dim = data['s_grid_dim']
            trans_mat = torch.tensor(np.array(data['trans_mat'], dtype=np.float32), device=self.device)
            z_trans = trans_mat[:, 2, -1]
            trans_mat_inv = torch.linalg.inv(trans_mat)

            ## compute the height of position (orignal scannet has uneven floor)
            ## Because the scene may be transformed in dataloader, we need to transform 
            ## the generated path back to use the height map.
            x3 = torch.cat([x, torch.zeros(B, T, 1, dtype=x.dtype, device=x.device)], dim=-1)
            x_trans = transform_verts(x3, trans_mat_inv)

            s_grid_min = s_grid_min.unsqueeze(1)
            s_grid_max = s_grid_max.unsqueeze(1)

            norm_x_batch = ((x_trans[..., 0:2] - s_grid_min) / (s_grid_max - s_grid_min) * 2 -1)
            height = F.grid_sample(
                s_grid_map.unsqueeze(1),   # <B, 1, H, W>
                norm_x_batch.view(-1, T, 1, 2), # <B, T, 1, 2>
                padding_mode='border', align_corners=True) # <B, 1, T, 1>
            height = height.view(B, T, 1) + z_trans.reshape(B, 1, 1)

            ## compute robotic collision with scene
            scene_verts = data['pos'].reshape(B, 1, -1, 3).repeat(1, T, 1, 1) # <B, T, N, 3>

            between = torch.logical_and(
                scene_verts[..., 2] > (height + self.robot_bottom), # <B, T, N> > <B, T, 1>
                scene_verts[..., 2] < (height + self.robot_top)     # <B, T, N> < <B, T, 1>
            )
            dist = torch.linalg.norm(scene_verts[..., 0:2] - x.unsqueeze(2), dim=-1) # <B, T, N>, norm(<B, T, N, 2> - <B, T, 1, 2>)
            dist = F.relu(self.robot_radius - dist)
            loss += self.collision_weight * (dist * between).sum()

        return (-1.0) * loss
     
    def gradient(self, x: torch.Tensor, data: Dict, variance: torch.Tensor) -> torch.Tensor:
        """ Compute gradient for optimizer constraint
        Args:
            x: the denosied signal at current step
            data: data dict that provides original data
            variance: variance at current step
        
        Return:
            Commputed gradient
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            obj = self.optimize(x_in, data)
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
