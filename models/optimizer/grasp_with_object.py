from typing import Dict
import os
import numpy as np
import torch
from omegaconf import DictConfig
from utils.handmodel import get_handmodel
from models.optimizer.optimizer import Optimizer
from models.base import OPTIMIZER
import pickle
import torch.functional as F

@OPTIMIZER.register()
class GraspWithObject(Optimizer):

    _BATCH_SIZE = 16
    _N_OBJ = 4096
    _joint_angle_lower = torch.tensor([-0.5235988, -0.7853982, -0.43633232, 0., 0., 0., -0.43633232, 0., 0., 0.,
                                       -0.43633232, 0., 0., 0., 0., -0.43633232, 0., 0., 0., -1.047, 0., -0.2618,
                                       -0.5237, 0.], device='cuda')
    _joint_angle_upper = torch.tensor([0.17453292, 0.61086524, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232,
                                       1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964,
                                       0.6981317, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 1.047, 1.309, 0.2618,
                                       0.5237, 1.], device='cuda')

    _global_trans_lower = torch.tensor([-0.13128923, -0.10665303, -0.45753425], device='cuda')
    _global_trans_upper = torch.tensor([0.12772022, 0.22954416, -0.21764427], device='cuda')

    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.
    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cpu'
        self.slurm = slurm
        self.scale = cfg.scale
        self.collision = cfg.collision
        self.collision_weight = cfg.collision_weight
        self.clip_grad_by_value = cfg.clip_grad_by_value

        self.modeling_keys = cfg.modeling_keys

        self.normalize_x = cfg.normalize_x
        self.normalize_x_trans = cfg.normalize_x_trans

        self.asset_dir = cfg.asset_dir_slrum if self.slurm else cfg.asset_dir
        self.obj_pcds_nors_dict = pickle.load(open(os.path.join(self.asset_dir, 'object_pcds_nors.pkl'), 'rb'))
        self.hand_model = get_handmodel(batch_size=self._BATCH_SIZE, device=self.device)

        self.relu = torch.nn.ReLU()

    def optimize(self, x: torch.Tensor, data: Dict, t: int) -> torch.Tensor:
        """ Compute gradient for optimizer constraint

        Args:
            x: the denosied signal at current step, which is detached and is required grad
            data: data dict that provides original data
            t: sample time

        Return:
            The optimizer objective value of current step
        """
        loss = 0.

        self.hand_model.update_kinematics(q=x)
        hand_pcd = self.hand_model.get_surface_points()
        n_hand = hand_pcd.shape[1]
        n_obj = self._N_OBJ
        obj_pcd_nor_list = []
        for object_name in data['scene_id'][t*self._BATCH_SIZE:(t+1)*self._BATCH_SIZE]:
            obj_pcd_nor_list.append(self.obj_pcds_nors_dict[object_name][:self._N_OBJ, :])
        obj_pcd_nor = np.stack(obj_pcd_nor_list, axis=0)
        obj_pcd_nor = torch.tensor(obj_pcd_nor, device='cuda')

        ## compute palm alignment optimization
        if self.collision:
            obj_pcd = obj_pcd_nor[..., :3]
            obj_nor = obj_pcd_nor[..., 3:6]
            # batch the obj pcd and hand pcd
            batch_obj_pcd = obj_pcd_nor[:, :, :3].view(self._BATCH_SIZE, 1, n_obj, 3).repeat(1, n_hand, 1, 1)
            batch_hand_pcd = hand_pcd.view(self._BATCH_SIZE, n_hand, 1, 3).repeat(1, 1, n_obj, 1)
            # compute the pair wise dist
            hand_obj_dist = (batch_obj_pcd - batch_hand_pcd).norm(dim=3)
            hand_obj_dist, hand_obj_indices = hand_obj_dist.min(dim=2)
            # gather the obj points and normals w.r.t. hand points
            hand_obj_points = torch.stack([obj_pcd[i, x, :] for i, x in enumerate(hand_obj_indices)], dim=0)
            hand_obj_normals = torch.stack([obj_nor[i, x, :] for i, x in enumerate(hand_obj_indices)], dim=0)
            # compute the signs
            hand_obj_signs = ((hand_obj_points - hand_pcd) * hand_obj_normals).sum(dim=2)
            hand_obj_signs = (hand_obj_signs > 0.).float()
            # signs dot dist to compute collision value
            # collision_value = (hand_obj_signs * hand_obj_dist).max(dim=1).values
            collision_value = (hand_obj_signs * hand_obj_dist).sum(dim=1)
            # collision_value = self.relu(collision_value - 0.1)
            # collision_value = torch.abs(collision_value - 0.005)
            loss += self.collision_weight * collision_value.mean()

        return (-1.0) * loss

    def gradient(self, x: torch.Tensor, data: Dict, variance: torch.Tensor) -> torch.Tensor:
        # print(f'compute gradient...')
        """ Compute gradient for optimizer constraint

        Args:
            x: the denosied signal at current step
            data: data dict that provides original data

        Return:
            Commputed gradient
        """
        assert (x.shape[0] % self._BATCH_SIZE == 0)
        with torch.enable_grad():
            # concatenate the id rot to x_in
            id_6d_rot = torch.tensor([1., 0., 0., 0., 1., 0.], device='cuda').view(1, 6).repeat(x.shape[0], 1)
            x = torch.cat([x[:, :3], id_6d_rot, x[:, 3:]], dim=-1)
            x_in = x.detach().requires_grad_(False)
            grad_list = []
            obj_list = []
            for i in range(x.shape[0] // self._BATCH_SIZE):
                i_x_in = x_in[i*self._BATCH_SIZE:(i+1)*self._BATCH_SIZE, :].detach().requires_grad_(True)
                if self.normalize_x_trans:
                    i_x_in_denorm_trans = self.trans_denormalize(i_x_in[:, :3])
                else:
                    i_x_in_denorm_trans = i_x_in[:, :3]
                if self.normalize_x:
                    i_x_in_denorm_angle = self.angle_denormalize(i_x_in[:, 9:])
                else:
                    i_x_in_denorm_angle = i_x_in[:, 9:]
                i_x_in_denorm = torch.cat([i_x_in_denorm_trans, i_x_in[:, 3:9], i_x_in_denorm_angle], dim=-1)
                obj = self.optimize(i_x_in_denorm, data, t=i)
                i_grad = torch.autograd.grad(obj, i_x_in)[0]
                obj_list.append(obj.abs().mean().detach().cpu())
                grad_list.append(i_grad)
            # print(f'loss: {np.mean(obj_list)}')
            grad = torch.cat(grad_list, dim=0)
            ## clip gradient by value
            # print(f'grad norm: {grad.abs().mean()}')
            grad = grad * self.scale
            grad = torch.clip(grad, **self.clip_grad_by_value)
            # grad = torch.cat([grad[:, :3], grad[:, 9:]], dim=-1)
            # grad = torch.cat([torch.zeros_like(grad[:, :3], device=self.device), grad[:, 9:]], dim=-1)
            grad = torch.cat([torch.zeros_like(grad[:, :3], device=self.device),
                              torch.zeros_like(grad[:, 9:11], device=self.device),
                              grad[:, 11:]], dim=-1)
            return grad

    def angle_denormalize(self, joint_angle: torch.Tensor):
        joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (self._joint_angle_upper - self._joint_angle_lower) + self._joint_angle_lower
        return joint_angle_denorm

    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self._global_trans_upper - self._global_trans_lower) + self._global_trans_lower
        return global_trans_denorm