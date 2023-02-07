from typing import Dict
import os
import smplx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from chamfer_distance import ChamferDistance as chamfer_dist

from utils.smplx_utils import convert_smplx_parameters_format
from models.optimizer.optimizer import Optimizer
from models.optimizer.utils import SMPLXGeometry, extract_smplx, SMPLXLayer, transform_verts
from models.base import OPTIMIZER

@OPTIMIZER.register()
class PoseInSceneOptimizer(Optimizer):

    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cpu'
        self.slurm = slurm
        self.scale = cfg.scale
        self.scale_type = cfg.scale_type
        self.contact = cfg.contact
        self.contact_weight = cfg.contact_weight
        self.contact_degree_threshold = cfg.contact_degree_threshold
        self.collision = cfg.collision
        self.collision_weight = cfg.collision_weight
        self.vposer = cfg.vposer
        self.vposer_weight = cfg.vposer_weight
        self.modeling_keys = cfg.modeling_keys
        self.contact_body_part = cfg.contact_body_part
        self.clip_grad_by_value = cfg.clip_grad_by_value
        self.gravity_dim = cfg.gravity_dim

        self.vposer_dir = cfg.vposer_dir_slurm if self.slurm else cfg.vposer_dir
        self.smplx_dir = cfg.smpl_dir_slurm if self.slurm else cfg.smpl_dir
        self.prox_dir = cfg.prox_dir_slurm if self.slurm else cfg.prox_dir
        self.body_segments_dir = os.path.join(self.prox_dir, 'body_segments')
        
        if self.vposer:
            vp, _ = load_model(self.vposer_dir, model_code=VPoser, 
                                remove_words_in_model_weights='vp_model.')
            self.vposer_model = vp.to(self.device)
        if self.contact or self.collision:
            self.SMPLX_neutral = SMPLXLayer(self.smplx_dir, self.device, cfg.num_pca_comps)
        if self.contact:
            self.smplx_geometry = SMPLXGeometry(self.body_segments_dir)
        
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

        ## compute vposer optimization
        if self.vposer:
            body_pose = extract_smplx(x, 'body_pose')
            shape_pre, dim = body_pose.shape[:-1], body_pose.shape[-1]
            pose_z = self.vposer_model.encode(body_pose.reshape(-1, dim)).mean
            pose_z = pose_z.reshape(*shape_pre, -1)
            vposer_opt = torch.mean(pose_z ** 2)

            loss += self.vposer_weight * vposer_opt
        
        ## if use contact or collision optimization, then compute the body mesh with smplx model
        if self.contact or self.collision:
            body_param_torch = convert_smplx_parameters_format(x, target='dict', keep_keys=self.modeling_keys)
            vertices, faces, joints = self.SMPLX_neutral.run(body_param_torch)

        ## compute contact optimization
        if self.contact:
            B = data['x'].shape[0]
            scene_verts = data['pos'].reshape(B, -1, 3)

            ## compute smplx vertex normal
            smplx_vertices = vertices.detach()
            smplx_face = torch.tensor(faces.astype(np.int64), device=vertices.device) # <F, 3>

            smplx_face_vertices = smplx_vertices[:, smplx_face] # <B, F, 3, 3>
            e1 = smplx_face_vertices[:, :, 1] - smplx_face_vertices[:, :, 0]
            e2 = smplx_face_vertices[:, :, 2] - smplx_face_vertices[:, :, 0]
            e1 = e1 / torch.norm(e1, dim=-1, p=2).unsqueeze(-1)
            e2 = e2 / torch.norm(e2, dim=-1, p=2).unsqueeze(-1)
            smplx_face_normal = torch.cross(e1, e2)  # <B, F, 3>

            smplx_vertex_normals = torch.zeros(smplx_vertices.shape).float().cuda() # <B, V, 3>
            smplx_vertex_normals.index_add_(1, smplx_face[:,0], smplx_face_normal)
            smplx_vertex_normals.index_add_(1, smplx_face[:,1], smplx_face_normal)
            smplx_vertex_normals.index_add_(1, smplx_face[:,2], smplx_face_normal)
            smplx_vertex_normals = smplx_vertex_normals / torch.norm(smplx_vertex_normals, dim=-1, p=2).unsqueeze(-1) # <B, V, 3>

            ## compute distance from scene to body
            ## 1. compute distance for all contact part vertex
            vid, fid = self.smplx_geometry.get_contact_id(self.contact_body_part)
            verts_contact = vertices[:, vid, :]
            dist1, dist2, idx1, idx2 = chamfer_dist(
                verts_contact.contiguous(), 
                scene_verts.contiguous()
            )
            ## 2. only consider the vertex that has a downward normal
            downward_mask = ((-1.) * smplx_vertex_normals[..., self.gravity_dim]) > np.cos(np.pi * self.contact_degree_threshold / 180) # <B, V>
            contact_mask = downward_mask[:, vid] # <B, CV>
            cham_dist = torch.mean((dist1 * contact_mask).sum(dim=-1) / (contact_mask.sum(dim=-1) + 1e-6)) # use unidirectional chamferdistance
            loss += self.contact_weight * cham_dist

        ## compute collision optimization
        if self.collision:
            s_grid_sdf = data['s_grid_sdf']
            s_grid_min = data['s_grid_min']
            s_grid_max = data['s_grid_max']
            s_grid_dim = data['s_grid_dim']

            cam_tran = torch.tensor(np.array(data['cam_tran']), device=self.device)
            origin_cam_tran = torch.tensor(np.array(data['origin_cam_tran']), device=self.device)
            ## Because the scene may be transformed in dataloader, we need to transform 
            ## the generated body back to use the scene sdf.
            ## scene_T @ origin_cam_T = cur_cam_T
            ## inv(scene_T) = inv(cur_cam_T @ inv(origin_cam_T))
            ##              = origin_cam_T @ inv(cur_cam_T)
            scene_trans_inv = torch.matmul(origin_cam_tran, torch.linalg.inv(cam_tran))
            vertices_trans = transform_verts(vertices, scene_trans_inv)

            s_grid_min = s_grid_min.unsqueeze(1)
            s_grid_max = s_grid_max.unsqueeze(1)

            norm_verts_batch = ((vertices_trans - s_grid_min) 
                                    / (s_grid_max - s_grid_min) * 2 -1)

            n_verts = norm_verts_batch.shape[1]
            body_sdf_batch = F.grid_sample(s_grid_sdf.unsqueeze(1), 
                            norm_verts_batch[:,:,[2,1,0]].view(-1,n_verts,1,1,3),
                            padding_mode='border', align_corners=False)

            # if there are no penetrating vertices then set sdf_penetration_loss = 0
            if body_sdf_batch.lt(0).sum().item() < 1:
                sdf_pene = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            else:
                sdf_pene = body_sdf_batch[body_sdf_batch < 0].abs().mean()
            
            loss += self.collision_weight * sdf_pene

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
