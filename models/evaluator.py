import os
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from omegaconf import DictConfig
from sklearn.metrics import pairwise_distances
from chamfer_distance import ChamferDistance as chamfer_dist

from utils.registry import Registry
from utils.smplx_utils import get_marker_indices, smplx_signed_distance
from models.optimizer.utils import SMPLXGeometry

EVALUATOR = Registry('Evaluator')

@EVALUATOR.register()
class PoseGenEval():
    def __init__(self, cfg: DictConfig) -> None:
        """ Evaluator class for pose generation task.

        Args:
            cfg: evaluator configuration
        """
        self.ksample = cfg.ksample
        self.contact_threshold = cfg.contact_threshold # e.g. 0.05
    
    @torch.no_grad()
    def evaluate(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        save_dir: str,
    ) -> None:
        """ Evaluate method

        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
        """
        smplx_geometry = SMPLXGeometry(os.path.join(
            dataloader.dataset.prox_dir, 'body_segments'))
        contact_body_part_vid, _ = smplx_geometry.get_contact_id(
            ['back','gluteus','L_Hand','R_Hand','L_Leg','R_Leg','thighs']
        )
        
        model.eval()
        device = model.device
        
        res = defaultdict(list)
        res['ksample'] = self.ksample
        res['contact_threshold'] = self.contact_threshold
        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            
            outputs = model.sample(data, k=self.ksample) # <B, k, T, ...>
            B = outputs.shape[0]
            scene_pos = data['pos'].reshape(B, -1, 3)

            for i in range(B):
                smplx_params = outputs[i, :, -1, :] # <k, D>

                ## diversity metrics
                ## 0. transl pairwise distance and standard deviation
                transl_np = smplx_params[:, 0:3].cpu().numpy()
                k, D = transl_np.shape
                transl_pdist = pairwise_distances(
                    transl_np, transl_np, metric='l2').sum() / (k * (k - 1))
                transl_std = np.std(transl_np, axis=0).mean()

                res['transl_pdist'].append(float(transl_pdist))
                res['transl_std'].append(float(transl_std))

                ## 1. smplx parameter pairwise distance and standard deviation
                smplx_params_np = smplx_params[:, 3:].cpu().numpy()
                k, D = smplx_params_np.shape
                param_pdist = pairwise_distances(
                    smplx_params_np, smplx_params_np, metric='l2').sum() / (k * (k - 1))
                param_std = np.std(smplx_params_np, axis=0).mean()

                res['param_pdist'].append(float(param_pdist))
                res['param_std'].append(float(param_std))

                ## 2. global body marker pairwise distance and standard deviation
                smplx_params_local = smplx_params.clone()
                smplx_params_local[:, 0:3] = 0
                body_verts_local, body_faces, body_joints_local = dataloader.dataset.SMPLX.run(smplx_params_local)
                body_verts_local_np = body_verts_local.cpu().numpy()
                k, V, D = body_verts_local_np.shape
                body_marker = body_verts_local_np.reshape(k, V, D)[:, get_marker_indices(), :]
                body_marker = body_marker.reshape(k, -1) # <k, M * 3>, concatenation of M marker coordinates
                marker_pdist = pairwise_distances(
                    body_marker, body_marker, metric='l2').sum() / (k * (k - 1))
                marker_std = np.std(body_marker, axis=0).mean()

                res['marker_pdist'].append(float(marker_pdist))
                res['marker_std'].append(float(marker_std))

                ## physics metrics
                ## non-collision score and contact score
                body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params)
                non_collision_score = []
                contact_score = []
                scene_verts = scene_pos[i].unsqueeze(0)
                body_verts_tensor = body_verts.to(device)
                body_faces_tensor = torch.tensor(body_faces.astype(np.int64)).to(device)
                for j in range(k):
                    scene_to_human_sdf, _ = smplx_signed_distance(
                        object_points=scene_verts,
                        smplx_vertices=body_verts_tensor[j:j+1],
                        smplx_face=body_faces_tensor
                    ) # <B, O> = D(<B, O, 3>, <B, H, 3>)
                    sdf = scene_to_human_sdf.cpu().numpy() # <1, O>
                    non_collision = np.sum(sdf <= 0) / sdf.shape[-1]
                    non_collision_score.append(non_collision)
                    
                    ## computation method of paper "Generating 3D People in Scenes without People"
                    ## not very reasonable
                    # if np.sum(sdf > 0) > 0:
                    #     contact = 1.0
                    # else:
                    #     contact = 0.0
                    # contact_score.append(contact)
                    ## we compute the chamfer distance between contact body part and scene
                    body_verts_contact = body_verts_tensor[j:j+1][:, contact_body_part_vid, :]
                    dist1, dist2, idx1, idx2 = chamfer_dist(
                        body_verts_contact.contiguous(), 
                        scene_verts.contiguous()
                    )
                    if torch.sum(dist1 < self.contact_threshold) > 0:
                        contact = 1.0
                    else:
                        contact = 0.0
                    contact_score.append(contact)

                res['non_collision'].append(sum(non_collision_score) / len(non_collision_score))
                res['contact'].append(sum(contact_score) / len(contact_score))
        
        for key in ['transl_pdist', 'transl_std', 'param_pdist', 'param_std', 'marker_pdist', 'marker_std', 'non_collision', 'contact']:
            res[key+'_average'] = sum(res[key]) / len(res[key])
        
        import json
        save_path = os.path.join(save_dir, 'metrics.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as fp:
            json.dump(res, fp)

@EVALUATOR.register()
class MotionGenEval():
    def __init__(self, cfg: DictConfig) -> None:
        """ Evaluator class for motion generation task.

        Args:
            cfg: evaluator configuration
        """
        self.ksample = cfg.ksample
        self.contact_threshold = cfg.contact_threshold # e.g. 0.05
        self.eval_case_num = cfg.eval_case_num
    
    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        save_dir: str,
    ) -> None:
        """ Evaluate method

        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
        """
        smplx_geometry = SMPLXGeometry(os.path.join(
            dataloader.dataset.prox_dir, 'body_segments'))
        contact_body_part_vid, _ = smplx_geometry.get_contact_id(
            ['back','gluteus','L_Hand','R_Hand','L_Leg','R_Leg','thighs']
        )
        
        model.eval()
        device = model.device

        res = defaultdict(list)
        res['ksample'] = self.ksample
        res['contact_threshold'] = self.contact_threshold
        res['eval_case'] = 0
        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            data['repr_type'] = dataloader.dataset.repr_type
            
            outputs = model.sample(data, k=self.ksample) # <B, k, T, ...>
            B = outputs.shape[0]
            scene_pos = data['pos'].reshape(B, -1, 3)

            for i in range(B):
                smplx_params = outputs[i, :, -1, :, :] # <k, T, D>

                ## diversity metrics
                ## 0. transl pairwise distance and standard deviation
                transl_np = smplx_params[:, :, 0:3].cpu().numpy()
                k, M, D = transl_np.shape
                transl_pdist = 0
                for j in range(M):
                    transl_pdist += pairwise_distances(
                        transl_np[:, j, :], transl_np[:, j, :], metric='l2').sum() / (k * (k - 1))
                transl_std = np.std(transl_np, axis=0).mean()

                res['transl_pdist'].append(float(transl_pdist) / M)
                res['transl_std'].append(float(transl_std))

                ## 1. smplx parameter pairwise distance and standard deviation
                smplx_params_np = smplx_params[:, :, 3:].cpu().numpy()
                k, M, D = smplx_params_np.shape
                param_pdist = 0
                for j in range(M):
                    param_pdist += pairwise_distances(
                        smplx_params_np[:, j, :], smplx_params_np[:, j, :], metric='l2').sum() / (k * (k - 1))
                param_std = np.std(smplx_params_np, axis=0).mean()

                res['param_pdist'].append(float(param_pdist) / M)
                res['param_std'].append(float(param_std))

                ## 2. global body marker pairwise distance and standard deviation
                smplx_params_local = smplx_params.clone()
                smplx_params_local[:, :, 0:3] = 0
                body_verts_local, body_faces, body_joints_local = dataloader.dataset.SMPLX.run(smplx_params_local)
                body_verts_local_np = body_verts_local.cpu().numpy()
                k, M, V, D = body_verts_local_np.shape
                body_marker = body_verts_local_np.reshape(k, M, V, D)[:, :, get_marker_indices(), :]
                body_marker = body_marker.reshape(k, M, -1) # <k, M, M' * 3>, concatenation of M' marker coordinates
                marker_pdist = 0
                for j in range(M):
                    marker_pdist += pairwise_distances(
                        body_marker[:, j, :], body_marker[:, j, :], metric='l2').sum() / (k * (k - 1))
                marker_std = np.std(body_marker, axis=0).mean()

                res['marker_pdist'].append(float(marker_pdist) / M)
                res['marker_std'].append(float(marker_std))

                ## physics metrics
                ## non-collision score and contact score
                body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params)
                non_collision_score = []
                contact_score = []
                scene_verts = scene_pos[i].unsqueeze(0)
                body_verts_tensor = body_verts.to(device)
                body_faces_tensor = torch.tensor(body_faces.astype(np.int64)).to(device)
                k, M, V, D = body_verts_tensor.shape
                for j in range(k):
                    non_collision_score_sequence = []
                    contact_score_sequence = []
                    for f in range(M):
                        scene_to_human_sdf, _ = smplx_signed_distance(
                            object_points=scene_verts,
                            smplx_vertices=body_verts_tensor[j, f:f+1],
                            smplx_face=body_faces_tensor
                        ) # <B, O> = D(<B, O, 3>, <B, H, 3>)
                        sdf = scene_to_human_sdf.cpu().numpy() # <1, O>
                        non_collision = np.sum(sdf <= 0) / sdf.shape[-1]
                        non_collision_score_sequence.append(non_collision)
                        
                        body_verts_contact = body_verts_tensor[j, f:f+1, contact_body_part_vid, :]
                        dist1, dist2, idx1, idx2 = chamfer_dist(
                            body_verts_contact.contiguous(), 
                            scene_verts.contiguous()
                        )
                        if torch.sum(dist1 < self.contact_threshold) > 0:
                            contact = 1.0
                        else:
                            contact = 0.0
                        contact_score_sequence.append(contact)
                    
                    non_collision_score.append(sum(non_collision_score_sequence) / M)
                    contact_score.append(sum(contact_score_sequence) / M)

                res['non_collision'].append(sum(non_collision_score) / len(non_collision_score))
                res['contact'].append(sum(contact_score) / len(contact_score))
                
                res['eval_case'] += 1

            ## only evaluate self.eval_case cases for saving time
            if res['eval_case'] >= self.eval_case_num:
                break
        
        for key in ['transl_pdist', 'transl_std', 'param_pdist', 'param_std', 'marker_pdist', 'marker_std', 'non_collision', 'contact']:
            res[key+'_average'] = sum(res[key]) / len(res[key])
        
        import json
        save_path = os.path.join(save_dir, 'metrics.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as fp:
            json.dump(res, fp)

def create_evaluator(cfg: DictConfig) -> nn.Module:
    """ Create a evaluator for quantitative evaluation
    Args:
        cfg: configuration object
        slurm: on slurm platform or not. This field is used to specify the data path
    
    Return:
        A evaluator
    """
    return EVALUATOR.get(cfg.name)(cfg)
