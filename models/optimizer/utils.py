from typing import Dict, List, Tuple
import os
import glob
import json
import smplx
import torch
import torch.nn.functional as F
import numpy as np

from utils.plot import singleton

class SMPLXLayer():
    """ A SMPLX layer used in neural work, because the original smplx_model doesn't support variable
    batch size, we implement this wrapper by recreate smplx body when the input's batch size is not 
    equal to the smplx model. The default gender is neutral and default batch size is 1.
    """

    def __init__(self, smplx_dir: str, device: str, num_pca_comps: int=12, batch_size: int=1) -> None:
        self.device = device
        self.smplx_dir = smplx_dir
        self.num_pca_comps = num_pca_comps
        self.batch_size = None
        self._create_smplx_model(batch_size)
    
    def _create_smplx_model(self, batch_size) -> None:
        """ Recreate smplx model if the required batch size is not satisfied

        Args:
            batch_size: the required batch size
        """
        if batch_size is None or self.batch_size != batch_size:
            self.body_model_neutral = smplx.create(
                self.smplx_dir, model_type='smplx',
                gender='neutral', ext='npz',
                num_pca_comps=self.num_pca_comps,
                create_global_orient=True,
                create_body_pose=True,
                create_betas=True,
                create_left_hand_pose=True,
                create_right_hand_pose=True,
                create_expression=True,
                create_jaw_pose=True,
                create_leye_pose=True,
                create_reye_pose=True,
                create_transl=True,
                batch_size=batch_size,
            ).to(device=self.device)

            self.batch_size = batch_size
    
    def run(self, torch_param: Dict) -> Tuple:
        """ Use smplx model to generate smplx body

        Args:
            param: smplx parameters, must be a dict and the element must be tensor, shape is <B, d>
            
        Return:
            Body mesh tuple, i.e., (vertices, faces, joints)
        """
        shape_pre = torch_param['transl'].shape[:-1]
        for key in torch_param:
            dim = torch_param[key].shape[-1]
            torch_param[key] = torch_param[key].reshape(-1, dim)
        
        self._create_smplx_model(torch_param['transl'].shape[0])

        output = self.body_model_neutral(return_verts=True, **torch_param)
        faces = self.body_model_neutral.faces

        vertices = output.vertices
        V, D = vertices.shape[-2:]
        vertices = vertices.reshape(*shape_pre, V, D)

        joints = output.joints
        J, D = joints.shape[-2:]
        joints = joints.reshape(*shape_pre, J, D)
        
        return vertices, faces, joints

@singleton
class SMPLXGeometry():

    def __init__(self, body_segments_dir: str) -> None:
        ## load contact part
        self.contact_verts_ids = {}
        self.contact_faces_ids = {}

        part_files = glob.glob(os.path.join(body_segments_dir, '*.json'))
        for pf in part_files:
            with open(pf, 'r') as f:
                part = pf.split('/')[-1][:-5]
                if part in ['body_mask']:
                    continue
                data = json.load(f)
                
                self.contact_verts_ids[part] = list(set(data["verts_ind"]))
                self.contact_faces_ids[part] = list(set(data["faces_ind"]))

    def get_contact_id(self, contact_body_part: List) -> Tuple:
        """ Get contact body part, i.e. vertices ids and faces ids

        Args:
            contact_body_part: contact body part list
        
        Return:
            Contact vertice index and faces index
        """
        verts_ids = []
        faces_ids = []
        for part in contact_body_part:
            verts_ids.append(self.contact_verts_ids[part])
            faces_ids.append(self.contact_faces_ids[part])

        verts_ids = np.concatenate(verts_ids)
        faces_ids = np.concatenate(faces_ids)

        return verts_ids, faces_ids

def extract_smplx(x: torch.Tensor, key: str) -> torch.Tensor:
    if key == 'trans':
        return x[..., 0:3]
    if key == 'orient':
        return x[..., 3:6]
    if key == 'betas':
        return x[..., 6:16]
    if key == 'body_pose':
        return x[..., 16:79]
    if key == 'left_hand_pose' and x.shape[-1] > 79:
        return x[..., 79:91]
    if key == 'right_hand_pose' and x.shape[-1] > 91:
        return x[..., 91:103]
    
    raise Exception('Unsupported key or dimension.')

def transform_verts(verts_batch: torch.Tensor, cam_ext_batch: torch.Tensor) -> torch.Tensor:
    """ Transform vertices in torch.Tensor format

    Args:
        verts_batch: vertices in batch
        cam_ext_batch: transformation matrix in batch
    
    Returns:
        Transformed vertices
    """
    verts_batch_homo = F.pad(verts_batch, (0,1), mode='constant', value=1)
    verts_batch_homo_transformed = torch.matmul(verts_batch_homo,
                                                cam_ext_batch.permute(0,2,1))

    verts_batch_transformed = verts_batch_homo_transformed[:,:,:-1]
    
    return verts_batch_transformed