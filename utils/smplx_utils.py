from typing import Any, Dict, List, Tuple
import numpy as np
import smplx
import torch
from pyquaternion import Quaternion as Q

from utils.plot import singleton

def convert_smplx_parameters_format(params: Any, target: str='tuple', keep_keys: List=None) -> Any:
    """ Convert smplx paramters among three different data type, i.e., tuple, np.array, dict.
    And return designated components accordining to `keep_keys`.
    The input params must contains ['transl', 'global_orient', 'betas', 'body_pose'], 
    optional components are ['left_hand_pose', 'right_hand_pose'].

    Each component has the following default dimention:
    - transl: 3
    - global_orient: 3
    - betas: 10
    - body_pose: 63
    - left_hand_pose: 12
    - right_hand_pose: 12

    Args:
        params: smplx parameters in any format
        target: target data format, can be tuple, array, dict

    Return:
        smplx parameters with designated data format
    """
    lh_pose = None
    rh_pose = None
    if isinstance(params, dict):
        trans = params['transl']
        orient = params['global_orient']
        betas = params['betas']
        body_pose = params['body_pose']
        if 'left_hand_pose' in params and 'right_hand_pose' in params:
            lh_pose = params['left_hand_pose']
            rh_pose = params['right_hand_pose']
    elif isinstance(params, (np.ndarray, torch.Tensor)):
        trans = params[..., 0:3]
        orient = params[..., 3:6]
        betas = params[..., 6:16]
        body_pose = params[..., 16:79]
        if params.shape[-1] > 79:
            lh_pose = params[..., 79:91]
            rh_pose = params[..., 91:103]
    elif isinstance(params, tuple):
        trans, orient, betas, body_pose, *hand_pose = params
        if len(hand_pose) != 0:
            lh_pose, rh_pose = hand_pose
    else:
        raise Exception('Unsupported smplx data format.')
    
    assert target in ['tuple', 'array', 'dict'], "Unsupported target data format."

    
    if keep_keys is None:
        keep_keys = ['transl', 'global_orient', 'betas', 'body_pose']
        if lh_pose is not None and rh_pose is not None:
            keep_keys += ['left_hand_pose', 'right_hand_pose']
    
    ## return parameters according to keep_keys
    return_params = []
    if 'transl' in keep_keys:
        return_params.append(trans)
    if 'global_orient' in keep_keys:
        return_params.append(orient)
    if 'betas' in keep_keys:
        return_params.append(betas)
    if 'body_pose' in keep_keys:
        return_params.append(body_pose)
    if 'left_hand_pose' in keep_keys:
        return_params.append(lh_pose)
    if 'right_hand_pose' in keep_keys:
        return_params.append(rh_pose)


    if target == 'tuple':
        return return_params

    if target == 'array':
        if isinstance(return_params[0], np.ndarray):
            return np.concatenate(return_params, axis=-1)
        elif isinstance(return_params[0], torch.Tensor):
            return torch.cat(return_params, dim=-1)
        else:
            raise Exception('Unknown input smplx parameter dtype.')

    if target == 'dict':
        return {keep_keys[i]:return_params[i] for i in range(len(keep_keys))}

def get_smplx_dimension_from_keys(keys: List) -> int:
    """ Accumulating the dimension of smplx parameters from keys

    Args:
        keys: the designated keys
    
    Return:
        The accumulated dimension.
    """
    key_dim = {
        'transl': 3,
        'global_orient': 3,
        'betas': 10,
        'body_pose': 63,
        'left_hand_pose': 12,
        'right_hand_pose': 12
    }

    dim = 0
    for key in keys:
        dim += key_dim[key]
    return dim

def convert_smplx_verts_transfomation_matrix_to_body(
    T: np.ndarray, trans: np.ndarray, orient: np.ndarray, pelvis: np.ndarray):
    """ Convert transformation to smplx trans and orient

    Reference: https://www.dropbox.com/scl/fi/zkatuv5shs8d4tlwr8ecc/Change-parameters-to-new-coordinate-system.paper?dl=0&rlkey=lotq1sh6wzkmyttisc05h0in0

    Args:
        T: target transformation matrix
        trans: origin trans of smplx parameters
        orient: origin orient of smplx parameters
        pelvis: origin pelvis
    
    Return:
        Transformed trans and orient smplx parameters
    """
    R = T[0:3, 0:3]
    t = T[0:3, -1]

    pelvis = pelvis - trans
    trans = np.matmul(R, trans + pelvis) - pelvis
    orient = np.matmul(R, Q(axis=orient/np.linalg.norm(orient), angle=np.linalg.norm(orient)).rotation_matrix)
    try:
        orient = Q(matrix=orient, rtol=1e-05, atol=1e-06)
    except:
        print(np.dot(orient, orient.conj().transpose()))
        exit(0)
    orient = orient.axis * orient.angle
    return trans + t, orient

@singleton
class SMPLXWrapper():
    """ A SMPLX model wrapper written with singleton
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
            self.body_model_male = smplx.create(
                self.smplx_dir, model_type='smplx',
                gender='male', ext='npz',
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

            self.body_model_female = smplx.create(
                self.smplx_dir, model_type='smplx',
                gender='female', ext='npz',
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

    def run(self, param: Any, gender: str='male') -> Tuple:
        """ Use smplx model to generate smplx body

        Args:
            param: smplx parameters, the element must be tensor.
            gender: the subject gender
        
        Return:
            Body mesh tuple, i.e., (vertices, faces, joints)
        """
        if isinstance(param, dict):
            param_dict = param
        else:
            param_dict = convert_smplx_parameters_format(param, 'dict')
        return self._forward(param_dict, gender)
    
    def _forward(self, torch_param: Dict, gender: str='male') -> Tuple:
        """ Use smplx model to generate smplx body

        Args:
            param: smplx parameters, must be a dict and the element must be tensor.
            gender: the subject gender
        
        Return:
            Body mesh tuple, i.e., (vertices, faces, joints)
        """
        shape_pre = torch_param['transl'].shape[:-1]

        for key in torch_param:
            fdim = torch_param[key].shape[-1]
            torch_param[key] = torch_param[key].reshape(-1, fdim).to(self.device)
        
        self._create_smplx_model(torch_param['transl'].shape[0])

        if gender == 'male':
            output = self.body_model_male(return_verts=True, **torch_param)
            faces = self.body_model_male.faces
        elif gender == 'female':
            output = self.body_model_female(return_verts=True, **torch_param)
            faces = self.body_model_female.faces
        else:
            raise Exception('Unsupported gender.')

        vertices = output.vertices.detach().cpu().reshape(*(*shape_pre, -1, 3))
        joints = output.joints.detach().cpu().reshape(*(*shape_pre, -1, 3))
        
        return vertices, faces, joints

def get_marker_indices():
    markers = {
        "gender": "unknown",
        "markersets": [
            {
                "distance_from_skin": 0.0095,
                "indices": {
                    "C7": 3832,
                    "CLAV": 5533,
                    "LANK": 5882,
                    "LFWT": 3486,
                    "LBAK": 3336,
                    "LBCEP": 4029,
                    "LBSH": 4137,
                    "LBUM": 5694,
                    "LBUST": 3228,
                    "LCHEECK": 2081,
                    "LELB": 4302,
                    "LELBIN": 4363,
                    "LFIN": 4788,
                    "LFRM2": 4379,
                    "LFTHI": 3504,
                    "LFTHIIN": 3998,
                    "LHEE": 8846,
                    "LIWR": 4726,
                    "LKNE": 3682,
                    "LKNI": 3688,
                    "LMT1": 5890,
                    "LMT5": 5901,
                    "LNWST": 3260,
                    "LOWR": 4722,
                    "LBWT": 5697,
                    "LRSTBEEF": 5838,
                    "LSHO": 4481,
                    "LTHI": 4088,
                    "LTHMB": 4839,
                    "LTIB": 3745,
                    "LTOE": 5787,
                    "MBLLY": 5942,
                    "RANK": 8576,
                    "RFWT": 6248,
                    "RBAK": 6127,
                    "RBCEP": 6776,
                    "RBSH": 7192,
                    "RBUM": 8388,
                    "RBUSTLO": 8157,
                    "RCHEECK": 8786,
                    "RELB": 7040,
                    "RELBIN": 7099,
                    "RFIN": 7524,
                    "RFRM2": 7115,
                    "RFRM2IN": 7303,
                    "RFTHI": 6265,
                    "RFTHIIN": 6746,
                    "RHEE": 8634,
                    "RKNE": 6443,
                    "RKNI": 6449,
                    "RMT1": 8584,
                    "RMT5": 8595,
                    "RNWST": 6023,
                    "ROWR": 7458,
                    "RBWT": 8391,
                    "RRSTBEEF": 8532,
                    "RSHO": 6627,
                    "RTHI": 6832,
                    "RTHMB": 7575,
                    "RTIB": 6503,
                    "RTOE": 8481,
                    "STRN": 5531,
                    "T8": 5487,
                    "LFHD": 707,
                    "LBHD": 2026,
                    "RFHD": 2198,
                    "RBHD": 3066
                },
                "marker_radius": 0.0095,
                "type": "body"
            }
        ]
    }

    marker_indic = list(markers['markersets'][0]['indices'].values())

    return marker_indic

def smplx_signed_distance(object_points, smplx_vertices, smplx_face):
    """ Compute signed distance between query points and mesh vertices.
    
    Args:
        object_points: (B, O, 3) query points in the mesh.
        smplx_vertices: (B, H, 3) mesh vertices.
        smplx_face: (F, 3) mesh faces.
    
    Return:
        signed_distance_to_human: (B, O) signed distance to human vertex on each object vertex
        closest_human_points: (B, O, 3) closest human vertex to each object vertex
        signed_distance_to_obj: (B, H) signed distance to object vertex on each human vertex
        closest_obj_points: (B, H, 3) closest object vertex to each human vertex
    """
    # compute vertex normals
    smplx_face_vertices = smplx_vertices[:, smplx_face]
    e1 = smplx_face_vertices[:, :, 1] - smplx_face_vertices[:, :, 0]
    e2 = smplx_face_vertices[:, :, 2] - smplx_face_vertices[:, :, 0]
    e1 = e1 / torch.norm(e1, dim=-1, p=2).unsqueeze(-1)
    e2 = e2 / torch.norm(e2, dim=-1, p=2).unsqueeze(-1)
    smplx_face_normal = torch.cross(e1, e2)     # (B, F, 3)

    # compute vertex normal
    smplx_vertex_normals = torch.zeros(smplx_vertices.shape).float().cuda()
    smplx_vertex_normals.index_add_(1, smplx_face[:,0], smplx_face_normal)
    smplx_vertex_normals.index_add_(1, smplx_face[:,1], smplx_face_normal)
    smplx_vertex_normals.index_add_(1, smplx_face[:,2], smplx_face_normal)
    smplx_vertex_normals = smplx_vertex_normals / torch.norm(smplx_vertex_normals, dim=-1, p=2).unsqueeze(-1)

    # compute paired distance of each query point to each face of the mesh
    pairwise_distance = torch.norm(object_points.unsqueeze(2) - smplx_vertices.unsqueeze(1), dim=-1, p=2)    # (B, O, H)
    
    # find the closest face for each query point
    distance_to_human, closest_human_points_idx = pairwise_distance.min(dim=2)  # (B, O)
    closest_human_point = smplx_vertices.gather(1, closest_human_points_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, O, 3)
    query_to_surface = closest_human_point - object_points
    query_to_surface = query_to_surface / torch.norm(query_to_surface, dim=-1, p=2).unsqueeze(-1)
    closest_vertex_normals = smplx_vertex_normals.gather(1, closest_human_points_idx.unsqueeze(-1).repeat(1, 1, 3))
    same_direction = torch.sum(query_to_surface * closest_vertex_normals, dim=-1)
    signed_distance_to_human = same_direction.sign() * distance_to_human    # (B, O)
    
    # find signed distance to object from human
    # signed_distance_to_object = torch.zeros([pairwise_distance.shape[0], pairwise_distance.shape[2]]).float().cuda()-10  # (B, H)
    # signed_distance_to_object, closest_obj_points_idx = torch_scatter.scatter_max(signed_distance_to_human, closest_human_points_idx, out=signed_distance_to_object)
    # closest_obj_points_idx[closest_obj_points_idx == pairwise_distance.shape[1]] = 0
    # closest_object_point = object_points.gather(1, closest_obj_points_idx.unsqueeze(-1).repeat(1,1,3))
    # return signed_distance_to_human, closest_human_point, signed_distance_to_object, closest_object_point, smplx_vertex_normals
    return signed_distance_to_human, closest_human_point