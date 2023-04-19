from typing import Any, Dict, List
import torch
import numpy as np

from utils.smplx_utils import convert_smplx_parameters_format
from utils.smplx_utils import convert_smplx_verts_transfomation_matrix_to_body
from trimesh import transform_points

class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms: Any) -> None:
        self.transforms = transforms

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> Any: 
        for t in self.transforms:
            args = t(data, *args, **kwargs)
        return args

## transforms for PROXPose and PROXMotion
class NumpyToTensor(object):
    """ Convert numpy data to torch.Tensor data
    """
    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        for key in data.keys():
            if key in ['x', 'pos', 'feat', 's_grid_sdf', 's_grid_min', 's_grid_max', 'start', 'end'] and not torch.is_tensor(data[key]):
                data[key] = torch.tensor(np.array(data[key]))
        
        return data

class SMPLXTupleToArray(object):
    """ Convert tuple-formated smplx parameters to array-formated
    """
    def __init__(self, **kwargs) -> None:
        if 'observation_frame' in kwargs:
            self.observation_frame = kwargs['observation_frame']

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        modeling_keys = kwargs['modeling_keys'] if 'modeling_keys' in kwargs else None
        data['x'] = convert_smplx_parameters_format(data['x'], 'array', modeling_keys)

        ## convert repr type
        if 'repr_type' in kwargs:
            if kwargs['repr_type'] == 'absolute':
                pass # original trajectory position
            elif kwargs['repr_type'] == 'relative':
                x_expand = np.concatenate([
                    data['x'][0:self.observation_frame], 
                    data['x'][self.observation_frame-1:-1]
                ], axis=0) # [x_0, ... , x_o, x_o, ..., x_{n-1}]
                data['x'] = data['x'] - x_expand
                data['x'][0:self.observation_frame] = data['start'].copy() # keep the observation frame as absolute repr
            else:
                raise Exception('Unsupported repr type.')

        ## normalize
        if 'normalizer' in kwargs and kwargs['normalizer'] is not None:
            normalizer = kwargs['normalizer']
            data['x'] = normalizer.normalize(data['x'])
        
        ## copy start observation
        if 'motion' in kwargs and kwargs['motion']:
            data['start'] = data['x'][0:self.observation_frame].copy()

        return data

class RandomRotation(object):
    """ Random rotation augmentation
    """
    def __init__(self, **kwargs) -> None:
        self.angle = kwargs['angle']

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]], dtype=np.float32)
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=np.float32)
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]], dtype=np.float32)
        trans_mat = np.eye(4, dtype=np.float32)
        trans_mat[0:3, 0:3] = np.dot(R_z, np.dot(R_y, R_x))

        data['cam_tran'] = trans_mat @ data['cam_tran']
        return data

class NormalizeToCenter(object):
    """ Normalize scene to center
    """
    def __init__(self, **kwargs) -> None:
        self.gravity_dim = kwargs['gravity_dim']

    def __call__(self, data: Dict, *args: Dict, **kwargs: Dict) -> dict:
        ## scene point cloud is transformed step by step
        xyz = data['pos']
        center = (xyz.max(axis=0) + xyz.min(axis=0)) * 0.5
        center[self.gravity_dim] = np.percentile(xyz[:, self.gravity_dim], 1)
        trans_mat = np.eye(4, dtype=np.float32)
        trans_mat[0:3, -1] -= center

        data['cam_tran'] = trans_mat @ data['cam_tran']
        return data

class ApplyTransformationPROXPose(object):
    """ Apply transformation for single pose
    """
    def __init__(self, **kwargs) -> None:
        pass
    
    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        cam_tran = data['cam_tran']
        origin_cam_tran = data['origin_cam_tran']
        trans_mat = cam_tran @ np.linalg.inv(origin_cam_tran) # scene_T @ origin_cam_T = cur_cam_T

        ## scene point cloud is transformed
        data['pos'] = transform_points(data['pos'], trans_mat).astype(np.float32)

        ## smplx body is transformed from original smplx parameters
        cur_transl, cur_global_orient, *rest = data['x']
        cur_transl, cur_global_orient = convert_smplx_verts_transfomation_matrix_to_body(
            cam_tran,
            data['origin_transl'],
            data['origin_global_orient'],
            data['origin_pelvis'],
        )
        
        data['x'] = (cur_transl.astype(np.float32), cur_global_orient.astype(np.float32), *rest)

        return data

class ApplyTransformationPROXMotion(object):
    """ Apply transformation for motion
    """
    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        cam_tran = data['cam_tran']
        origin_cam_tran = data['origin_cam_tran']
        trans_mat = cam_tran @ np.linalg.inv(origin_cam_tran) # scene_T @ origin_cam_T = cur_cam_T

        ## scene point cloud is transformed
        data['pos'] = transform_points(data['pos'], trans_mat).astype(np.float32)

        ## smplx body is transformed from original smplx parameters
        cur_transl, cur_global_orient, *rest = data['x']
        cur_transl_, cur_global_orient_ = [], []
        for i in range(len(cur_transl)):
            t, o = convert_smplx_verts_transfomation_matrix_to_body(
                cam_tran,
                data['origin_transl'][i],
                data['origin_global_orient'][i],
                data['origin_pelvis'][i],
            )
            cur_transl_.append(t)
            cur_global_orient_.append(o)
        
        data['x'] = (
            np.array(cur_transl_).astype(np.float32),
            np.array(cur_global_orient_).astype(np.float32),
            *rest
        )

        return data

class NumpyToTensorPath(object):
    """ Convert numpy data to torch.Tensor data
    """
    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> Dict:
        for key in data.keys():
            if key in ['x', 'start', 'target', 'pos', 'feat', 's_grid_map', 's_grid_dim', 's_grid_min', 's_grid_max'] and not torch.is_tensor(data[key]):
                data[key] = torch.tensor(np.array(data[key]), dtype=torch.float32)
        
        return data

class NormalizeToCenterPath(object):
    """ Normalize scene to center
    """

    def __init__(self, **kwargs) -> None:
        self.gravity_dim = kwargs['gravity_dim']

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> Dict:
        ## transform point cloud step by step
        xyz = data['pos']
        center = (xyz.max(axis=0) + xyz.min(axis=0)) * 0.5
        center[self.gravity_dim] = np.percentile(xyz[:, self.gravity_dim], 1)
        trans_mat = np.eye(4, dtype=np.float32)
        trans_mat[0:3, -1] -= center
        data['pos'] = transform_points(xyz, trans_mat).astype(np.float32)
        data['x'] = transform_points(data['x'], trans_mat).astype(np.float32)
        data['target'] = transform_points(data['target'][None, :], trans_mat).astype(np.float32)[0]
        data['trans_mat'] = trans_mat @ data['trans_mat']

        return data

class RandomRotationPath(object):
    """ Random rotation augmentation
    """
    def __init__(self, **kwargs) -> None:
        self.angle = kwargs['angle']
    
    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> Dict:
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]], dtype=np.float32)
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=np.float32)
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]], dtype=np.float32)
        trans_mat = np.eye(4, dtype=np.float32)
        trans_mat[0:3, 0:3] = np.dot(R_z, np.dot(R_y, R_x))

        ## transform point cloud step by step
        data['pos'] = transform_points(data['pos'], trans_mat).astype(np.float32)
        data['x'] = transform_points(data['x'], trans_mat).astype(np.float32)
        data['target'] = transform_points(data['target'][None, :], trans_mat).astype(np.float32)[0]
        data['trans_mat'] = trans_mat @ data['trans_mat']

        return data

class ProjectTo2DPath(object):
    """ Project 3D path to 2D
    """
    def __init__(self, **kwargs) -> None:
        self.project_dim = np.array([True, True, True])
        self.project_dim[kwargs['gravity_dim']] = False
    
    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> Dict:
        data['x'] = data['x'][:, self.project_dim]
        data['target'] = data['target'][self.project_dim]

        return data

class CreatePlanningDataPath(object):
    """ Convert path to observation and action
    """
    def __init__(self, **kwargs) -> None:
        self.observation_frame = kwargs['observation_frame']

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> Dict:
        ## copy start observation from path
        data['start'] = data['x'][0:self.observation_frame].copy()

        ## convert repr type
        if 'repr_type' in kwargs:
            if kwargs['repr_type'] == 'absolute':
                pass # original trajectory position
            elif kwargs['repr_type'] == 'relative':
                x_expand = np.concatenate([
                    data['x'][0:self.observation_frame], 
                    data['x'][self.observation_frame-1:-1]
                ], axis=0) # [x_0, ... , x_o, x_o, ..., x_{n-1}]
                data['x'] = data['x'] - x_expand
            else:
                raise Exception('Unsupported repr type.')

        ## normalize
        if 'normalizer' in kwargs and kwargs['normalizer'] is not None:
            normalizer = kwargs['normalizer']
            data['x'] = normalizer.normalize(data['x'])

        return data

TRANSFORMS = {
    'NumpyToTensor': NumpyToTensor,
    'SMPLXTupleToArray': SMPLXTupleToArray,
    'RandomRotation': RandomRotation,
    'NormalizeToCenter': NormalizeToCenter,
    'ApplyTransformationPROXPose': ApplyTransformationPROXPose,
    'ApplyTransformationPROXMotion': ApplyTransformationPROXMotion,
    'NumpyToTensorPath': NumpyToTensorPath,
    'NormalizeToCenterPath': NormalizeToCenterPath,
    'RandomRotationPath': RandomRotationPath,
    'ProjectTo2DPath': ProjectTo2DPath,
    'CreatePlanningDataPath': CreatePlanningDataPath,
}

def make_default_transform(cfg: dict, phase: str) -> Compose:
    """ Make default transform

    Args:
        cfg: global configuration
        phase: process phase
    
    Return:
        Composed transforms.
    """
    ## generate transform configuration
    transform_cfg = {'phase': phase, **cfg.transform_cfg}

    ## compose
    transforms = []
    transforms_list = cfg.train_transforms if phase == 'train' else cfg.test_transforms
    for t in transforms_list:
        transforms.append(TRANSFORMS[t](**transform_cfg))

    return Compose(transforms)
