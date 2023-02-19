from typing import Any, Tuple, Dict
import os
import json
import glob
from tqdm import tqdm
import pickle
import trimesh
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig

from utils.smplx_utils import convert_smplx_verts_transfomation_matrix_to_body
from utils.smplx_utils import SMPLXWrapper
from datasets.transforms import make_default_transform
from datasets.normalize import NormalizerPoseMotion
from datasets.base import DATASET

@DATASET.register()
class LEMOPose(Dataset):
    """ Dataset for pose generation, training with LEMO dataset
    """

    _train_split = ['BasementSittingBooth', 'MPH11', 'MPH112', 'MPH8', 'N0Sofa', 'N3Library', 'N3Office', 'Werkraum']
    _test_split = ['MPH16', 'MPH1Library', 'N0SittingBooth', 'N3OpenArea']
    _all_split = ['BasementSittingBooth', 'MPH11', 'MPH112', 'MPH8', 'N0Sofa', 'N3Library', 'N3Office', 'Werkraum', 
    'MPH16', 'MPH1Library', 'N0SittingBooth', 'N3OpenArea']
    # _train_split = ['BasementSittingBooth']
    # _test_split = ['MPH16']

    _female_subjects_ids = [162, 3452, 159, 3403]

    def __init__(self, cfg: DictConfig, phase: str, slurm: bool, case_only: bool=False, **kwargs: Dict) -> None:
        super(LEMOPose, self).__init__()
        self.phase = phase
        self.slurm = slurm
        if 'specific_scene' in kwargs:
            self.split = [kwargs['specific_scene']]
        else:
            if self.phase == 'train':
                self.split = self._train_split
            elif self.phase == 'test':
                self.split = self._test_split
            elif self.phase == 'all':
                self.split = self._all_split
            else:
                raise Exception('Unsupported phase.')
        self.frame_interval = cfg.frame_interval_train if self.phase == 'train' else cfg.frame_interval_test # interval sampling
        self.modeling_keys = cfg.modeling_keys
        self.num_points = cfg.num_points
        self.use_color = cfg.use_color
        self.use_normal = cfg.use_normal
        self.transform = make_default_transform(cfg, phase)

        ## resource folders
        self.data_dir = cfg.data_dir_slurm if self.slurm else cfg.data_dir
        self.smpl_dir = cfg.smpl_dir_slurm if self.slurm else cfg.smpl_dir
        self.prox_dir = cfg.prox_dir_slurm if self.slurm else cfg.prox_dir
        self.prox_scene_ply = os.path.join(self.prox_dir, 'scenes')
        self.prox_scene_npy = os.path.join(self.prox_dir, 'preprocess_scenes')
        self.prox_scene_sdf = os.path.join(self.prox_dir, 'sdf')
        self.prox_cam2world = os.path.join(self.prox_dir, 'cam2world')

        self.SMPLX = SMPLXWrapper(self.smpl_dir, cfg.smplx_model_device, cfg.smplx_pca_comps) # singleton

        self.normalizer = None
        self.repr_type = 'absolute'
        if cfg.use_normalize:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(cur_dir, 'lemo/normalization.pkl'), 'rb') as fp:
                data = pickle.load(fp)
            xmin = data['xmin'].astype(np.float32)
            xmax = data['xmax'].astype(np.float32)
            self.normalizer = NormalizerPoseMotion((xmin, xmax))
        ## load data
        self._pre_load_data(case_only)

    def _pre_load_data(self, case_only: bool) -> None:
        """ Load dataset

        Args:
            case_only: only load single case for testing, if ture, the dataset will be smaller.
                        This is useful in after-training visual evaluation.
        """
        self.scene_meshes = {}
        self.scene_pcds = {}
        self.scene_sdf = {}
        self.cam_trans = {}
        self.frames = []
        
        ## load original mesh
        ## load preprocessed scene point cloud
        ## load camera transformation
        for s in self.split:
            scene_mesh = trimesh.load(os.path.join(self.prox_scene_ply, s + '.ply'))
            self.scene_meshes[s] = scene_mesh
            
            scene_pcd = np.load(os.path.join(self.prox_scene_npy, s + '.npy'))
            self.scene_pcds[s] = scene_pcd.astype(np.float32)

            with open(os.path.join(self.prox_scene_sdf, s + '.json')) as f:
                sdf_data = json.load(f)
                grid_min = np.array(sdf_data['min'], dtype=np.float32)
                grid_max = np.array(sdf_data['max'], dtype=np.float32)
                grid_dim = sdf_data['dim']
            grid_sdf = np.load(os.path.join(self.prox_scene_sdf, s + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
            self.scene_sdf[s] = {'grid_min': grid_min, 'grid_max': grid_max, 'grid_dim': grid_dim, 'grid_sdf': grid_sdf}
            
            with open(os.path.join(self.prox_cam2world, s + '.json'), 'r') as f:
                trans = np.array(json.load(f))
            self.cam_trans[s] = trans.astype(np.float32)
            
        ## load frames of all available sequences
        scene_loaded = {s: False for s in self.split}
        dirs = os.listdir(self.data_dir)
        for record_id in tqdm(dirs):
            record_dir = os.path.join(self.data_dir, record_id)
            if not os.path.isdir(record_dir):
                continue
            
            scene_id, subject_id, _ = record_id.split('_')
            if scene_id not in self.split:
                continue

            if subject_id in self._female_subjects_ids:
                subject_gender = 'female'
            else:
                subject_gender = 'male'

            pkls = sorted(glob.glob(os.path.join(record_dir, 'results', '*', '000.pkl')))
            for pkl in pkls[::self.frame_interval]:
                if not os.path.exists(pkl):
                    continue
                
                # only load scenes, only load one frame
                if case_only and scene_loaded[scene_id]:
                    continue
                scene_loaded[scene_id] = True

                with open(pkl, 'rb') as fp:
                    ## keys: ['pose_embedding', 'camera_rotation', 'camera_translation', 'betas', 
                    ## 'global_orient', 'transl', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 
                    ## 'leye_pose', 'reye_pose', 'expression', 'body_pose']
                    param = pickle.load(fp)

                frame_name = pkl.split('/')[-2]
                frame_num = int(frame_name.split('_')[2])
                param_keep = {'record': record_id, 'scene': scene_id, 'frame': frame_num, 'gender': subject_gender}
                torch_param = {}
                for key in param:
                    if key not in ['pose_embedding', 'camera_rotation', 'camera_translation']:
                        param_keep[key] = param[key].squeeze(axis=0) # <FDim>
                        torch_param[key] = torch.tensor(param[key]) # <1, FDim>
                
                ## We fix the scene and transform the smplx body with the camera transformation matrix,
                ## which is different from the PROX official code tranforming scenes.
                ## So we first need to compute the body pelvis location, see more demonstration at 
                ## https://www.dropbox.com/scl/fi/zkatuv5shs8d4tlwr8ecc/Change-parameters-to-new-coordinate-system.paper?dl=0&rlkey=lotq1sh6wzkmyttisc05h0in0
                _, _, joints = self.SMPLX.run(torch_param, subject_gender)
                pelvis = joints[:, 0, :].numpy()

                cur_transl, cur_global_orient = convert_smplx_verts_transfomation_matrix_to_body(
                    self.cam_trans[scene_id],
                    param['transl'].squeeze(axis=0),
                    param['global_orient'].squeeze(axis=0),
                    pelvis.squeeze(axis=0)
                )

                param_keep['cur_transl'] = cur_transl.astype(np.float32)
                param_keep['cur_global_orient'] = cur_global_orient.astype(np.float32)
                param_keep['pelvis'] = pelvis.astype(np.float32).squeeze(axis=0)

                self.frames.append(param_keep)
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index: Any) -> Tuple:
        frame = self.frames[index]

        ## load data, containing scene point cloud and point pose
        scene_id = frame['scene']
        scene_pc = self.scene_pcds[scene_id]
        scene_sdf_data = self.scene_sdf[scene_id]
        scene_grid_min = scene_sdf_data['grid_min']
        scene_grid_max = scene_sdf_data['grid_max']
        scene_grid_dim = scene_sdf_data['grid_dim']
        scene_grid_sdf = scene_sdf_data['grid_sdf']
        cam_tran = self.cam_trans[scene_id]

        ## randomly resample points
        if self.phase != 'train':
            np.random.seed(0) # resample point cloud with a fixed random seed
        idx = np.random.permutation(len(scene_pc))
        scene_pc = scene_pc[idx[:self.num_points]]

        ## format point cloud xyz and feature
        xyz = scene_pc[:, 0:3]
        feat = scene_pc[:, 3:3]

        if self.use_color:
            color = scene_pc[:, 3:6] / 255.
            feat = np.concatenate([feat, color], axis=-1)

        if self.use_normal:
            normal = scene_pc[:, 6:9]
            feat = np.concatenate([feat, normal], axis=-1)
        
        ## format smplx parameters
        smplx_param = (
            frame['cur_transl'],
            frame['cur_global_orient'],
            frame['betas'],
            frame['body_pose'],
            frame['left_hand_pose'],
            frame['right_hand_pose']
        )
        
        data = {
            'x': smplx_param, 
            'pos': xyz, 
            'feat': feat, 
            'cam_tran': cam_tran, 
            'scene_id': scene_id, 
            'gender': frame['gender'], 
            'origin_cam_tran': cam_tran, 
            'origin_pelvis': frame['pelvis'],
            'origin_transl': frame['transl'],
            'origin_global_orient': frame['global_orient'],
            's_grid_sdf': scene_grid_sdf,
            's_grid_min': scene_grid_min,
            's_grid_max': scene_grid_max,
            's_grid_dim': scene_grid_dim,
        }

        if self.transform is not None:
            data = self.transform(data, modeling_keys=self.modeling_keys, normalizer=self.normalizer)

        return data

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    p = '/home/wangzan/Data/LEMO/PROX_temporal/PROX_temporal/PROXD_temp/BasementSittingBooth_00142_01/results/s001_frame_00001__00.00.00.029/000.pkl'
    with open(p, 'rb') as fp:
        param = pickle.load(fp)
    
    print(param.keys())
    for key in param:
        if isinstance(param[key], np.ndarray):
            print(key, param[key].shape, param[key].dtype)
        else:
            print(key, param[key])
