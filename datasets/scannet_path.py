from typing import Any, Dict, Tuple
import os
import glob
import pickle
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader

from datasets.transforms import make_default_transform
from datasets.normalize import NormaizerPathPlanning
from datasets.base import DATASET

@DATASET.register()
class ScanNetPath(Dataset):
    """ Dataset for path planning, constructed from ScanNet
    """

    def __init__(self, cfg: DictConfig, phase: str, slurm: bool, case_only: bool=False, **kwargs: Dict) -> None:
        super(ScanNetPath, self).__init__()
        self.phase = phase
        self.slurm = slurm
        self.data_dir = cfg.data_dir_slurm if self.slurm else cfg.data_dir
        self._load_split()

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
        
        self.num_points = cfg.num_points
        self.use_color = cfg.use_color
        self.use_normal = cfg.use_normal
        self.horizon = cfg.horizon
        self.frame_interval = cfg.frame_interval # interval sampling

        self.normalizer = None
        self.repr_type = cfg.repr_type
        if cfg.use_normalize:
            self.normalizer = NormaizerPathPlanning(cfg.normalize_cfg[self.repr_type])

        self.transform = make_default_transform(cfg, phase)

        ## load data
        self._pre_load_data(case_only)
    
    def _load_split(self) -> None:
        """ Load train and test split from scan id """
        self._train_split = []
        self._test_split = []
        self._all_split = []

        files = glob.glob(os.path.join(self.data_dir, 'path', '*.pkl'))
        for f in files:
            scan_id = f.split('/')[-1][0:-4]
            if int(scan_id[5:9]) < 600:
                self._train_split.append(scan_id)
            else:
                self._test_split.append(scan_id)
            self._all_split.append(scan_id)

    def _pre_load_data(self, case_only: bool) -> None:
        self.scene_pcds = {}
        self.scene_height = {}
        self.paths = []

        ## load preprocessed scene point cloud
        ## load height map
        ## load paths
        for s in self.split:
            scene_pcd = np.load(os.path.join(self.data_dir, 'scene', f'{s}.npy'))
            self.scene_pcds[s] = scene_pcd.astype(np.float32)
        
            with open(os.path.join(self.data_dir, 'height', f'{s}.pkl'), 'rb') as fp:
                self.scene_height[s] = pickle.load(fp)
            
            with open(os.path.join(self.data_dir, 'path', f'{s}.pkl'), 'rb') as fp:
                paths = pickle.load(fp)
                for coarse_path, refined_path in paths:
                    if refined_path.shape[-1] == 2:
                        refined_path = np.concatenate([refined_path, np.zeros((len(refined_path), 1))], axis=-1) # expand the z dim
                    self.paths.append((s, refined_path))
        
        ## segment path to fixed horizon for training
        self.indices = []
        for i, path in enumerate(self.paths):
            s, refined_path = path

            max_start = len(refined_path) - self.horizon
            if max_start < 0:
                continue

            ## only load case instead of training segments
            ## this is used for evaluation
            if case_only:
                self.indices.append((i, 0, self.horizon, refined_path[-1]))
                continue

            for start in range(0, max_start + 1, self.frame_interval):
                end = start + self.horizon
                self.indices.append((i, start, end, refined_path[-1])) # <path_idx, start_idx, end_idx, target_pos>
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index: Any) -> Tuple:
        path_idx, start, end, target = self.indices[index]
        scan_id, path = self.paths[path_idx]

        ## load path segment
        path_seg = path[start: end]

        ## load data
        scene_pc = self.scene_pcds[scan_id]
        height = self.scene_height[scan_id]
        s_grid_map = height['height'].astype(np.float32)
        s_grid_min = np.array([height['minx'], height['miny']], dtype=np.float32)
        s_grid_max = np.array([height['maxx'], height['maxy']], dtype=np.float32)
        s_grid_dim = height['dim']

        ## randomly resample points
        if self.phase != 'train':
            np.random.seed(0) # resample point cloud with a fixed random seed
        idx = np.random.permutation(len(scene_pc))
        scene_pc = scene_pc[idx[:self.num_points]]

        ## format point cloud xyz and feature
        xyz = scene_pc[:, 0:3]
        feat = scene_pc[:, 3:3] # empty array with shape <N, 0>

        if self.use_color:
            color = scene_pc[:, 3:6] / 255
            feat = np.concatenate([feat, color], axis=-1)
        
        if self.use_normal:
            normal = scene_pc[:, 6:9]
            feat = np.concatenate([feat, normal], axis=-1)
            ## TODO add normal rotation in RandomRotation transform
        
        data = {
            'x': path_seg,
            'target': target,
            'pos': xyz,
            'feat': feat,
            'scene_id': scan_id,
            'trans_mat': np.eye(4),
            's_grid_map': s_grid_map,
            's_grid_min': s_grid_min,
            's_grid_max': s_grid_max,
            's_grid_dim': s_grid_dim,
            'traj_length': len(path),
        }

        if self.transform is not None:
            data = self.transform(data, repr_type=self.repr_type, normalizer=self.normalizer)
        
        return data
    
    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)
    