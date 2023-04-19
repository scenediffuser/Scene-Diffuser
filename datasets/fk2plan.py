import time
from typing import Any, Tuple, Dict
import os
import json
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf

from datasets.misc import collate_fn_general
from datasets.transforms import make_default_transform
from datasets.base import DATASET

@DATASET.register()
class FK2Plan(Dataset):
    """ Dataset for fk franka planning, training with FK2PLAN Dataset
    """

    _scene_pre_code = 'dthvl15jruz9i2fok6bsy3qamp8c4nex'
    _train_split = [f"dthvl15jruz9i2fok6bsy3qamp8c4nex{str(i).zfill(3)}" for i in range(160)]
    # _test_split = [f"dthvl15jruz9i2fok6bsy3qamp8c4nex{str(i).zfill(3)}" for i in range(160, 200)]
    _test_split = [f"dthvl15jruz9i2fok6bsy3qamp8c4nex{str(i).zfill(3)}" for i in range(160, 200)]  ##todo: here to test
    _all_split = [f"dthvl15jruz9i2fok6bsy3qamp8c4nex{str(i).zfill(3)}" for i in range(200)]
    # _test_split = [f"dthvl15jruz9i2fok6bsy3qamp8c4nex{str(i).zfill(3)}" for i in range(200)]

    _joint_angle_lower = np.array([-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671])
    _joint_angle_upper = np.array([2.9671, 1.8326, 2.9671, 0.0873, 2.9671, 3.8223, 2.9671])

    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.

    def __init__(self, cfg: DictConfig, phase: str, slurm: bool, case_only: bool=False, **kwargs: Dict) -> None:
        super(FK2Plan, self).__init__()
        self.phase = phase
        self.slurm = slurm
        if self.phase == 'train':
            self.split = self._train_split
        elif self.phase == 'test':
            self.split = self._test_split
        elif self.phase == 'all':
            self.split = self._all_split
        else:
            raise Exception('Unsupported phase.')
        self.device = cfg.device
        self.is_downsample = cfg.is_downsample
        self.modeling_keys = cfg.modeling_keys
        self.num_points = cfg.num_points
        self.use_color = cfg.use_color
        self.use_normal = cfg.use_normal
        self.normalize_x = cfg.normalize_x
        self.horizon = cfg.horizon
        self.frame_interval = cfg.frame_interval  # interval sampling
        self.sample_trajs_per_scene = cfg.sample_trajs_per_scene
        self.sample_frame_interval = cfg.sample_frame_interval
        self.planner_batch_size = cfg.planner_batch_size
        self.obj_dim = int(3 + 3 * self.use_color + 3 * self.use_normal)
        self.transform = make_default_transform(cfg, phase)

        ## resource folders
        self.data_dir = cfg.data_dir_slurm if self.slurm else cfg.data_dir
        with open(os.path.join(self.data_dir, 'desc.json'), 'r') as f:
            self.dataset_desc = json.load(f)

        ## load data
        self._pre_load_data(case_only)

    def _pre_load_data(self, case_only: bool) -> None:
        """ Load dataset
        Args:
            case_only: only load single case for testing, if ture, the dataset will be smaller.
                        This is useful in after-training visual evaluation.
        """
        self.trajectories = []
        self.indices = []
        self.scene_pcds_nors = {}

        fk2plan_dataset = pickle.load(open(os.path.join(self.data_dir, 'fk2plan_dataset.pkl'), 'rb'))
        self.scene_pcds_nors = pickle.load(open(os.path.join(self.data_dir, 'scene_pcds_nors.pkl'), 'rb'))
        # todo: length the scene points cloud
        num_points = np.max([a.shape[0] for a in list(self.scene_pcds_nors.values())])
        for scene_id in self.scene_pcds_nors.keys():
            if self.scene_pcds_nors[scene_id].shape[0] < num_points:
                self.scene_pcds_nors[scene_id] = np.concatenate([self.scene_pcds_nors[scene_id],
                                                                 self.scene_pcds_nors[scene_id][0:num_points-self.scene_pcds_nors[scene_id].shape[0]]], axis=0)
        self.dataset_info = fk2plan_dataset['info']
        self.dataset_desc = json.load(open(os.path.join(self.data_dir, 'desc.json'), 'r'))

        ## load paths
        for mdata in fk2plan_dataset['metadata']:
            mdata_scene_id = mdata[0]
            mdata_start_goal_pose = mdata[1]
            mdata_tra_qpos = mdata[2]
            if self.normalize_x:
                mdata_tra_qpos = self.angle_normalize(mdata_tra_qpos)
            if mdata_scene_id in self.split:
                self.trajectories.append((mdata_scene_id, mdata_start_goal_pose, mdata_tra_qpos))

        ## segment path to fixed horizon for training
        if case_only:
            loaded_counter = {s: 0 for s in self.split}
        for i, traj in enumerate(self.trajectories):
            mdata_scene_id, mdata_start_goal_pose, mdata_tra_qpos = traj
            max_start = mdata_tra_qpos.shape[0] - self.horizon
            if max_start <= 0:
                continue

            if case_only:
                loaded_counter[mdata_scene_id] += 1
                if loaded_counter[mdata_scene_id] > self.planner_batch_size:
                    continue
                # self.indices.append((i, 0, self.horizon + 1))
                # continue

            if case_only:
                self.indices.append((i, 0, self.horizon + 1))
            else:
                for start in range(0, max_start, self.frame_interval):
                    end = start + self.horizon + 1
                    self.indices.append((i, start, end))
        print('Finishing Pre-load in FK2Plan')

    def angle_normalize(self, joint_angle: np.ndarray):
        joint_angle_norm = np.divide((joint_angle - self._joint_angle_lower),
                                     (self._joint_angle_upper - self._joint_angle_lower))
        joint_angle_norm = joint_angle_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        if type(joint_angle_norm) == torch.Tensor:
            return joint_angle_norm.to(torch.float32)
        else:
            return joint_angle_norm.astype(np.float32)

    def angle_denormalize(self, joint_angle: np.ndarray):
        joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (self._joint_angle_upper - self._joint_angle_lower) + self._joint_angle_lower
        if type(joint_angle_denorm) == torch.Tensor:
            return joint_angle_denorm.to(torch.float32)
        else:
            return joint_angle_denorm.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: Any) -> Tuple:
        traj_idx, start, end = self.indices[index]
        scene_id, start_goal_pose, tra_qpos = self.trajectories[traj_idx]

        ## load trajectory segment
        traj_seg = tra_qpos[start:end]

        ## load data
        scene_pc = self.scene_pcds_nors[scene_id]

        ## randomly resample points
        if self.phase != 'train':
            np.random.seed(0)  # resample point cloud with a fixed random seed
        resample_indices = np.random.permutation(len(scene_pc))
        scene_pc = scene_pc[resample_indices[:self.num_points]]

        ## format point cloud xyz and feature
        xyz = scene_pc[:, 0:3]
        feat = scene_pc[:, 3:3]  # empty array with shape <N, 0>

        if self.use_color:
            raise NotImplementedError

        if self.use_normal:
            normal = scene_pc[:, 3:6]
            feat = np.concatenate([feat, normal], axis=-1)

        data = {
            'x': traj_seg,
            'start': traj_seg[0:1, :],
            'target': tra_qpos[-1],
            'start_goal_pose': start_goal_pose,
            'pos': xyz,
            'feat': feat,
            'scene_id': scene_id,
        }
        if self.transform is not None:
            data = self.transform(data, modeling_keys=self.modeling_keys)

        return data

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    config_path = "../configs/task/franka_planning.yaml"
    cfg = OmegaConf.load(config_path)
    dataloader = FK2Plan(cfg.dataset, 'train', False).get_dataloader(batch_size=128,
                                                                     collate_fn=collate_fn_general,
                                                                     num_workers=4,
                                                                     pin_memory=True,
                                                                     shuffle=True,)

    device = 'cuda'
    st = time.time()
    print(len(dataloader.dataset))
    for it, d in enumerate(dataloader):
        for key in d:
            if torch.is_tensor(d[key]):
                d[key] = d[key].to(device)
        print(f'{time.time() - st}')
        st = time.time()
