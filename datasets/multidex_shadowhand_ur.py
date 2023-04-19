from typing import Any, Tuple, Dict
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf

from datasets.misc import collate_fn_squeeze_pcd_batch_grasp
from datasets.transforms import make_default_transform
from datasets.base import DATASET

@DATASET.register()
class MultiDexShadowHandUR(Dataset):
    """ Dataset for pose generation, training with MultiDex Dataset
    """

    _train_split = ["contactdb+alarm_clock", "contactdb+banana", "contactdb+binoculars",
                    "contactdb+cell_phone", "contactdb+cube_large", "contactdb+cube_medium",
                    "contactdb+cube_small", "contactdb+cylinder_large", "contactdb+cylinder_small",
                    "contactdb+elephant", "contactdb+flashlight", "contactdb+hammer",
                    "contactdb+light_bulb", "contactdb+mouse", "contactdb+piggy_bank", "contactdb+ps_controller",
                    "contactdb+pyramid_large", "contactdb+pyramid_medium", "contactdb+pyramid_small",
                    "contactdb+stanford_bunny", "contactdb+stapler", "contactdb+toothpaste", "contactdb+torus_large",
                    "contactdb+torus_medium", "contactdb+torus_small", "contactdb+train",
                    "ycb+bleach_cleanser", "ycb+cracker_box", "ycb+foam_brick", "ycb+gelatin_box", "ycb+hammer",
                    "ycb+lemon", "ycb+master_chef_can", "ycb+mini_soccer_ball", "ycb+mustard_bottle", "ycb+orange",
                    "ycb+peach", "ycb+pitcher_base", "ycb+plum", "ycb+power_drill", "ycb+pudding_box",
                    "ycb+rubiks_cube", "ycb+sponge", "ycb+strawberry", "ycb+sugar_box", "ycb+toy_airplane",
                    "ycb+tuna_fish_can", "ycb+wood_block"]
    _test_split = ["contactdb+apple", "contactdb+camera", "contactdb+cylinder_medium", "contactdb+rubber_duck",
                   "contactdb+door_knob",  "contactdb+water_bottle", "ycb+baseball", "ycb+pear", "ycb+potted_meat_can",
                   "ycb+tomato_soup_can"]
    _all_split = ["contactdb+alarm_clock", "contactdb+banana", "contactdb+binoculars",
                  "contactdb+cell_phone", "contactdb+cube_large", "contactdb+cube_medium",
                  "contactdb+cube_small", "contactdb+cylinder_large", "contactdb+cylinder_small",
                  "contactdb+elephant", "contactdb+flashlight", "contactdb+hammer",
                  "contactdb+light_bulb", "contactdb+mouse", "contactdb+piggy_bank", "contactdb+ps_controller",
                  "contactdb+pyramid_large", "contactdb+pyramid_medium", "contactdb+pyramid_small",
                  "contactdb+stanford_bunny", "contactdb+stapler", "contactdb+toothpaste", "contactdb+torus_large",
                  "contactdb+torus_medium", "contactdb+torus_small", "contactdb+train",
                  "ycb+bleach_cleanser", "ycb+cracker_box", "ycb+foam_brick", "ycb+gelatin_box", "ycb+hammer",
                  "ycb+lemon", "ycb+master_chef_can", "ycb+mini_soccer_ball", "ycb+mustard_bottle", "ycb+orange",
                  "ycb+peach", "ycb+pitcher_base", "ycb+plum", "ycb+power_drill", "ycb+pudding_box",
                  "ycb+rubiks_cube", "ycb+sponge", "ycb+strawberry", "ycb+sugar_box", "ycb+toy_airplane",
                  "ycb+tuna_fish_can", "ycb+wood_block", "contactdb+apple", "contactdb+camera", "contactdb+cylinder_medium", "contactdb+rubber_duck",
                  "contactdb+door_knob",  "contactdb+water_bottle", "ycb+baseball", "ycb+pear", "ycb+potted_meat_can",
                  "ycb+tomato_soup_can"]

    _joint_angle_lower = torch.tensor([-0.5235988, -0.7853982, -0.43633232, 0., 0., 0., -0.43633232, 0., 0., 0.,
                                       -0.43633232, 0., 0., 0., 0., -0.43633232, 0., 0., 0., -1.047, 0., -0.2618,
                                       -0.5237, 0.])
    _joint_angle_upper = torch.tensor([0.17453292, 0.61086524, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232,
                                       1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964,
                                       0.6981317, 0.43633232, 1.5707964,  1.5707964, 1.5707964, 1.047, 1.309, 0.2618,
                                       0.5237, 1.])

    _global_trans_lower = torch.tensor([-0.13128923, -0.10665303, -0.45753425])
    _global_trans_upper = torch.tensor([0.12772022, 0.22954416, -0.21764427])

    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.

    def __init__(self, cfg: DictConfig, phase: str, slurm: bool, case_only: bool=False, **kwargs: Dict) -> None:
        super(MultiDexShadowHandUR, self).__init__()
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
        self.normalize_x_trans = cfg.normalize_x_trans
        self.obj_dim = int(3 + 3 * self.use_color + 3 * self.use_normal)
        self.transform = make_default_transform(cfg, phase)

        ## resource folders
        self.asset_dir = cfg.asset_dir_slrum if self.slurm else cfg.asset_dir
        self.data_dir = os.path.join(self.asset_dir, 'shadowhand')
        self.scene_path = os.path.join(self.asset_dir, 'object_pcds.pkl')
        self._joint_angle_lower = self._joint_angle_lower.cpu()
        self._joint_angle_upper = self._joint_angle_upper.cpu()
        self._global_trans_lower = self._global_trans_lower.cpu()
        self._global_trans_upper = self._global_trans_upper.cpu()

        ## load data
        self._pre_load_data(case_only)

    def _pre_load_data(self, case_only: bool) -> None:
        """ Load dataset

        Args:
            case_only: only load single case for testing, if ture, the dataset will be smaller.
                        This is useful in after-training visual evaluation.
        """
        self.frames = []
        self.scene_pcds = {}

        grasp_dataset = torch.load(os.path.join(self.data_dir, 'shadowhand_downsample.pt' if self.is_downsample else 'shadowhand.pt'))
        self.scene_pcds = pickle.load(open(self.scene_path, 'rb'))
        self.dataset_info = grasp_dataset['info']
        # pre-process the dataset info
        for obj in grasp_dataset['info']['num_per_object'].keys():
            if obj not in self.split:
                self.dataset_info['num_per_object'][obj] = 0
        # for obj in self.scene_pcds.keys():
        #     self.scene_pcds[obj] = torch.tensor(self.scene_pcds[obj], device=self.device)
        for mdata in grasp_dataset['metadata']:
            mdata_qpos = mdata[0].cpu()
            joint_angle = mdata_qpos.clone().detach()[9:]
            global_trans = mdata_qpos.clone().detach()[:3]
            if self.normalize_x:
                joint_angle = self.angle_normalize(joint_angle)
            if self.normalize_x_trans:
                global_trans = self.trans_normalize(global_trans)
            mdata_qpos = torch.cat([global_trans, joint_angle], dim=0).requires_grad_(True)
            if mdata[2] in self.split:
                self.frames.append({'robot_name': 'shadowhand',
                                    'object_name': mdata[2],
                                    'object_rot_mat': mdata[1].clone().detach().numpy(),
                                    'qpos': mdata_qpos})
        # print('Finishing Pre-load in MultiDexShadowHand')

    def trans_normalize(self, global_trans: torch.Tensor):
        global_trans_norm = torch.div((global_trans - self._global_trans_lower), (self._global_trans_upper - self._global_trans_lower))
        global_trans_norm = global_trans_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return global_trans_norm

    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self._global_trans_upper - self._global_trans_lower) + self._global_trans_lower
        return global_trans_denorm

    def angle_normalize(self, joint_angle: torch.Tensor):
        joint_angle_norm = torch.div((joint_angle - self._joint_angle_lower), (self._joint_angle_upper - self._joint_angle_lower))
        joint_angle_norm = joint_angle_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return joint_angle_norm

    def angle_denormalize(self, joint_angle: torch.Tensor):
        joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (self._joint_angle_upper - self._joint_angle_lower) + self._joint_angle_lower
        return joint_angle_denorm

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index: Any) -> Tuple:

        frame = self.frames[index]

        ## load data, containing scene point cloud and point pose
        scene_id = frame['object_name']
        scene_rot_mat = frame['object_rot_mat']
        scene_pc = self.scene_pcds[scene_id]
        scene_pc = np.einsum('mn, kn->km', scene_rot_mat, scene_pc)
        cam_tran = None

        ## randomly resample points
        if self.phase != 'train':
            np.random.seed(0) # resample point cloud with a fixed random seed
        # np.random.shuffle(scene_pc)
        # scene_pc = scene_pc[:self.num_points]
        resample_indices = np.random.permutation(len(scene_pc))
        scene_pc = scene_pc[resample_indices[:self.num_points]]

        ## format point cloud xyz and feature
        xyz = scene_pc[:, 0:3]

        if self.use_color:
            color = scene_pc[:, 3:6] / 255.
            feat = np.concatenate([color], axis=-1)

        if self.use_normal:
            normal = scene_pc[:, 6:9]
            feat = np.concatenate([normal], axis=-1)

        ## format smplx parameters
        grasp_qpos = (
            frame['qpos']
        )
        
        data = {
            'x': grasp_qpos,
            'pos': xyz,
            'scene_rot_mat': scene_rot_mat,
            'cam_tran': cam_tran, 
            'scene_id': scene_id,
        }

        if self.transform is not None:
            data = self.transform(data, modeling_keys=self.modeling_keys)

        return data

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    config_path = "../configs/task/grasp_gen.yaml"
    cfg = OmegaConf.load(config_path)
    dataloader = MultiDexShadowHandUR(cfg.dataset, 'train', False).get_dataloader(batch_size=4,
                                                                                  collate_fn=collate_fn_squeeze_pcd_batch_grasp,
                                                                                  num_workers=0,
                                                                                  pin_memory=True,
                                                                                  shuffle=True,)

    device = 'cuda'
    for it, data in enumerate(dataloader):
        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(device)
        print()