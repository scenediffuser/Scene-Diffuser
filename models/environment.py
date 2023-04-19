from typing import Dict
import os
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import cos, sin
from tqdm import tqdm
from omegaconf import DictConfig
from collections import defaultdict
import json
import pickle
import copy
import yaml

from models.optimizer.utils import transform_verts
from utils.registry import Registry
from utils.misc import random_str
from utils.visualize import create_trimesh_node, create_trimesh_nodes_path, render_scannet_path
from envs.tasks.franka_panda import FrankaMotion_Player
from isaacgym import gymapi, gymtorch, gymutil

ENV = Registry('Env')

class ObservationQueue():

    def __init__(self, size: int, repr_type: str, normalizer: object) -> None:
        self.size = size
        self.repr_type = repr_type
        self.normalizer = normalizer
    
    def initialize(self, start: torch.Tensor) -> None:
        self.start = start.clone()
        self.x = []
        self.size = self.size - start.shape[1]
    
    def push(self, action: torch.Tensor) -> None:
        if len(action.shape) == 2:
            action = action.unsqueeze(1)
        
        if self.repr_type == 'absolute':
            x = self.start[:, -1:, :] + action
        elif self.repr_type == 'relative':
            x = action.clone()
        else:
            raise Exception('Unsupported repr type.')
        
        self.x.append(x)

        if len(self.x) > self.size:
            if self.repr_type == 'absolute':
                self.start = torch.cat([self.start[:, 1:, :], self.x[0]], dim=1)
            elif self.repr_type == 'relative':
                last_start = self.start[:, -1:, :] + self.x[0]
                self.start = torch.cat([self.start[:, 1:, :], last_start], dim=1)
            else:
                raise Exception('Unsupported repr type.')
            
            self.x = self.x[1:]

    def obser(self) -> torch.Tensor:
        start = self.start.clone()
        if len(self.x) == 0:
            return start, None
        
        obser = torch.cat(self.x, dim=1)
        if self.normalizer is not None:
            obser = self.normalizer.normalize(obser)

        return start, obser

class MP3DPathPlanningEnvCore():
    def __init__(self, data: Dict, robot_radius: float, robot_bottom: float,
    robot_top: float, arrive_threshold: float=0.1, max_trajectory_length: int=250,
    env_adaption: bool=True, inpainting_horizon: int=32) -> None:
        """ A MP3D path planning environment wrapper used for giving feedback. Support batch data.

        Args:
            data: dataloader-provided data dict
            robot_radius, robot_bottom, robot_top: configuration of robot
            arrive_threshold: threshold of arriving the target
            max_trajectory_length: max length of the trajectory
        """
        self.device = data['start'].device
        self.batch = data['start'].shape[0]
        self.state = data['start'].clone() # the start is the initial state, <B, 1, D>
        self.target = data['target'] # <B, D>
        self.scene_pos = data['pos'].reshape(self.batch, -1, 3)
        trans_mat = torch.tensor(np.array(data['trans_mat'], dtype=np.float32), device=self.device) # <B, 4, 4>
        self.z_trans = trans_mat[:, 2, -1]
        self.trans_mat_inv = torch.linalg.inv(trans_mat)

        self.observation_queue = ObservationQueue(size=inpainting_horizon-1, repr_type=data['repr_type'], normalizer=data['normalizer'])
        self.observation_queue.initialize(data['start'])

        self.s_grid_map = data['s_grid_map']
        self.s_grid_min = data['s_grid_min']
        self.s_grid_max = data['s_grid_max']
        self.s_grid_dim = data['s_grid_dim']
        self.traj_length = data['traj_length']

        self.robot_radius = robot_radius
        self.robot_bottom = robot_bottom
        self.robot_top = robot_top

        self.arrive_threshold = arrive_threshold
        self.max_trajectory_length = max_trajectory_length

        self.end = torch.zeros(self.batch, dtype=bool, device=self.device)
        self.trajectory = [[self.state[i].clone()] for i in range(self.batch)]
        self.trajectory_length = torch.ones(self.batch, 
            dtype=torch.int32, device=self.device) * max_trajectory_length

        if env_adaption:
            angles = np.linspace(0, torch.pi * 2, 13)[0:12] # 12 bins 
            self.rotate_angles = np.zeros(12)
            self.rotate_angles[0::2] = angles[0:6]
            self.rotate_angles[1::2] = angles[-1:5:-1]
            self.rotate_angles = torch.tensor(self.rotate_angles)
        else:
            self.rotate_angles = torch.zeros(1)

    def verify_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """ Verify if the action is valid

        Args:
            actions: action tensor, <B, 1, 2>
        
        Return:
            A bool tensor indicates the validity
        """
        ## compute the next state position, and convert to absolute coordinates
        B, T, _ = self.state.shape # T == 1
        next_state = actions + self.state # <B, 1, 2>

        ## debug
        # for i in range(B):
        #     S = trimesh.Scene()
        #     scene_pc = self.scene_pos[i].cpu().numpy()
        #     S.add_geometry(trimesh.PointCloud(vertices=scene_pc))

        #     pos = self.state.cpu().numpy()[i:i+1].squeeze()
        #     pos_mesh = create_trimesh_node(pos, radius=0.08)
        #     S.add_geometry(pos_mesh)

        #     next_pos = next_state.cpu().numpy()[i:i+1].squeeze()
        #     next_pos_mesh = create_trimesh_node(next_pos, radius=0.08, color=np.array([0, 255, 0], dtype=np.uint8))
        #     S.add_geometry(next_pos_mesh)
            
        #     S.add_geometry(trimesh.creation.axis())
        #     S.show()

        ## compute the height of current state
        next_state3 = torch.cat([next_state, torch.zeros(B, T, 1, dtype=next_state.dtype, device=self.device)], dim=-1)
        next_state_trans = transform_verts(next_state3, self.trans_mat_inv)

        s_grid_min = self.s_grid_min.unsqueeze(1)
        s_grid_max = self.s_grid_max.unsqueeze(1)

        norm_x_batch = ((next_state_trans[..., 0:2] - s_grid_min) / (s_grid_max - s_grid_min) * 2 -1)
        height = F.grid_sample(
            self.s_grid_map.unsqueeze(1),   # <B, 1, H, W>
            norm_x_batch.view(-1, T, 1, 2), # <B, T, 1, 2>
            padding_mode='border', align_corners=True) # <B, 1, T, 1>
        height = height.view(B, 1) + self.z_trans.unsqueeze(-1)

        ## compute robotic collision with scene
        scene_verts = self.scene_pos.reshape(B, -1, 3) # <B, N, 3>

        between = torch.logical_and(
            scene_verts[..., 2] > (height + self.robot_bottom), # <B, N> > <B, 1>
            scene_verts[..., 2] < (height + self.robot_top)     # <B, N> < <B, 1>
        ) # <B, N>
        dist = torch.linalg.norm(scene_verts[..., 0:2] - next_state, dim=-1) # <B, N>, norm(<B, N, 2> - <B, 1, 2>)
        within = dist < self.robot_radius # <B, N>

        ## debug
        # mask = torch.logical_and(between, within)
        # print(mask.shape)
        # print(mask.sum(-1))
        # for i in range(B):
        #     S = trimesh.Scene()
        #     scene_pc = self.scene_pos[i].cpu().numpy()
        #     color = np.ones((len(scene_pc), 4), dtype=np.uint8) * 255
        #     color[:, 0:3] = 0
        #     color[mask.cpu().numpy()[i], :] = np.array([255, 0, 0, 255], dtype=np.uint8)
        #     S.add_geometry(trimesh.PointCloud(vertices=scene_pc, colors=color))

        #     pos = self.state.cpu().numpy()[i:i+1].squeeze()
        #     pos_mesh = create_trimesh_node(pos, radius=0.08)
        #     S.add_geometry(pos_mesh)

        #     next_pos = next_state.cpu().numpy()[i:i+1].squeeze()
        #     next_pos_mesh = create_trimesh_node(next_pos, radius=0.08, color=np.array([0, 255, 0], dtype=np.uint8))
        #     S.add_geometry(next_pos_mesh)
            
        #     S.add_geometry(trimesh.creation.axis())
        #     S.show()

        return torch.logical_and(between, within).sum(-1) == 0 # <B>
        
    def step(self, step: int, actions: torch.Tensor) -> None:
        """ Step in the environment

        Args:
            step: the step number, start from 1
            actions: action tensor
        """
        actions = actions[:, 0:1, :] # only consider 1 actions

        ## execute the action and compute the next state
        ## 1. normalize the step length of action
        B, K, D = actions.shape # <B, k, D>
        actions_norm = torch.norm(actions, dim=-1)
        mask = actions_norm > self.robot_radius # Important !!!
        actions[mask, ...] *= (self.robot_radius / actions_norm[mask]).unsqueeze(-1)

        ## 2.check the validity of actions
        actions_valid = torch.zeros_like(actions) # <B, 1, D>
        actions_find = self.end.clone() # <B>
        for angle in self.rotate_angles:
            rot_mat = torch.tensor(np.array([[cos(angle), -sin(angle)], 
                [sin(angle), cos(angle)]]), dtype=actions.dtype, device=self.device)
            
            actions_rotate = torch.mm(
                rot_mat, # <2, 2>
                actions.squeeze(1).t() # <2, B>
            ).t().unsqueeze(1) # <B, 1, D>
            
            action_valid_mask = self.verify_actions(actions_rotate)
            action_valid_mask = torch.logical_and(action_valid_mask, ~actions_find)

            ## debug
            # for i in range(B):
            #     S = trimesh.Scene()
            #     scene_pc = self.scene_pos[i].cpu().numpy()
            #     S.add_geometry(trimesh.PointCloud(vertices=scene_pc))
            #     pos = self.state.cpu().numpy()[i:i+1].squeeze()
            #     pos_mesh = create_trimesh_node(pos, radius=0.08)
            #     S.add_geometry(pos_mesh)
            #     S.add_geometry(trimesh.creation.axis())
            #     S.show()
            # print(self.state)

            actions_valid[action_valid_mask, ...] = actions_rotate[action_valid_mask, ...]
            actions_find = torch.logical_or(action_valid_mask, actions_find)

            if torch.all(actions_find):
                break

        ## 3.compute next state
        next_state = actions_valid + self.state
        self.observation_queue.push(actions_valid)

        ## 4. update the state
        self.state[~self.end, ...] = next_state[~self.end, ...]
        for i in range(self.batch):
            self.trajectory[i].append(self.state[i].clone())
        
        ## debug
        # for i in range(B):
        #     S = trimesh.Scene()
        #     scene_pc = self.scene_pos[i].cpu().numpy()
        #     S.add_geometry(trimesh.PointCloud(vertices=scene_pc))
        #     pos = self.state.cpu().numpy()[i:i+1].squeeze()
        #     pos_mesh = create_trimesh_node(pos, radius=0.08)
        #     S.add_geometry(pos_mesh)
        #     S.add_geometry(trimesh.creation.axis())
        #     S.show()
        # print(self.state)
        # print(self.trajectory)
        
        ## 5. check arriving the target position
        dist = torch.norm(self.state.squeeze(1) - self.target, dim=-1)
        arrived = dist < self.arrive_threshold

        self.trajectory_length[torch.logical_and(arrived, ~self.end)] = step
        self.end = torch.logical_or(self.end, arrived)

    def all_end(self) -> bool:
        """ If all case is end

        Return:
            A bool
        """
        return torch.all(self.end)
    
    def get_target(self) -> torch.Tensor:
        """ Get the target position in batch

        Return:
            Target position bacth with shape <B, 2>
        """
        return self.target
    
    def get_trajectory(self) -> torch.Tensor:
        """ Get the trajectory in batch
        
        Return:
            Trajectory bacth with shape <B, T, 2>
        """
        trajectory_batch = [torch.cat(self.trajectory[i], dim=0) for i in range(self.batch)]
        trajectory_batch = torch.stack(trajectory_batch)
        
        return trajectory_batch
    
    def get_trajectory_length(self) -> torch.Tensor:
        """ Get trajectory length in batch
        """
        return self.trajectory_length

class FK2PlanningEnvCore():
    def __init__(self, data: Dict, scene_id: str, sims_per_step: int = 10,
                 arrive_threshold: float = 0.3, max_trajectory_length: int = 500) -> None:
        """ A Franka Forward kinematics planning environment

        Args:
            data: dataloader provided data dict
            scene_id: the scene id of data
            max_trajectory_length: max length of the trajectory
        """
        self.scene_id = scene_id
        self.device = data['start'].device
        self.batch = data['start'].shape[0]
        self.max_trajectory_length = max_trajectory_length
        self.arrive_threshold = arrive_threshold
        self.sims_per_step = sims_per_step
        #
        # self.angle_normalize = angle_normalize
        # self.angle_denormalize = angle_denormalize

        self.state = data['start'].squeeze(1)
        self.target = data['target']

        self.end = torch.zeros(self.batch, dtype=bool, device=self.device)
        self.trajectory = [[self.state[i].clone().cpu()] for i in range(self.batch)]
        self.franka_isaac_env = None
        self.initial_qpos = self.state
        self.isaac_config = yaml.safe_load(open("envs/tasks/franka_panda.yaml"))
        self.trajectory_length = torch.ones(self.batch, dtype=torch.int32, device=self.device) * max_trajectory_length
        self.init_isaac_env(initial_qpos=self.initial_qpos)

    def get_sim_param(self):
        # initialize sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1. / 60.
        sim_params.num_client_threads = 0
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = True
        sim_params.physx.num_subscenes = 0
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.physx.num_threads = 0
        return sim_params

    def __del__(self):
        del self.franka_isaac_env

    def init_isaac_env(self, initial_qpos: torch.Tensor):
        self.initial_qpos = initial_qpos.detach().clone()
        sim_params = self.get_sim_param()
        self.franka_isaac_env = FrankaMotion_Player(config=self.isaac_config, sim_params=sim_params,
                                                    physics_engine=gymapi.SIM_PHYSX,
                                                    device_type=self.device, device_id=0, headless=False,
                                                    scene_id=self.scene_id, init_qpos=initial_qpos)

    def step(self, step: int, target_qpos: torch.Tensor):
        ## 1. step target qpos
        next_state = self.step_target_qpos(target_qpos)
        ## 2. compute next state
        self.state[~self.end, ...] = next_state[~self.end, ...]
        for i in range(self.batch):
            self.trajectory[i].append(self.target[i].clone().cpu() if self.end[i] else self.state[i].clone().cpu())
        ## 3. check arriving the target position
        dist = torch.norm(self.state - self.target, dim=-1)
        arrived = dist < self.arrive_threshold

        # print(self.end)
        self.trajectory_length[torch.logical_and(arrived, ~self.end)] = step
        self.end = torch.logical_or(self.end, arrived)

    def get_trajectories(self):
        return self.trajectory

    def step_target_qpos(self, target_qpos: torch.Tensor):
        for i in range(self.sims_per_step):
            self.franka_isaac_env.step_qpos(qpos=target_qpos)
        next_cur_qpos = self.get_current_qpos()
        return next_cur_qpos

    def get_current_qpos(self):
        cur_qpos = self.franka_isaac_env.get_cur_qpos()
        return cur_qpos.squeeze(1)

    def all_end(self) -> bool:
        """ If all case is end
        Returns:
            A bool
        """
        return torch.all(self.end)

@ENV.register()
@torch.no_grad()
class FK2PlanningEnvWrapper():
    def __init__(self, cfg: DictConfig) -> None:
        """ FK2Planning environment class for path planning task.

        Args:
            cfg:
        """
        self.max_sample_each_step = cfg.max_sample_each_step
        self.max_trajectory_length = cfg.max_trajectory_length
        self.arrive_threshold = cfg.arrive_threshold
        self.eval_case_num_per_scene = cfg.eval_case_num_per_scene
        self.sims_per_step = cfg.sims_per_step
        self.arrive_threshold = cfg.arrive_threshold

        self.angle_normalize = None
        self.angle_denormalize = None

    def run(self,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            save_dir: str
    ) -> None:
        """ Planning within the environment
        
        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save_directory of result
        """
        # save_record_path = os.path.join('/home/puhao/dev/SceneDiffuser', 'succ_record.json')
        model.eval()
        device = model.device

        self.angle_normalize = dataloader.dataset.angle_normalize
        self.angle_denormalize = dataloader.dataset.angle_denormalize
        # task_name = 'pla-0.1@demo'
        # res_saver_path = os.path.join('outputs', task_name, 'metrics.json')
        # replay_saver_path = os.path.join('outputs', task_name, 'replay.pkl')

        replay_res = {}

        # succ_scene_record = json.load(open(save_record_path, 'r'))
        res = defaultdict(list)
        res['succ'] = []
        res['eval_cnt'] = 0
        res['max_trajectory_length'] = self.max_trajectory_length
        res['arrive_threshold'] = self.arrive_threshold
        res['scene_list'] = []
        # if os.path.exists(res_saver_path):
        #     with open(res_saver_path, 'r') as fp:
        #         res = json.load(fp)
        # else:
        #     with open(res_saver_path, 'w') as fp:
        #         json.dump(res, fp)
        for i, data in enumerate(dataloader):
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            i_scene_id = data['scene_id'][0]
            i_target_qpos = np.array(data['target'].cpu())
            # if data['scene_id'][0] in succ_scene_record.keys():
            #     continue
            if data['scene_id'][0] in res['scene_list']:
                continue
            succ_record_list = []
            ## de-normalize for env input
            if dataloader.dataset.normalize_x:
                data['x'] = self.angle_denormalize(data['x'].cpu()).cuda()
                data['target'] = self.angle_denormalize(data['target'].cpu()).cuda()
                data['start'] = self.angle_denormalize(data['start'].cpu()).cuda()

            env = FK2PlanningEnvCore(
                data=copy.deepcopy(data),
                scene_id=data['scene_id'][0],
                arrive_threshold=self.arrive_threshold,
                sims_per_step=self.sims_per_step,
                max_trajectory_length=self.max_trajectory_length
            )

            ## re-normalize for model input
            if dataloader.dataset.normalize_x:
                data['x'] = self.angle_normalize(data['x'].cpu()).cuda()
                data['target'] = self.angle_normalize(data['target'].cpu()).cuda()
                data['start'] = self.angle_normalize(data['start'].cpu()).cuda()

            for j in tqdm(range(self.max_trajectory_length)):
                outputs = model.sample(data, k=self.max_sample_each_step)
                # assert torch.sum(outputs[:, 0:1, -1, 0, :] - env.state) < 1e-6
                assert (outputs.shape[1] == 1)

                pred_next_qpos = outputs[:, 0, -1, 10, :]
                # # todo: l2 loss
                # pred_next_qpos = data['target']

                if dataloader.dataset.normalize_x:
                    pred_next_qpos = self.angle_denormalize(pred_next_qpos.cpu()).cuda()
                env.step(j + 1, pred_next_qpos)

                data['start'] = env.state.clone()
                if dataloader.dataset.normalize_x:
                    data['start'] = self.angle_normalize(data['start'].cpu()).cuda()
                    data['start'] = data['start'].unsqueeze(1)
                if env.all_end():
                    break
            print()
            i_trajectories = env.get_trajectories()
            i_trajectories = np.array([np.stack(case) for case in i_trajectories])
            # save replayer recorder
            replay_res[i_scene_id] = {'sample_trajs': i_trajectories,
                                      'target_qpos': i_target_qpos}

            for j in range(env.batch):
                succ_record_list.append(bool(env.end[j].cpu()))
                res['scene_list'].append(data['scene_id'][0])
                res['succ'].append(bool(env.end[j].cpu()))
                res['length'].append(float(env.trajectory_length[j]))
            # print(f'[{i}/{200}]')
            # succ_scene_record[data['scene_id'][0]] = succ_record_list
            res['eval_cnt'] += env.batch

            # json.dump(succ_scene_record, open(save_record_path, 'w'))

            del env
            # with open(res_saver_path, 'w') as fp:
            #     json.dump(res, fp)

        # with open(replay_saver_path, 'wb') as fp:
        #     pickle.dump(replay_res, fp)
        for key in ['succ', 'length']:
            res[key+'_average'] = sum(res[key]) / len(res[key])

        ## save quantitative results
        save_path = os.path.join(save_dir, 'metrics.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as fp:
            json.dump(res, fp)
        # with open(res_saver_path, 'w') as fp:
        #     json.dump(res, fp)

@ENV.register()
@torch.no_grad()
class PathPlanningEnvWrapper():
    def __init__(self, cfg: DictConfig) -> None:
        """ Path palnning environment class for path planning task.

        Args:
            cfg: environment configuration
        """
        self.max_sample_each_step = cfg.max_sample_each_step
        self.inpainting_horizon = cfg.inpainting_horizon
        self.max_trajectory_length = cfg.max_trajectory_length
        self.arrive_threshold = cfg.arrive_threshold
        self.eval_case_num = cfg.eval_case_num
        self.vis_case_num = cfg.vis_case_num

        self.robot_radius = cfg.robot_radius
        self.robot_bottom = cfg.robot_bottom
        self.robot_top = cfg.robot_top
        self.env_adaption = cfg.env_adaption

        self.scannet_mesh_dir = cfg.scannet_mesh_dir

    def run(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        save_dir: str,
    ) -> None:
        """ Palnning within the environment

        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save_directory of results
            run_cnt: eval sample count
        """
        model.eval()
        device = model.device

        res = defaultdict(list)
        res['eval_cnt'] = 0
        res['max_trajectory_length'] = self.max_trajectory_length
        res['arrive_threshold'] = self.arrive_threshold
        for i, data in enumerate(dataloader):
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            data['repr_type'] = dataloader.dataset.repr_type
            
            env = MP3DPathPlanningEnvCore(
                data,
                robot_radius=self.robot_radius,
                robot_bottom=self.robot_bottom,
                robot_top=self.robot_top,
                arrive_threshold=self.arrive_threshold,
                max_trajectory_length=self.max_trajectory_length,
                env_adaption=self.env_adaption,
                inpainting_horizon=self.inpainting_horizon,
            )

            for i in tqdm(range(self.max_trajectory_length)):
                outputs = model.sample(data, k=self.max_sample_each_step) # <B, k, T, L, D>

                ## the model outputs have the shape <Batch, k_sample, denoising_step_T, horizon, dim>
                ## we will sample valid action from the k_sample
                ## use the second frame of the horizon as action, the first frame is the observation
                O = data['start'].shape[1] + (data['obser'].shape[1] if 'obser' in data else 0)
                assert torch.sum(outputs[:, 0:1, -1, O-1, :] - env.state) < 1e-6

                pred_next_state = outputs[:, :, -1, O, :] # <B, k, D>
                actions = pred_next_state - env.state # <B, k, D>

                env.step(i + 1, actions) # execute actions

                ## update observation
                start, obser = env.observation_queue.obser()
                data['start'] = start
                if obser is not None:
                    data['obser'] = obser

                if env.all_end():
                    break
            
            for i in range(env.batch):
                res['succ'].append(float(env.end[i]))
                res['length'].append(float(env.trajectory_length[i]))
                res['traj_length'].append(env.traj_length[i])
            
            ## save visual results
            if res['eval_cnt'] < self.vis_case_num:
                scene_id = data['scene_id']
                trans_mat = data['trans_mat']
                target = env.get_target()
                trajectory = env.get_trajectory()
                trajectory_length = env.get_trajectory_length()
                for i in range(env.batch):
                    rand_str = random_str()

                    ## load scene mesh and camera pose
                    scene_mesh = trimesh.load(os.path.join(
                        self.scannet_mesh_dir, 'mesh', f'{scene_id[i]}_vh_clean_2.ply'))
                    scene_mesh.apply_transform(trans_mat[i])
                    camera_pose = np.eye(4)
                    camera_pose[0:3, -1] = np.array([0, 0, 7])
                    
                    tar = target[i].cpu().numpy()   # <B, 2>
                    path = trajectory[i].cpu().numpy() # <B, T, 2>
                    length = trajectory_length[i]
                    path = path[0:length+1]

                    render_scannet_path(
                        {'scene': scene_mesh,
                        'target': create_trimesh_node(tar, color=np.array([0, 255, 0], dtype=np.uint8)),
                        'path': create_trimesh_nodes_path(path, merge=True)},
                        camera_pose=camera_pose,
                        save_path=os.path.join(save_dir, f'{scene_id[i]}_{rand_str}', f'res_{length:0>3d}.png')
                    )

            res['eval_cnt'] += env.batch
            if res['eval_cnt'] >= self.eval_case_num:
                break
        
        for key in ['succ', 'length']:
            res[key+'_average'] = sum(res[key]) / len(res[key])
        
        ## save quantitative results
        import json
        save_path = os.path.join(save_dir, 'metrics.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as fp:
            json.dump(res, fp)

@ENV.register()
@torch.no_grad()
class PathPlanningEnvWrapperHF():
    def __init__(self, cfg: DictConfig) -> None:
        """ Path palnning environment class for path planning task.

        Args:
            cfg: environment configuration
        """
        self.max_sample_each_step = cfg.max_sample_each_step
        self.inpainting_horizon = cfg.inpainting_horizon
        self.max_trajectory_length = cfg.max_trajectory_length
        self.arrive_threshold = cfg.arrive_threshold

        self.robot_radius = cfg.robot_radius
        self.robot_bottom = cfg.robot_bottom
        self.robot_top = cfg.robot_top
        self.env_adaption = cfg.env_adaption

        self.scannet_mesh_dir = cfg.scannet_mesh_dir

    def run(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """ Palnning within the environment

        Args:
            model: diffusion model
            dataloader: test dataloader
        """
        model.eval()
        device = model.device

        for i, data in enumerate(dataloader):
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            data['repr_type'] = dataloader.dataset.repr_type
            
            env = MP3DPathPlanningEnvCore(
                data,
                robot_radius=self.robot_radius,
                robot_bottom=self.robot_bottom,
                robot_top=self.robot_top,
                arrive_threshold=self.arrive_threshold,
                max_trajectory_length=self.max_trajectory_length,
                env_adaption=self.env_adaption,
                inpainting_horizon=self.inpainting_horizon,
            )

            for i in range(self.max_trajectory_length):
                outputs = model.sample(data, k=self.max_sample_each_step) # <B, k, T, L, D>

                ## the model outputs have the shape <Batch, k_sample, denoising_step_T, horizon, dim>
                ## we will sample valid action from the k_sample
                ## use the second frame of the horizon as action, the first frame is the observation
                O = data['start'].shape[1] + (data['obser'].shape[1] if 'obser' in data else 0)
                assert torch.sum(outputs[:, 0:1, -1, O-1, :] - env.state) < 1e-6

                pred_next_state = outputs[:, :, -1, O, :] # <B, k, D>
                actions = pred_next_state - env.state # <B, k, D>

                env.step(i + 1, actions) # execute actions

                ## update observation
                start, obser = env.observation_queue.obser()
                data['start'] = start
                if obser is not None:
                    data['obser'] = obser

                if env.all_end():
                    break
            
            scene_id = data['scene_id']
            trans_mat = data['trans_mat']
            target = env.get_target()
            trajectory = env.get_trajectory()
            trajectory_length = env.get_trajectory_length()
            i = 0
                
            ## load scene mesh and camera pose
            scene_mesh = trimesh.load(os.path.join(
                self.scannet_mesh_dir, 'mesh', f'{scene_id[i]}_vh_clean_2.ply'))
            scene_mesh.apply_transform(trans_mat[i])
            camera_pose = np.eye(4)
            camera_pose[0:3, -1] = np.array([0, 0, 10])
            
            tar = target[i].cpu().numpy()   # <B, 2>
            path = trajectory[i].cpu().numpy() # <B, T, 2>
            length = trajectory_length[i].item()
            path = path[0:length+1]

            img = render_scannet_path(
                {'scene': scene_mesh,
                'target': create_trimesh_node(tar, color=np.array([0, 255, 0], dtype=np.uint8)),
                'path': create_trimesh_nodes_path(path, merge=True)},
                camera_pose=camera_pose,
                save_path=None
            )

            return ([img], length)   

def create_enviroment(cfg: DictConfig) -> nn.Module:
    """ Create a planning environment for planning task
    Args:
        cfg: configuration object
        slurm: on slurm platform or not. This field is used to specify the data path
    
    Return:
        A Plan Env
    """
    return ENV.get(cfg.name)(cfg)