import os
import torch
import torch.nn as nn
import numpy as np
import trimesh
import pickle
from omegaconf import DictConfig
from plotly import graph_objects as go
from typing import Any

from utils.misc import random_str
from utils.registry import Registry
from utils.visualize import frame2gif, render_prox_scene, render_scannet_path
from utils.visualize import create_trimesh_nodes_path, create_trimesh_node
from utils.handmodel import get_handmodel
from utils.plotly_utils import plot_mesh

VISUALIZER = Registry('Visualizer')

@VISUALIZER.register()
@torch.no_grad()
class PoseGenVisualizer():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for pose generation task.

        Args:
            cfg: visuzalizer configuration
        """
        self.vis_case_num = cfg.vis_case_num
        self.ksample = cfg.ksample
        self.vis_denoising = cfg.vis_denoising
        self.save_mesh = cfg.save_mesh

    def visualize(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        save_dir: str,
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
        """
        model.eval()
        device = model.device
        
        cnt = 0
        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            
            ksample = 1 if self.vis_denoising else self.ksample
            outputs = model.sample(data, k=ksample) # <B, k, T, D>
            
            for i in range(outputs.shape[0]):
                scene_id = data['scene_id'][i]
                cam_tran = data['cam_tran'][i]
                gender = data['gender'][i]
                
                origin_cam_tran = data['origin_cam_tran'][i]
                scene_trans = cam_tran @ np.linalg.inv(origin_cam_tran) # scene_T @ origin_cam_T = cur_cam_T
                scene_mesh = dataloader.dataset.scene_meshes[scene_id].copy()
                scene_mesh.apply_transform(scene_trans)

                ## calculate camera pose
                camera_pose = np.eye(4)
                camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
                camera_pose = cam_tran @ camera_pose

                if self.vis_denoising:
                    ## generate smplx bodies in all denoising steps
                    ## visualize one body in all steps, visualize the denoising procedure
                    smplx_params = outputs[i, 0, ...] # <T, ...>
                    body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
                    body_verts = body_verts.numpy()

                    save_path_gif = os.path.join(save_dir, f'{scene_id}', '000.gif')
                    save_imgs_dir = os.path.join(save_dir, f'{scene_id}', 'series')
                    timesteps = list(range(len(body_verts))) + [len(body_verts) - 1] * 10 # repeat last frame
                    for f, t in enumerate(timesteps):
                        meshes = {
                            'scenes': [scene_mesh], 
                            'bodies': [trimesh.Trimesh(vertices=body_verts[t], faces=body_faces)]
                        }
                        save_path = os.path.join(save_imgs_dir, f'{f:0>3d}.png')
                        render_prox_scene(meshes, camera_pose, save_path)
                    
                    ## convert images to gif
                    frame2gif(os.path.join(save_dir, f'{scene_id}', 'series'), save_path_gif, size=(480, 270))
                    os.system(f'rm -rf {save_imgs_dir}')
                else:
                    ## generate smplx bodies in last denoising step
                    ## only visualize the body in last step, but visualize multi bodies
                    smplx_params = outputs[i, :, -1, ...] # <k, ...>
                    body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
                    body_verts = body_verts.numpy()

                    if self.save_mesh:
                        os.makedirs(os.path.join(save_dir, f'{scene_id}'), exist_ok=True)
                        scene_mesh.export(os.path.join(save_dir, f'{scene_id}', 'scene.ply'))

                        for j in range(len(body_verts)):
                            body_mesh = trimesh.Trimesh(vertices=body_verts[j], faces=body_faces)
                            ## render generated body separately
                            render_prox_scene({
                                'scenes': [scene_mesh],
                                'bodies': [body_mesh],
                            }, camera_pose, os.path.join(save_dir, f'{scene_id}', f'{j:0>3d}.png'))
                            ## save generated body mesh separately
                            body_mesh.export(os.path.join(save_dir, f'{scene_id}', f'body{j:0>3d}.ply'))
                    else:
                        meshes = {'scenes': [scene_mesh]}
                        meshes['bodies'] = []
                        for j in range(len(body_verts)):
                            meshes['bodies'].append(trimesh.Trimesh(vertices=body_verts[j], faces=body_faces))
                        save_path = os.path.join(save_dir, f'{scene_id}', '000.png')
                        render_prox_scene(meshes, camera_pose, save_path)
                
                cnt += 1
            
            if cnt >= self.vis_case_num:
                break

@VISUALIZER.register()
@torch.no_grad()
class MotionGenVisualizer():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for motion generation task.

        Args:
            cfg: visuzalizer configuration
        """
        self.vis_case_num = cfg.vis_case_num
        self.ksample = cfg.ksample
        self.vis_denoising = cfg.vis_denoising
        self.save_mesh = cfg.save_mesh
    
    def visualize(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        save_dir: str,
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
        """
        model.eval()
        device = model.device

        cnt = 0
        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            data['repr_type'] = dataloader.dataset.repr_type
            
            ksample = 1 if self.vis_denoising else self.ksample
            outputs = model.sample(data, k=ksample) # <B, k, T, L, D>

            for i in range(outputs.shape[0]):
                scene_id = data['scene_id'][i]
                cam_tran = data['cam_tran'][i]
                gender = data['gender'][i]

                origin_cam_tran = data['origin_cam_tran'][i]
                scene_trans = cam_tran @ np.linalg.inv(origin_cam_tran) # scene_T @ origin_cam_T = cur_cam_T
                scene_mesh = dataloader.dataset.scene_meshes[scene_id].copy()
                scene_mesh.apply_transform(scene_trans)

                ## calculate camera pose
                camera_pose = np.eye(4)
                camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
                camera_pose = cam_tran @ camera_pose

                if self.vis_denoising:
                    ## generate smplx bodies in all denoising steps
                    ## visualize bodies in all steps, visualize the denoising procedure
                    smplx_params = outputs[i, 0, ...] # <T, ...>
                    body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
                    body_verts = body_verts.numpy()

                    rand_str = random_str(4)
                    save_path_gif = os.path.join(save_dir, f'{scene_id}_{rand_str}', '000.gif')
                    save_imgs_dir = os.path.join(save_dir, f'{scene_id}_{rand_str}', 'series')
                    timesteps = list(range(len(body_verts))) + [len(body_verts) - 1] * 10 # repeat last frame
                    for f, t in enumerate(timesteps):
                        meshes = {
                            'scenes': [scene_mesh], 
                            'bodies': [trimesh.Trimesh(vertices=bv, faces=body_faces) for bv in body_verts[t]]
                        }
                        save_path = os.path.join(save_imgs_dir, f'{f:0>3d}.png')
                        render_prox_scene(meshes, camera_pose, save_path)
                    
                    ## convert images to gif
                    frame2gif(save_imgs_dir, save_path_gif, size=(480, 270))
                    os.system(f'rm -rf {save_imgs_dir}')
                else:
                    ## generate smplx bodies in all denoising step
                    ## only visualize the body in last step, visualize with gif
                    smplx_params = outputs[i, :, -1, ...] # <k, ...>
                    body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
                    body_verts = body_verts.numpy()

                    rand_str = random_str(4)
                    if self.save_mesh:
                        os.makedirs(os.path.join(save_dir, f'{scene_id}_{rand_str}'), exist_ok=True)
                        scene_mesh.export(os.path.join(save_dir, f'{scene_id}_{rand_str}', 'scene.ply'))
                    
                    for k in range(len(body_verts)):
                        save_path_gif = os.path.join(save_dir, f'{scene_id}_{rand_str}', f'{k:3d}.gif')
                        save_imgs_dir = os.path.join(save_dir, f'{scene_id}_{rand_str}', 'series')
                        for j, body in enumerate(body_verts[k]):
                            body_mesh = trimesh.Trimesh(vertices=body, faces=body_faces)
                            meshes = {
                                'scenes': [scene_mesh],
                                'bodies': [body_mesh]
                            }
                            save_path = os.path.join(save_imgs_dir, f'{j:0>3d}.png')
                            render_prox_scene(meshes, camera_pose, save_path)

                            if self.save_mesh:
                                save_mesh_dir = os.path.join(save_dir, f'{scene_id}_{rand_str}', f'mesh{k:3d}')
                                os.makedirs(save_mesh_dir, exist_ok=True)
                                body_mesh.export(os.path.join(
                                    save_mesh_dir, f'body{j:0>3d}.obj'
                                ))
                        
                        ## convert image to gif
                        frame2gif(save_imgs_dir, save_path_gif, size=(480, 270))
                        os.system(f'rm -rf {save_imgs_dir}')

                cnt += 1
                if cnt >= self.vis_case_num:
                    break
            
            if cnt >= self.vis_case_num:
                break

@VISUALIZER.register()
@torch.no_grad()
class PathPlanningRenderingVisualizer():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for path planning task. Directly rendering images.

        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample
        self.vis_case_num = cfg.vis_case_num
        self.vis_denoising = cfg.vis_denoising
        self.scannet_mesh_dir = cfg.scannet_mesh_dir
    
    def visualize(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        save_dir: str,
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
            vis_denoising: visualize denoising procedure, default is False
            vis_cnt: visualized sample count
        """
        model.eval()
        device = model.device

        cnt = 0
        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            data['repr_type'] = dataloader.dataset.repr_type
            
            ksample = 1 if self.vis_denoising else self.ksample
            outputs = model.sample(data, k=ksample) # <B, k, T, L, D>

            scene_id = data['scene_id']
            trans_mat = data['trans_mat']
            target = data['target'].cpu().numpy()
            for i in range(outputs.shape[0]):
                rand_str = random_str()

                ## load scene and camera pose
                scene_mesh = trimesh.load(os.path.join(
                    self.scannet_mesh_dir, 'mesh', f'{scene_id[i]}_vh_clean_2.ply'))
                scene_mesh.apply_transform(trans_mat[i])
                camera_pose = np.eye(4)
                camera_pose[0:3, -1] = np.array([0, 0, 7])
                
                ## save trajectory
                if self.vis_denoising:
                    save_path_gif = os.path.join(save_dir, f'{scene_id[i]}_{rand_str}', '000.gif')
                    save_imgs_dir = os.path.join(save_dir, f'{scene_id[i]}_{rand_str}', 'series')

                    sequences = outputs[i, 0, ...] # <T, horizon, 2>
                    timesteps = list(range(len(sequences))) + [len(sequences) - 1] * 10 # repeat last frame
                    for f, t in enumerate(timesteps):
                        path = sequences[t].cpu().numpy() # <horizon, 2>

                        render_scannet_path(
                            {'scene': scene_mesh, 
                            'target': create_trimesh_node(target[i], color=np.array([0, 255, 0], dtype=np.uint8)),
                            'path': create_trimesh_nodes_path(path, merge=True)},
                            camera_pose=camera_pose,
                            save_path=os.path.join(save_imgs_dir, f'{f:0>3d}.png')
                        )
                    frame2gif(save_imgs_dir, save_path_gif, size=(480, 270))
                    os.system(f'rm -rf {save_imgs_dir}')
                else:
                    save_imgs_dir = os.path.join(save_dir, f'{scene_id[i]}_{rand_str}')

                    sequences = outputs[i, :, -1, ...] # <k, horizon, 2>
                    for t in range(len(sequences)):
                        path = sequences[t].cpu().numpy() # <horizon, 2>

                        render_scannet_path(
                            {'scene': scene_mesh,
                            'target': create_trimesh_node(target[i], color=np.array([0, 255, 0], dtype=np.uint8)),
                            'path': create_trimesh_nodes_path(path, merge=True)},
                            camera_pose=camera_pose,
                            save_path=os.path.join(save_imgs_dir, f'{t:0>3d}.png')
                        )
                
                cnt += 1
                if cnt >= self.vis_case_num:
                    break
            
            if cnt >= self.vis_case_num:
                break

@VISUALIZER.register()
@torch.no_grad()
class GraspGenVisualizer():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for pose generation task.

        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample
        self.hand_model = get_handmodel(batch_size=self.ksample, device='cuda')

    def visualize(
            self,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            save_dir: str,
            vis_denoising: bool = False,
            vis_cnt: int = 20,
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
            vis_denoising: visualize denoising procedure, default is False
            vis_cnt: visualized sample count
        """
        model.eval()
        device = model.device

        cnt = 0
        ksample = 1 if vis_denoising else self.ksample
        assert (vis_denoising is False)

        os.makedirs(save_dir, exist_ok=True)
        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            outputs = model.sample(data, k=ksample)  # B x ksample x n_steps x 33
            for i in range(outputs.shape[0]):
                scene_id = data['scene_id'][i]
                scene_dataset, scene_object = scene_id.split('+')
                mesh_path = os.path.join('assets/object', scene_dataset, scene_object, f'{scene_object}.stl')
                obj_mesh = trimesh.load(mesh_path)

                hand_qpos = outputs[i, :, -1, ...]
                self.hand_model.update_kinematics(q=hand_qpos)
                for j in range(ksample):
                    vis_data = [plot_mesh(obj_mesh, color='lightblue')]
                    vis_data += self.hand_model.get_plotly_data(i=j, opacity=0.8, color='pink')
                    save_path = os.path.join(save_dir, f'{scene_id}+sample-{j}.html')
                    fig = go.Figure(data=vis_data)
                    fig.write_html(save_path)
                cnt += 1
                if cnt >= vis_cnt:
                    break

@VISUALIZER.register()
@torch.no_grad()
class PoseGenVisualizerHF():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for pose generation task.

        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample

    def visualize(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
    ) -> Any:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
        
        Return:
            Results for gradio rendering.
        """
        model.eval()
        device = model.device

        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            
            outputs = model.sample(data, k=self.ksample) # <B, k, T, D>
            
            i = 0
            scene_id = data['scene_id'][i]
            cam_tran = data['cam_tran'][i]
            gender = data['gender'][i]
            
            origin_cam_tran = data['origin_cam_tran'][i]
            scene_trans = cam_tran @ np.linalg.inv(origin_cam_tran) # scene_T @ origin_cam_T = cur_cam_T
            scene_mesh = dataloader.dataset.scene_meshes[scene_id].copy()
            scene_mesh.apply_transform(scene_trans)

            ## calculate camera pose
            camera_pose = np.eye(4)
            camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
            camera_pose = cam_tran @ camera_pose

            ## generate smplx bodies in last denoising step
            ## only visualize the body in last step, but visualize multi bodies
            smplx_params = outputs[i, :, -1, ...] # <k, ...>
            body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
            body_verts = body_verts.numpy()

            res_images = []
            for j in range(len(body_verts)):
                body_mesh = trimesh.Trimesh(vertices=body_verts[j], faces=body_faces)
                ## render generated body separately
                img = render_prox_scene({'scenes': [scene_mesh], 'bodies': [body_mesh]}, camera_pose, None)
                res_images.append(img)
            return res_images

@VISUALIZER.register()
@torch.no_grad()
class MotionGenVisualizerHF():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for motion generation task.

        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample
    
    def visualize(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
        
        Return:
            Results for gradio rendering.
        """
        model.eval()
        device = model.device

        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            data['repr_type'] = dataloader.dataset.repr_type
            
            outputs = model.sample(data, k=self.ksample) # <B, k, T, L, D>

            i = 0
            scene_id = data['scene_id'][i]
            cam_tran = data['cam_tran'][i]
            gender = data['gender'][i]

            origin_cam_tran = data['origin_cam_tran'][i]
            scene_trans = cam_tran @ np.linalg.inv(origin_cam_tran) # scene_T @ origin_cam_T = cur_cam_T
            scene_mesh = dataloader.dataset.scene_meshes[scene_id].copy()
            scene_mesh.apply_transform(scene_trans)

            ## calculate camera pose
            camera_pose = np.eye(4)
            camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
            camera_pose = cam_tran @ camera_pose

            ## generate smplx bodies in all denoising step
            ## only visualize the body in last step, visualize with gif
            smplx_params = outputs[i, :, -1, ...] # <k, ...>
            body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
            body_verts = body_verts.numpy()
            
            res_ksamples = []
            for k in range(len(body_verts)):
                res_images = []
                for j, body in enumerate(body_verts[k]):
                    body_mesh = trimesh.Trimesh(vertices=body, faces=body_faces)
                    img = render_prox_scene({'scenes': [scene_mesh], 'bodies': [body_mesh]}, camera_pose, None)
                    res_images.append(img)
                res_ksamples.append(res_images)
            return res_ksamples

@VISUALIZER.register()
@torch.no_grad()
class PathPlanningRenderingVisualizerHF():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for path planning task. Directly rendering images.

        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample
        self.scannet_mesh_dir = cfg.scannet_mesh_dir
    
    def visualize(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
        """
        model.eval()
        device = model.device

        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            data['repr_type'] = dataloader.dataset.repr_type
            
            outputs = model.sample(data, k=self.ksample) # <B, k, T, L, D>

            scene_id = data['scene_id']
            trans_mat = data['trans_mat']
            target = data['target'].cpu().numpy()
            i = 0

            ## load scene and camera pose
            scene_mesh = trimesh.load(os.path.join(
                self.scannet_mesh_dir, 'mesh', f'{scene_id[i]}_vh_clean_2.ply'))
            scene_mesh.apply_transform(trans_mat[i])
            camera_pose = np.eye(4)
            camera_pose[0:3, -1] = np.array([0, 0, 10])

            sequences = outputs[i, :, -1, ...] # <k, horizon, 2>
            res_images = []
            for t in range(len(sequences)):
                path = sequences[t].cpu().numpy() # <horizon, 2>

                img = render_scannet_path(
                    {'scene': scene_mesh,
                    'target': create_trimesh_node(target[i], color=np.array([0, 255, 0], dtype=np.uint8)),
                    'path': create_trimesh_nodes_path(path, merge=True)},
                    camera_pose=camera_pose,
                    save_path=None
                )
                res_images.append(img)
            
            return res_images

def create_visualizer(cfg: DictConfig) -> nn.Module:
    """ Create a visualizer for visual evaluation
    Args:
        cfg: configuration object
        slurm: on slurm platform or not. This field is used to specify the data path
    
    Return:
        A visualizer
    """
    return VISUALIZER.get(cfg.name)(cfg)

