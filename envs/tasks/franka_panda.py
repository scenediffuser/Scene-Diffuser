import os
from requests import get
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
import yaml
from envs.tasks.base_task import BaseTask
from tqdm import tqdm
import numpy as np
import torch
import trimesh as tm


class FrankaMotion_Player(BaseTask):

    def __init__(self, config, sim_params, physics_engine,
                 device_type, device_id, headless, scene_id,
                 init_qpos: torch.tensor):
        self.gym = None
        self.viewer = None
        self.num_sim = 0
        self.cfg = config
        self.scene_id = scene_id
        self.sim_params = sim_params
        self.device = device_type
        self.init_qpos = init_qpos.clone()
        self.physics_engine = physics_engine
        self.up_axis = 'z'
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self.device_type = device_type
        self.device_id = device_id
        self.headless = headless

        self.device = "cuda"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        config["env"]["numTrain"] = init_qpos.shape[0]
        config["env"]["numEnvs"] = init_qpos.shape[0]
        self.env_num_train = config["env"]["numTrain"]
        self.env_num = self.env_num_train
        self.asset_root = config["env"]["asset"]["assetRoot"]
        self.num_train = config["env"]["numTrain"]
        self.tot_num = 1
        self.exp_name = config['env']["env_name"]

        print("Simulator: number of objects", self.tot_num)
        print("Simulator: number of environments", self.env_num)
        if self.num_train:
            assert (self.env_num_train % self.num_train == 0)

        # the number of used length must less than real length
        # each object should have equal number envs
        assert (self.env_num % self.tot_num == 0)
        self.env_per_object = self.env_num // self.tot_num

        self.env_ptr_list = []
        self.obj_loaded = False
        self.dexterous_loaded = False

        super().__init__(cfg=self.cfg, enable_camera_sensors=config["env"]["enableCameraSensors"],
                         cam_pos=(-1.2, -1.2, 1.2), cam_target=(0., 0., 0.5))

        # acquire tensors
        self.root_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)).to(self.device)
        self.dof_state_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)).to(self.device)
        # self.rigid_body_tensor = gymtorch.wrap_tensor(
        #     self.gym.acquire_rigid_body_state_tensor(self.sim)).to(self.device).reshape(self.num_envs, -1, 13)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_tensor = self.root_tensor.view(self.num_envs, -1, 13)
        self.dof_state_tensor = self.dof_state_tensor.view(
            self.num_envs, -1, 2)
        self.initial_dof_states = self.dof_state_tensor.clone()
        self.initial_root_states = self.root_tensor.clone()
        # self.initial_obj_state = self.initial_root_states[:, 1, :3]

        self.dexterous_dof_tensor = self.dof_state_tensor[:, :self.dexterous_num_dofs, :]

        self.dexterous_root_tensor = self.root_tensor[:, 0, :]
        # self.object_root_tensor = self.root_tensor[:, 1, :]

        self.dof_dim = self.dexterous_num_dofs
        self.pos_act = self.initial_dof_states[:, :self.dexterous_num_dofs, 0].clone()
        self.eff_act = torch.zeros(
            (self.num_envs, self.dof_dim), device=self.device)
        self.damping = 0.05
        self.prepare_ik()

    def __del__(self):
        if self.gym is not None:
            self.gym.destroy_sim(self.sim)
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        del self.gym

    def step_sim_q(self, sim_q):
        sim_q = sim_q.to(self.device)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(sim_q))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_action))

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # update viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(
            self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id,
                                      self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._place_agents(
            self.cfg["env"]["numTrain"], self.cfg["env"]["envSpacing"])

    def _load_agent(self, env_ptr, env_id):

        if self.dexterous_loaded == False:
            self.dexterous_actor_list = []
            asset_root = self.asset_root
            dexterous_asset_file = "franka_description/robots/panda.urdf"
            # dexterous_asset_file = "franka_description/robots/movable_dexterous_hand_fine_collision.urdf"
            asset_options = gymapi.AssetOptions()
            asset_options.density = self.cfg['agent']['density']
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
            asset_options.flip_visual_attachments = True
            asset_options.armature = 0.01
            asset_options.use_mesh_materials = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.override_com = True  # recompute center of mesh
            asset_options.override_inertia = True  # recompute inertia
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            # asset_options.vhacd_params.resolution = 3000000
            asset_options.vhacd_params.resolution = 3000000

            self.dexterous_asset = self.gym.load_asset(
                self.sim, asset_root, dexterous_asset_file, asset_options)
            self.dexterous_loaded = True

        dexterous_dof_max_torque, self.dexterous_dof_lower_limits, self.dexterous_dof_upper_limits = self._get_dof_property(
            self.dexterous_asset)

        dof_props = self.gym.get_asset_dof_properties(self.dexterous_asset)
        if self.cfg["env"]["driveMode"] in ["pos", "ik"]:
            dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"][:].fill(self.cfg['agent']['dof_props']['stiffness'])
            dof_props["velocity"][:].fill(self.cfg['agent']['dof_props']['velocity'])
            dof_props["damping"][:].fill(self.cfg['agent']['dof_props']['damping'])
        else:  # osc
            dof_props["driveMode"][:].fill(gymapi.DOF_MODE_EFFORT)
            dof_props["stiffness"][:].fill(0.0)
            dof_props["damping"][:].fill(0.0)
        # root pose
        initial_dexterous_pose = gymapi.Transform()
        initial_dexterous_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        initial_dexterous_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

        # set start dof
        self.dexterous_num_dofs = self.gym.get_asset_dof_count(self.dexterous_asset)

        # default_dof_pos = np.zeros(self.dexterous_num_dofs, dtype=np.float32)
        default_dof_pos = self.init_qpos[env_id, :].cpu()

        # initialize for the pose and rotation
        # default_dof_pos = self.compute_pos_action(self.translation, self.rpy, self.new_joint_angle, env_id)

        dexterous_dof_state = np.zeros_like(
            dexterous_dof_max_torque, gymapi.DofState.dtype)
        dexterous_dof_state["pos"] = default_dof_pos

        dexterous_actor = self.gym.create_actor(
            env_ptr,
            self.dexterous_asset,
            initial_dexterous_pose,
            "dexterous",
            env_id,
            0,
            0)
        for i_body in range(len(self.gym.get_actor_rigid_body_names(env_ptr, dexterous_actor))):
            color_vec = np.random.uniform(0., 1., 3)
            self.gym.set_rigid_body_color(env_ptr, 0, i_body, gymapi.MESH_VISUAL,
                                          gymapi.Vec3(color_vec[0], color_vec[1], color_vec[2]))
        dexterous_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, dexterous_actor)
        for shape in dexterous_shape_props:
            shape.friction = self.cfg['agent']['shape']['friction']

        self.gym.set_actor_rigid_shape_properties(env_ptr, dexterous_actor, dexterous_shape_props)
        self.gym.set_actor_dof_properties(env_ptr, dexterous_actor, dof_props)
        self.gym.set_actor_dof_states(
            env_ptr, dexterous_actor, dexterous_dof_state, gymapi.STATE_ALL)
        self.dexterous_actor_list.append(dexterous_actor)
        self.dexterous_link_dict = self.gym.get_asset_rigid_body_dict(self.dexterous_asset)

    def prepare_ik(self):

        # get dof state tensor
        self.dof_pos = self.dexterous_dof_tensor[:, :, 0].view(self.num_envs, self.dexterous_num_dofs, 1)
        self.dof_vel = self.dexterous_dof_tensor[:, :, 1].view(self.num_envs, self.dexterous_num_dofs, 1)

        # Set action tensors
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        self.effort_action = torch.zeros_like(self.pos_action)

    def step_qpos(self, qpos):
        qpos = qpos.clone().to(self.device)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.pos_action = qpos.clone()
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_action))

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        self.num_sim += 1
        # update viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def get_cur_qpos(self):
        sim_q_state = self.dof_state_tensor[:, :, 0].clone()
        cur_qpos = torch.tensor(sim_q_state, device=self.device)
        return cur_qpos

    def _get_dof_property(self, asset):
        dof_props = self.gym.get_asset_dof_properties(asset)
        dof_num = self.gym.get_asset_dof_count(asset)
        dof_lower_limits = []
        dof_upper_limits = []
        dof_max_torque = []
        for i in range(dof_num):
            dof_max_torque.append(dof_props['effort'][i])
            dof_lower_limits.append(dof_props['lower'][i])
            dof_upper_limits.append(dof_props['upper'][i])
        dof_max_torque = np.array(dof_max_torque)
        dof_lower_limits = np.array(dof_lower_limits)
        dof_upper_limits = np.array(dof_upper_limits)
        return dof_max_torque, dof_lower_limits, dof_upper_limits

    def _load_obj_asset(self):

        self.obj_name_list = []
        self.obj_asset_list = []
        self.table_asset_list = []
        self.obj_pose_list = []
        self.table_pose_list = []
        self.obj_actor_list = []
        self.table_actor_list = []

        used_len = 1
        with tqdm(total=used_len) as pbar:
            pbar.set_description('Loading assets:')
            cur = 0

            obj_asset_list = []
            # prepare the assets to be used
            if self.scene_id is None:
                self.num_scenes = self.cfg['env']['numScenes']
                scene_hash = 'dthvl15jruz9i2fok6bsy3qamp8c4nex'
                for scene_idx in range(self.num_scenes):
                    scene_urdf_path = f"{scene_hash}{str(scene_idx).zfill(3)}/fine_urdf/fine_scene.urdf"
                    print(f'load object asset into IsaacGym | Scene ID: {scene_urdf_path}')
                    object_asset_options = gymapi.AssetOptions()
                    object_asset_options.density = self.cfg['object']['density']
                    # update
                    object_asset_options.linear_damping = self.cfg['object']['damping']['linear']
                    object_asset_options.angular_damping = self.cfg['object']['damping']['angular']

                    object_asset_options.fix_base_link = True
                    object_asset_options.disable_gravity = True
                    object_asset_options.armature = 0.01
                    object_asset_options.use_mesh_materials = True
                    object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
                    object_asset_options.override_com = True  # recompute center of mesh
                    object_asset_options.override_inertia = True  # recompute inertia
                    object_asset_options.vhacd_enabled = True
                    object_asset_options.vhacd_params = gymapi.VhacdParams()
                    object_asset_options.vhacd_params.resolution = 3000000

                    obj_asset = self.gym.load_asset(
                        self.sim, './envs/assets/scene_description', scene_urdf_path, object_asset_options)
                    self.obj_asset_list.append(obj_asset)
                    obj_start_pose = gymapi.Transform()
                    obj_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
                    obj_start_pose.r = gymapi.Quat(0., 0., 0., 1.)
                    self.obj_pose_list.append(obj_start_pose)
            else:
                print(f'load object asset into IsaacGym: {self.scene_id}')
                # scene_urdf_path = f"{self.scene_id}/SceneDescription.urdf"
                scene_urdf_path = f"{self.scene_id}/fine_urdf/fine_scene.urdf"
                object_asset_options = gymapi.AssetOptions()
                object_asset_options.density = self.cfg['object']['density']
                # update
                object_asset_options.linear_damping = self.cfg['object']['damping']['linear']
                object_asset_options.angular_damping = self.cfg['object']['damping']['angular']

                object_asset_options.fix_base_link = True
                object_asset_options.disable_gravity = True
                object_asset_options.armature = 0.01
                object_asset_options.use_mesh_materials = True
                object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
                object_asset_options.override_com = True  # recompute center of mesh
                object_asset_options.override_inertia = True  # recompute inertia
                object_asset_options.vhacd_enabled = True
                object_asset_options.vhacd_params = gymapi.VhacdParams()
                object_asset_options.vhacd_params.resolution = 3000000

                obj_asset = self.gym.load_asset(
                    self.sim, './envs/assets/scene_description', scene_urdf_path, object_asset_options)
                self.obj_asset_list.append(obj_asset)
                obj_start_pose = gymapi.Transform()
                obj_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
                obj_start_pose.r = gymapi.Quat(0., 0., 0., 1.)
                self.obj_pose_list.append(obj_start_pose)

    def _load_obj(self, env_ptr, env_id):

        if self.obj_loaded == False:
            self._load_obj_asset()
            self.obj_loaded = True
        if self.scene_id is None:
            obj_type = env_id % self.num_scenes
            subenv_id = env_id // self.num_scenes
        else:
            obj_type = env_id // self.env_per_object
            subenv_id = env_id % self.env_per_object
        obj_actor = self.gym.create_actor(
            env_ptr,
            self.obj_asset_list[obj_type],
            self.obj_pose_list[obj_type],
            "scene{}-{}".format(obj_type, subenv_id),
            env_id,
            0,
            0)
        # color of scene

        obj_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, obj_actor)
        self.gym.set_rigid_body_color(env_ptr, 1, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.75, 0.75, 0.75))
        for shape in obj_shape_props:
            shape.friction = self.cfg['object']['shape']['friction']
        self.gym.set_actor_rigid_shape_properties(env_ptr, obj_actor, obj_shape_props)

        self.obj_actor_list.append(obj_actor)
        assert(self.gym.get_actor_rigid_body_names(self.env_ptr_list[0], 1)[0] == 'table')

    def _place_agents(self, env_num, spacing):

        print("Simulator: creating agents")

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.space_middle = torch.zeros((env_num, 3), device=self.device)
        self.space_range = torch.zeros((env_num, 3), device=self.device)
        self.space_middle[:, 0] = self.space_middle[:, 1] = 0
        self.space_middle[:, 2] = spacing / 2
        self.space_range[:, 0] = self.space_range[:, 1] = spacing
        self.space_middle[:, 2] = spacing / 2
        num_per_row = int(np.sqrt(env_num))

        with tqdm(total=env_num) as pbar:
            pbar.set_description('Enumerating envs:')
            for env_id in range(env_num):
                env_ptr = self.gym.create_env(
                    self.sim, lower, upper, num_per_row)
                self.env_ptr_list.append(env_ptr)
                self._load_agent(env_ptr, env_id)
                self._load_obj(env_ptr, env_id)
                pbar.update(1)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.1
        plane_params.static_friction = 0.1
        plane_params.dynamic_friction = 0.1
        self.gym.add_ground(self.sim, plane_params)

    def _draw_line(self, src, dst):
        line_vec = np.stack([src, dst]).flatten().astype(np.float32)
        color = np.array([1, 0, 0], dtype=np.float32)
        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(
            self.viewer,
            self.env_ptr_list[0],
            self.env_num,
            line_vec,
            color
        )


def get_sim_param():
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_idx', default=-1, type=int)
    parser.add_argument('--num_envs', default=1024, type=int)
    args = parser.parse_args()

    SCENE_ID = f'dthvl15jruz9i2fok6bsy3qamp8c4nex{str(args.scene_idx).zfill(3)}'
    num_envs = args.num_envs
    init_qpos = torch.tensor([[0., -0.785, 0, -2.356, 0., 1.571, 0.785]], device='cuda').repeat(num_envs, 1)
    sim_params = get_sim_param()
    path = "envs/tasks/franka_panda.yaml"
    with open(path) as f:
        config = yaml.safe_load(f)
        scene_config = None
    if args.scene_idx == -1:
        env = FrankaMotion_Player(config=config, sim_params=sim_params, physics_engine=gymapi.SIM_PHYSX,
                                  device_type="cuda", device_id=0, headless=False,
                                  scene_id=None, init_qpos=init_qpos)
    else:
        env = FrankaMotion_Player(config=config, sim_params=sim_params, physics_engine=gymapi.SIM_PHYSX,
                                  device_type="cuda", device_id=0, headless=False,
                                  scene_id=f"dthvl15jruz9i2fok6bsy3qamp8c4nex000", init_qpos=init_qpos)
    while not env.gym.query_viewer_has_closed(env.viewer):
        env.step_qpos(init_qpos)
        cur_qpos = env.get_cur_qpos()
        print(f'cur pose: {cur_qpos.mean(0)}')