import torch
import trimesh
import numpy as np
import torch.nn.functional as F

def debug_visualize_prox_pose(dataloader):
    """ visualize prox pose dataset for debug
    """
    for data in dataloader:
        for key in data:
            if isinstance(data[key], (np.ndarray, torch.Tensor)):
                print(key, data[key].shape, data[key].dtype)
            else:
                print(key, data[key])
        normalizer = dataloader.dataset.normalizer

        B = len(data['x'])
        x = data['x']
        if normalizer is not None:
            x = normalizer.unnormalize(x)
        
        pos = data['pos'].reshape(B, -1, 3)
        N = pos.shape[1]
        feat = data['feat'].reshape(B, N, -1)
        scene_id = data['scene_id']
        cam_tran = data['cam_tran']
        origin_cam_tran = data['origin_cam_tran']
        gender = data['gender']

        for i in range(B):
            s = scene_id[i]
            scene_mesh = dataloader.dataset.scene_meshes[s].copy()
            pcd = pos[i].numpy()
            color = (feat[i][:, 0:3].numpy() * 255).astype(np.uint8)
            c_t = cam_tran[i]
            o_c_t = origin_cam_tran[i]
            smplx_param = x[i]

            scene_trans = c_t @ np.linalg.inv(o_c_t) # scene_T @ origin_cam_T = cur_cam_T
            scene_mesh = scene_mesh.apply_transform(scene_trans)

            body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_param, gender[i])
            body_verts = body_verts.numpy()
            body_mesh = trimesh.Trimesh(vertices=body_verts, faces=body_faces, process=False)

            S = trimesh.Scene()
            S.add_geometry(scene_mesh)
            S.add_geometry(body_mesh)
            S.add_geometry(trimesh.creation.axis())
            S.show()

            S = trimesh.Scene()
            S.add_geometry(trimesh.PointCloud(vertices=pcd, colors=color))
            S.add_geometry(body_mesh)
            S.add_geometry(trimesh.creation.axis())
            S.show()
        break

def debug_visualize_prox_motion(dataloader):
    """ visualize prox motion dataset for debug
    """
    for data in dataloader:
        for key in data:
            if isinstance(data[key], (np.ndarray, torch.Tensor)):
                print(key, data[key].shape, data[key].dtype)
            else:
                print(key, data[key])
        
        normalizer = dataloader.dataset.normalizer
        repr_type = dataloader.dataset.repr_type
        
        B = len(data['x'])
        x = data['x']
        if normalizer is not None:
            x = normalizer.unnormalize(x)

        B, O, D = data['start'].shape
        x[:, 0:O, :] = data['start'].clone() # copy start observation to x after unnormalize
        if repr_type == 'absolute':
            pass
        elif repr_type == 'relative':
            x[:, O-1:, :] = torch.cumsum(x[:, O-1:, :], dim=1)
        else:
            raise Exception('Unsupported repr type.')

        pos = data['pos'].reshape(B, -1, 3)
        N = pos.shape[1]
        feat = data['feat'].reshape(B, N, -1)
        scene_id = data['scene_id']
        cam_tran = data['cam_tran']
        origin_cam_tran = data['origin_cam_tran']
        gender = data['gender']

        for i in range(B):
            s = scene_id[i]
            scene_mesh = dataloader.dataset.scene_meshes[s].copy()
            pcd = pos[i].numpy()
            color = (feat[i][:, 0:3].numpy() * 255).astype(np.uint8)
            c_t = cam_tran[i]
            o_c_t = origin_cam_tran[i]
            smplx_param = x[i]

            scene_trans = c_t @ np.linalg.inv(o_c_t) # scene_T @ origin_cam_T = cur_cam_T
            scene_mesh = scene_mesh.apply_transform(scene_trans)

            body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_param, gender[i])
            body_verts = body_verts.numpy()
            body_meshes = []
            for i in range(0, len(body_verts), 2):
                body_mesh = trimesh.Trimesh(vertices=body_verts[i], faces=body_faces, process=False)
                body_meshes.append(body_mesh)

            S = trimesh.Scene()
            S.add_geometry(scene_mesh)
            S.add_geometry(body_meshes)
            S.add_geometry(trimesh.creation.axis())
            S.show()

            S = trimesh.Scene()
            S.add_geometry(trimesh.PointCloud(vertices=pcd, colors=color))
            S.add_geometry(body_meshes)
            S.add_geometry(trimesh.creation.axis())
            S.show()
        break

def debug_visualize_path_data(dataloader):
    """ Visualize path dataset for debug
    """
    from utils.visualize import create_trimesh_node, create_trimesh_nodes_path
    
    for data in dataloader:
        for key in data:
            if isinstance(data[key], (np.ndarray, torch.Tensor)):
                print(key, data[key].shape, data[key].dtype)
            else:
                print(key, data[key])
        
        B = len(data['offset'])
        N = data['offset'][0].item()

        normalizer = dataloader.dataset.normalizer
        repr_type = dataloader.dataset.repr_type

        x = data['x'] # <B, T, D>
        if normalizer is not None:
            x = normalizer.unnormalize(x)
        
        B, O, D = data['start'].shape
        x[:, 0:O, :] = data['start'].clone() # copy start observation to x after unnormalize

        if repr_type == 'absolute':
            pass
        elif repr_type == 'relative':
            x[:, O-1:, :] = torch.cumsum(x[:, O-1:, :], dim=1)
        else:
            raise Exception('Unsupported repr type.')

        pos = data['pos'].reshape(B, N, 3)
        feat = data['feat'].reshape(B, N, -1)
        scene_id = data['scene_id']
        trans_mat = data['trans_mat']
        target = data['target']
        s_grid_map = data['s_grid_map']
        s_grid_min = data['s_grid_min']
        s_grid_max = data['s_grid_max']
        s_grid_dim = data['s_grid_dim']

        
        for i in range(B):
            pcd = pos[i].numpy()
            color = (feat[i][:, 0:3].numpy() * 255).astype(np.uint8)

            path = x[i].numpy()
            target_pos = target[i].numpy()
            
            ## visualize scene and path in pcd
            S = trimesh.Scene()
            S.add_geometry(trimesh.PointCloud(vertices=pcd, colors=color))

            S.add_geometry(
                create_trimesh_nodes_path(path)
            )
            S.add_geometry(
                create_trimesh_node(target_pos, color=np.array([0, 255, 0], dtype=np.uint8))
            )

            S.add_geometry(trimesh.creation.axis())
            S.show()

            ## visualize height map
            S = trimesh.Scene()
            pcd = trimesh.transform_points(pcd, np.linalg.inv(trans_mat[i]))
            S.add_geometry(trimesh.PointCloud(vertices=pcd, colors=color))

            minx, miny = s_grid_min[i]
            maxx, maxy = s_grid_max[i]
            dim = s_grid_dim[i]
            x_ = torch.linspace(minx, maxx, dim)
            y_ = torch.linspace(miny, maxy, dim)
            xx, yy = torch.meshgrid(x_, y_) # <H, W>
            
            pos2d = torch.cat([xx[..., None], yy[..., None]], axis=-1).reshape(-1, 2).unsqueeze(0)
            norm_pos = ((pos2d - s_grid_min[i:i+1].unsqueeze(1)) 
                            / (s_grid_max[i:i+1].unsqueeze(1) - s_grid_min[i:i+1].unsqueeze(1)) * 2 -1) # <B, N, 2>
            
            n_verts = norm_pos.shape[1]
            zz = F.grid_sample(
                s_grid_map[i:i+1].unsqueeze(1),   # <B, 1, H, W>
                norm_pos.view(-1, n_verts, 1, 2), # <B, N, 1, 2>
                padding_mode='border', align_corners=True)
            zz = zz.view(*xx.shape)

            height_nodes = torch.cat([xx[..., None], yy[..., None], zz[..., None]], axis=-1).reshape(-1, 3).numpy()
            S.add_geometry(trimesh.PointCloud(vertices=height_nodes))
            
            S.add_geometry(trimesh.creation.axis())
            S.show()
            

        exit(0)