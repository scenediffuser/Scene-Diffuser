from typing import Any, Dict
import cv2 as cv
import os
import glob
import trimesh
import numpy as np
from PIL import Image
from natsort import natsorted
import networkx as nx

if os.environ.get('SLURM') is None:
    if os.environ.get('RENDERING_BACKEND') == "egl":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    elif os.environ.get('RENDERING_BACKEND') == 'osmesa':
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    else:
        pass
    import pyrender

def render_prox_scene(meshes: Dict, camera_pose: np.ndarray, save_path: str, add_axis: bool=True) -> None:
    """ Render prox scene, 

    Args:
        meshes: the trimesh.Trimesh list, contaning scene meshes and bodies meshes
        camera_pose: the camera pose
        save_path: saving path of the rendered image
        add_axis: add axis or not
    """
    ## default configuration in PROX rendering scripts
    H, W = 1080, 1920
    camera_center = np.array([951.30, 536.77])

    camera = pyrender.camera.IntrinsicsCamera(
        fx=1060.53, fy=1060.38,
        cx=camera_center[0], cy=camera_center[1])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

    body_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    
    ## construct scene
    scene = pyrender.Scene()
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    if add_axis:
        axis = trimesh.creation.axis()
        axis = pyrender.Mesh.from_trimesh(axis, smooth=False)
        scene.add(axis)

    for key in meshes:
        if key == 'scenes':
            for mesh in meshes[key]:
                scene_mesh = pyrender.Mesh.from_trimesh(mesh)
                scene.add(scene_mesh, 'mesh')
        elif key == 'bodies':
            for mesh in meshes[key]:
                body_mesh = pyrender.Mesh.from_trimesh(mesh, material=body_material)
                scene.add(body_mesh, 'mesh')
        else:
            raise Exception('Unsupported mesh type.')

    ## rendering
    r = pyrender.OffscreenRenderer(
        viewport_width=W,
        viewport_height=H,
    )
    color, _ = r.render(scene)
    color = color.astype(np.float32) / 255.0
    img = Image.fromarray((color * 255).astype(np.uint8))

    r.delete()
    if save_path is None:
        return img

    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    img.save(save_path)
    
    # r.delete()

def frame2video(frames_path: Any, video: str, start: int=0, framerate=30) -> None:
    """ Convert image frames to video, use ffmpeg to implement the convertion.

    Args:
        frames_dir: image path, a string template
        video: save path of video result
        start: start index
        framerate: the frame rate
    """
    cmd = 'ffmpeg -y -framerate {} -start_number {} -i {} -pix_fmt yuv420p {}'.format(
        framerate, start, frames_path, video)
    os.system(cmd)

def frame2gif(frames: Any, gif: str, size: Any=None, duration: int=33.33, ):
    """ Convert image frames to gif, use PIL to implement the convertion.

    Args:
        frames: a image list or a image directory
        gif: save path of gif result
        size: resize the image into given size, can be tuple or float type
        duration: the duration(ms) of images in gif
    """
    if isinstance(frames, list):
        frames = natsorted(frames)
    elif os.path.isdir(frames):
        frames = natsorted(glob.glob(os.path.join(frames, '*.png')))
    else:
        raise Exception('Unsupported input type.')

    images = []
    for f in frames:
        im = Image.open(f)
        if isinstance(size, tuple):
            im = im.resize(size)
        elif isinstance(size, float):
            im = im.resize((int(im.width / size), int(im.height / size)))
        
        images.append(im)

    img, *imgs = images

    os.makedirs(os.path.dirname(gif), exist_ok=True)
    img.save(fp=gif, format='GIF', append_images=imgs,
            save_all=True, duration=duration, loop=0)

def create_color_array(n: int, c: np.ndarray=np.array([255, 0, 0], dtype=np.uint8)) -> np.ndarray:
    """ Create color array """
    color = np.ones((n, 4), dtype=np.uint8) * 255
    color[:, 0:3] = c
    return color

def get_multi_colors_by_hsl(begin_color, end_color, coe) -> np.ndarray:
    """ Get multi color by interpolation with hsl color format

    Args:
        begin_color: begin color array, RGB color
        end_color: end color array, RGB color
        coe: coefficient <B>
    
    Return:
        RGB color with shape <B, 3>
    """
    begin_color = begin_color.reshape(1,1,3).repeat(len(coe), axis=1)
    begin_rgb = begin_color / 255
    begin_hls = cv.cvtColor(np.array(begin_rgb, dtype=np.float32), cv.COLOR_RGB2HLS)
    end_color = end_color.reshape(1,1,3).repeat(len(coe), axis=1)
    end_rgb = end_color / 255
    end_hls = cv.cvtColor(np.array(end_rgb, dtype=np.float32), cv.COLOR_RGB2HLS)

    hls = ((end_hls - begin_hls) * coe.reshape(-1, 1).repeat(3, axis=1) + begin_hls)
    rgb = cv.cvtColor(np.array(hls, dtype=np.float32), cv.COLOR_HLS2RGB)
    return (rgb*255).astype(np.uint8).reshape(-1, 3)

def create_trimesh_node(node: np.ndarray, radius: float=0.1, 
    color: np.ndarray=np.array([255, 0, 0], dtype=np.ndarray)) -> trimesh.Trimesh:
    """ Create trimesh node for visualization 
    
    Args:
        node: node position, <2> or <3>
        radius: ball radius for visualization
    
    Return:
        A trimesh.Trimesh obejct
    """
    if len(node) == 2:
        node = np.array([*node, 1.0], dtype=np.float32)
    m = np.eye(4, dtype=np.float32)
    m[0:3, -1] = node
    node_ball = trimesh.creation.uv_sphere(radius=radius)
    node_ball.visual.vertex_colors = create_color_array(len(node_ball.vertices), color)
    
    node_ball.apply_transform(m)
    return node_ball

def create_trimesh_nodes_path(nodes: np.ndarray, radius: float=0.1, merge: bool=False) -> Any:
    """ Create trimesh nodes

    Args:
        nodes: nodes with shape <N, 2> or <N, 3>
        radius: ball radius for visualization
        merge: merge the node meshes
    
    Return:
        trimesh node list
    """
    end_color = np.array([204, 8, 8], dtype=np.uint8)
    begin_color = np.array([245, 171, 171], dtype=np.uint8)

    coe = np.linspace(0, 1, len(nodes))
    colors = get_multi_colors_by_hsl(begin_color, end_color, coe)
    
    node_meshes = []
    for i in range(len(nodes)):
        node_meshes.append(create_trimesh_node(nodes[i], radius, colors[i]))
    
    if merge:
        traj_verts = []
        traj_color = []
        traj_faces = []
        offset = 0
        for m in node_meshes:
            traj_verts.append(m.vertices)
            traj_color.append(m.visual.vertex_colors)
            traj_faces.append(m.faces + offset)
            offset += len(m.vertices)
        traj_verts = np.concatenate(traj_verts, axis=0)
        traj_color = np.concatenate(traj_color, axis=0)
        traj_faces = np.concatenate(traj_faces, axis=0)
        return trimesh.Trimesh(vertices=traj_verts, faces=traj_faces, vertex_colors=traj_color)
    else:
        return node_meshes

def get_rotation_matrix_from_two_vectors(v1, v2):
    """ Compute rotation matrix from two vectors """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    sita = np.arccos(np.dot(v1, v2))
    norm_v = np.cross(v1, v2)
    norm_v = norm_v / np.linalg.norm(norm_v)

    norm_v_invert = np.array([
        [0, -norm_v[2], norm_v[1]],
        [norm_v[2], 0, -norm_v[0]],
        [-norm_v[1], norm_v[0], 0]
    ], dtype=np.float32)

    # R_ = Q(axis=norm_v, angle=sita)
    # print(R_.rotation_matrix)

    R = np.eye(3) + np.sin(sita) * norm_v_invert + (norm_v_invert @ norm_v_invert) * (1 - np.cos(sita))
    return R

def create_trimesh_edge(edge, radius=0.01):
    """ Create trimesh edge for visualization """
    e_n1, e_n2 = edge
    if len(e_n1) == 2:
        e_n1 = np.array([*e_n1, 1.0], dtype=np.float32)
    if len(e_n2) == 2:
        e_n2 = np.array([*e_n2, 1.0], dtype=np.float32)

    height = np.sqrt(((e_n1 - e_n2) ** 2).sum())
    edge_line = trimesh.creation.cylinder(radius, height=height)
    edge_line.visual.vertex_colors = create_color_array(len(edge_line.vertices), c=np.array([128, 0, 0], dtype=np.uint8))

    m = np.eye(4, dtype=np.float32)
    ## rotation
    origin_vector = np.array([0, 0, 1.0], dtype=np.float32)
    final_vector = e_n1 - e_n2
    m[0:3, 0:3] = get_rotation_matrix_from_two_vectors(origin_vector, final_vector)

    ## translation
    m[0:3, -1] = 0.5 * (e_n1 + e_n2)

    edge_line.apply_transform(m)
    return edge_line

def visualize_scene_and_nav_graph(scene: Any, G: nx.Graph) -> None:
    """ visualize a nav graph in a scene """
    S = trimesh.Scene()
    S.add_geometry(scene)

    for n in list(G.nodes):
        S.add_geometry(create_trimesh_node(
            G.nodes[n]['position']
        ))
    
    for e in list(G.edges):
        e_n1, e_n2 = e
        S.add_geometry(create_trimesh_edge((
            G.nodes[e_n1]['position'],
            G.nodes[e_n2]['position']
        )))
    
    S.add_geometry(trimesh.creation.axis())
    S.show()

def render_scannet_path(meshes: Dict, camera_pose: np.ndarray, save_path: str, add_axis: bool=True):
    """ Render scannet scene and path 

    Args:
        meshes: the trimesh.Trimesh list, contaning scene meshes, start position, path meshes
        camera_pose: the camera pose
        save_path: saving path of the rendered image
        add_axis: add axis or not
    """
    H, W = 1080, 1920
    scene = pyrender.Scene()
    camera = pyrender.camera.IntrinsicsCamera(
        fx=1060, fy=1060,
        cx=951.30, cy=536.77)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)

    if add_axis:
        axis_mesh = trimesh.creation.axis(origin_size=0.02)
        scene.add(pyrender.Mesh.from_trimesh(axis_mesh, smooth=False), 'mesh_axis')
    for key in meshes:
        scene.add(pyrender.Mesh.from_trimesh(meshes[key], smooth=False), f'mesh_{key}')

    r = pyrender.OffscreenRenderer(viewport_width=W,viewport_height=H)
    color, _ = r.render(scene)
    color = color.astype(np.float32) / 255.0
    img = Image.fromarray((color * 255).astype(np.uint8))
    
    r.delete()
    if save_path is None:
        return img
    
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    img.save(save_path)

    # r.delete()

if __name__ == '__main__':
    nodes = np.linspace(np.array([0,0,0]), np.array([5,5,5]), 32)
    S = trimesh.Scene()
    node_meshes = create_trimesh_nodes_path(nodes)
    traj_verts = []
    traj_color = []
    traj_faces = []
    offset = 0
    for m in node_meshes:
        traj_verts.append(m.vertices)
        traj_color.append(m.visual.vertex_colors)
        traj_faces.append(m.faces + offset)
        offset += len(m.vertices)
    traj_verts = np.concatenate(traj_verts, axis=0)
    traj_color = np.concatenate(traj_color, axis=0)
    traj_faces = np.concatenate(traj_faces, axis=0)
    trimesh.Trimesh(vertices=traj_verts, faces=traj_faces, vertex_colors=traj_color).show()
    