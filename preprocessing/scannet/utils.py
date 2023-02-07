import string
import random
from typing import List, Any
import trimesh
import numpy as np
import networkx as nx

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


def create_color_array(n, c=np.array([255, 0, 0], dtype=np.uint8)):
    """ Create color array """
    color = np.ones((n, 4), dtype=np.uint8) * 255
    color[:, 0:3] = c
    return color

def create_trimesh_node(node, radius=0.08):
    """ Create trimesh node for visualization """
    if len(node) == 2:
        node = np.array([*node, 1.0], dtype=np.float32)
    m = np.eye(4, dtype=np.float32)
    m[0:3, -1] = node
    node_ball = trimesh.creation.uv_sphere(radius=radius)
    node_ball.visual.vertex_colors = create_color_array(len(node_ball.vertices))
    
    node_ball.apply_transform(m)
    return node_ball

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

def random_str(length: int=4) -> str:
    """ Generate random string with given length
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

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

def visualize_scene_and_path(scene: Any, path: List, G: nx.Graph=None, visualize_edge=True) -> None:
    """ visualize a path in a scene """
    S = trimesh.Scene()
    S.add_geometry(scene)

    if G is not None:
        nodes = []
        for nid in path:
            nodes.append(G.nodes[nid]['position'])
        path = np.array(nodes)

    for n in path:
        S.add_geometry(create_trimesh_node(
            n
        ))
    
    if visualize_edge:
        for i in range(len(path) - 1):
            e_n1, e_n2 = path[i], path[i + 1]
            S.add_geometry(create_trimesh_edge((
                e_n1,
                e_n2
            )))
    
    S.add_geometry(trimesh.creation.axis())
    S.show()