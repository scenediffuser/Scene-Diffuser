import pickle
import os
from typing import List
import numpy as np
import networkx as nx
import glob
import trimesh
import argparse
from sklearn.neighbors import KDTree
from natsort import natsorted
from networkx.readwrite import json_graph

from utils import visualize_scene_and_nav_graph, create_trimesh_node

scannet_dir = '/home/wangzan/Data/scannet/preprocessing/scannet_scenes'

def start_satisfy(start: str, selected_nodes: List, G: nx.Graph, points: np.ndarray, min_hops=1, 
    robot_radius=0.08, robot_bottom=0.1, robot_top=0.6) -> bool:
    """ Select start from G, the input start node should have more than `min_hops` hops 
    with the nodes in `selected_nodes`.

    Args:
        start: canditate start node in G
        selected_nodes: selected node list
        G: the graph
        points: scene point cloud
        min_hop: minimal hops between start and selected nodes
    """
    satisfy = True
    for n in selected_nodes:
        try:
            hop = nx.dijkstra_path_length(G, start, n, weight=lambda u, v, d: 1)
        except:
            continue

        if hop < min_hops:
            satisfy = False
            break
    
    start_pos = G.nodes[start]['position']

    ## check collision
    floor_xyz = points[points[:, -1] == 0, 0:3]
    floor_kdtree = KDTree(floor_xyz[:, 0:2], leaf_size=0.6 * len(floor_xyz))
    neigh_idx = floor_kdtree.query(start_pos.reshape((-1, 2)), k=10, return_distance=False)[0]
    neigh_height = floor_xyz[neigh_idx, -1].mean()

    mask = np.logical_and(points[:, 2] > (neigh_height + robot_bottom), points[:, 2] < (neigh_height + robot_top))
    dist = np.linalg.norm(points[mask, 0:2] - start_pos, axis=-1)
    if dist.min() < robot_radius:
        satisfy = False
    
    # if not satisfy:
    #     # visualize
    #     print(start_pos)
    #     S = trimesh.Scene()
    #     S.add_geometry(create_trimesh_node(start_pos))
    #     S.add_geometry(trimesh.PointCloud(vertices=points[:, 0:3]))
    #     S.add_geometry(trimesh.creation.axis())
    #     S.show()
    
    return satisfy

def end_satisfy(start: str, end: str, G: nx.Graph, min_hops: int=2, max_hops: int=10) -> bool:
    """ Select end from G, the input end node should have more than `min_hops` hops and less 
    than `max_hops` with the given start nodes.

    Args:
        start: start node
        end: candidate end node
        G: the graph
        min_hop: minimal hops between start and end nodes
        max_hop: maximal hops between start and end nodes
    """
    try:
        hop = nx.dijkstra_path_length(G, start, end, weight=lambda u, v, d: 1)
    except:
        return False
    
    if hop < min_hops or hop > max_hops:
        return False

    return True

def refine_path(path: List, MAX_SEG=0.08) -> np.ndarray:
    """ Refine Path, split the path segment between two node if the path segment is too long

    Args:
        path: original path sampled from nav graph
    
    Return:
        Refined path
    """
    refined_path = []
    for i in range(len(path) - 1):
        s = path[i]
        e = path[i+1]

        dist = np.linalg.norm(s[0:2] - e[0:2]) # use 2D dist to split
        nsplit = int(dist // MAX_SEG)

        refined_path.append(s)
        for t in range(nsplit):
            k = (t + 1) / (nsplit + 1)
            split_pos = s * (1 - k) + e * k
            refined_path.append(split_pos)
        
    refined_path.append(path[-1])
    return refined_path

def generate_path(G: nx.Graph, points: np.ndarray, nstart_ratio: float=0.5, nend_each_start: int=6,
    path_min_length: int=16, path_max_length: int=120) -> List:
    """ Sample path from a given Graph

    Args:
        G: the graph
        points: scene points
    
    Return:
        The sampled path
    """
    path_list = []

    ## sample start and end position
    nodes = G.nodes.data('position')
    ids = [item[0] for item in nodes]
    pos = [item[1] for item in nodes]
    pos = np.array(pos, dtype=np.float32)
    num = len(ids)

    start_end_pairs = {}
    ## 1. sample start position
    start_candidate = list(range(num))
    for _ in range(int(nstart_ratio * num)): # select `int(nstart_ratio * num)` node at most
        if len(start_candidate) == 0:
                break
        
        start = None
        while True:
            ## if the cadidate list is empty or find a valid start node in previous loop
            if len(start_candidate) == 0 or start is not None:
                break

            s = np.random.choice(start_candidate) # random select from cadidate list
            if start_satisfy(ids[s], list(start_end_pairs.keys()), G, points): # test validity
                start = ids[s]
            
            start_candidate.remove(s) # remove the node from cadidate list
        
        if start is not None:
            start_end_pairs[start] = [] # put the valid start in start_end_pairs dict
    
    ## 2. select `nend_each_start` end position for each start
    for start in start_end_pairs:
        end_candidate = list(range(num))
        end_candidate.remove(ids.index(start))

        for _ in range(nend_each_start):
            if len(end_candidate) == 0:
                break

            end = None
            while True:
                if len(end_candidate) == 0 or end is not None:
                    break

                e = np.random.choice(end_candidate)
                if end_satisfy(start, ids[e], G):
                    end = ids[e]
                
                end_candidate.remove(e)

            if end is not None:
                start_end_pairs[start].append(end)

    ## generate one path with given start and end node
    for start in start_end_pairs:
        for end in start_end_pairs[start]:
            ## generate path
            try:
                path = nx.dijkstra_path(G, start, end, weight='weight')
            except:
                continue

            path_pos = [G.nodes[n]['position'] for n in path]
            refined_path = np.array(refine_path(path_pos))

            if len(refined_path) < path_min_length or len(refined_path) > path_max_length:
                continue

            path_list.append((path, refined_path))

    return path_list

if __name__ == '__main__':
    np.random.seed(2022) # fix random seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default='./path_planning')
    args = parser.parse_args()

    total = 0

    graphs = natsorted(glob.glob('./graph/*.pkl'))
    for i, gp in enumerate(graphs):
        scan_id = gp.split('/')[-1][:-4]
        print(i, scan_id)

        ## load graph
        with open(f'./graph/{scan_id}.pkl', 'rb') as fp:
            graph_data = pickle.load(fp)
            G = json_graph.node_link_graph(graph_data)

        ## load scene point cloud
        points = np.load(os.path.join(scannet_dir, f'{scan_id}.npy'))

        # verts = points[:, 0:3]
        # color = np.ones((len(verts), 4), dtype=np.uint8) * 255
        # color[:, 0:3] = points[:, 3:6].astype(np.uint8)
        # scene = trimesh.PointCloud(vertices=verts, colors=color)
        # visualize_scene_and_nav_graph(scene, G)
        # continue

        path_list = generate_path(G, points, nstart_ratio=1.0, path_min_length=32)
        if len(path_list) == 0:
            print(f'Skip {scan_id} beacause there is no path in this scene.')
            continue

        save_p_path = os.path.join(args.out_dir, 'path', f'{scan_id}.pkl')
        os.makedirs(os.path.dirname(save_p_path), exist_ok=True)
        with open(save_p_path, 'wb') as fp:
            pickle.dump(path_list, fp)
        
        total += len(path_list)
    
    print(total)
        
