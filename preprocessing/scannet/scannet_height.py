from typing import List
import numpy as np
import glob
import trimesh
import os
from natsort import natsorted
from sklearn.neighbors import KDTree
import argparse
import pickle

def get_height_map(points: np.ndarray, HEIGHT_MAP_DIM: int=256) -> List:
    """ Load region meshes of a scenes

    Args:
        points: scene point cloud
        HEIGHT_MAP_DIM: height map dimension
    
    Return:
        Return the floor height map and axis-aligned scene bounding box
    """
    ## compute floor height map
    minx, miny = points[:, 0].min(), points[:, 1].min()
    maxx, maxy = points[:, 0].max(), points[:, 1].max()

    x = np.linspace(minx, maxx, HEIGHT_MAP_DIM)
    y = np.linspace(miny, maxy, HEIGHT_MAP_DIM)
    xx, yy = np.meshgrid(x, y)
    pos2d = np.concatenate([xx[..., None], yy[..., None]], axis=-1)

    floor_mask = points[:, -1] == 0
    floor_xyz = points[floor_mask, 0:3]

    floor_kdtree = KDTree(floor_xyz[:, 0:2], leaf_size=0.6 * len(floor_xyz))
    neigh_idx = floor_kdtree.query(pos2d.reshape((-1, 2)), k=1, return_distance=False, dualtree=True)
    neigh_idx = neigh_idx.reshape(-1)
    height_map = floor_xyz[neigh_idx, 2]
    height_map = height_map.reshape(xx.shape)
    

    ## visualize
    # height_pos = np.concatenate([xx[..., None], yy[..., None], height_map[..., None]], axis=-1).reshape(-1, 3)
    # S = trimesh.Scene()
    # floor_pc = trimesh.PointCloud(vertices=floor_xyz)

    # colors = np.ones((len(height_pos), 4), dtype=np.uint8) * 255
    # colors[:, 1:3] = 0
    # height_pos[:, -1] += 1.0
    # height_pos_pc = trimesh.PointCloud(vertices=height_pos, colors=colors)
    # S.add_geometry(floor_pc)
    # S.add_geometry(height_pos_pc)
    # S.show()

    return height_map, minx, maxx, miny, maxy

if __name__ == '__main__':
    height_map_dim = 256

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default='./path_planning')
    args = parser.parse_args()

    graphs = natsorted(glob.glob('./graph/*.pkl'))
    for i, gp in enumerate(graphs):
        scan_id = gp.split('/')[-1][:-4]

        points = np.load(os.path.join(args.out_dir, 'scene', f'{scan_id}.npy'))

        # verts = points[:, 0:3]
        # color = np.ones((len(verts), 4), dtype=np.uint8) * 255
        # color[:, 0:3] = points[:, 3:6].astype(np.uint8)
        # S = trimesh.Scene()
        # S.add_geometry(trimesh.PointCloud(vertices=verts, colors=color))
        # S.add_geometry(trimesh.creation.axis())
        # S.show()

        height_map, minx, maxx, miny, maxy = get_height_map(points, HEIGHT_MAP_DIM=height_map_dim)

        save_p_height = os.path.join(args.out_dir, 'height', f'{scan_id}.pkl')
        os.makedirs(os.path.dirname(save_p_height), exist_ok=True)
        with open(save_p_height, 'wb') as fp:
            pickle.dump(
                {'dim': height_map_dim, 'height': height_map, 'minx': minx, 'maxx': maxx, 'miny': miny, 'maxy': maxy}, fp
            )

        

