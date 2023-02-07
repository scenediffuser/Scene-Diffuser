import os
import sys
import json
import time
import numpy as np
from easydict import EasyDict
from plyfile import PlyData, PlyElement
import trimesh

scene_dir = '/home/wangzan/Data/SHADE/PROX/scenes/'
preprocess_scenes_dir = '/home/wangzan/Data/SHADE/PROX/preprocess_scenes/'
scene_name = ['BasementSittingBooth', 'MPH11', 'MPH112', 'MPH8', 'N0Sofa', 'N3Library', \
    'MPH16', 'MPH1Library', 'N0SittingBooth', 'N3OpenArea', 'N3Office', 'Werkraum']

NUM_MAX_PTS = 100000

def read_ply_xyzrgbnormal(filename):
    """ read XYZ RGB normals point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    scene = trimesh.load(filename)

    pc = scene.vertices
    color = scene.visual.vertex_colors
    normal = scene.vertex_normals

    vertices = np.concatenate((pc, color[:, 0:3], normal), 1)
    return vertices

def collect_one_scene_data_label(scene_name, out_filename):
    # Over-segmented segments: maps from segment to vertex/point IDs
    ply_filename = os.path.join(scene_dir, scene_name+'.ply')
    
    # Raw points in XYZRGBA
    points = read_ply_xyzrgbnormal(ply_filename)

    # Refactor data format
    instance_labels = np.zeros((len(points), 1))
    semantic_labels = np.zeros((len(points), 1))
    data = np.concatenate((points, instance_labels, semantic_labels), 1)

    if data.shape[0] > NUM_MAX_PTS:
        choices = np.random.choice(data.shape[0], NUM_MAX_PTS, replace=False)
        data = data[choices]

    print("shape of subsampled scene data: {}".format(data.shape))
    np.save(out_filename, data)

if __name__=='__main__':
    os.makedirs(preprocess_scenes_dir, exist_ok=True)
    
    for i, scene_name in enumerate(scene_name):
        start = time.time()
        out_filename = scene_name+'.npy' # scene0000_00.npy
        collect_one_scene_data_label(scene_name, os.path.join(preprocess_scenes_dir, out_filename))

    print("done!")