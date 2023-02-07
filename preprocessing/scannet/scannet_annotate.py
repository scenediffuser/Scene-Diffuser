import glob
import os
from turtle import position
import numpy as np
import trimesh
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import networkx as nx
from networkx.readwrite import json_graph
from utils import random_str
import pickle

scannet_dir = '/home/wangzan/Data/scannet/preprocessing/scannet_scenes'

## https://kaldir.vc.in.tum.de/scannet_browse/scans/scannet/grouped
scans = [
'scene0000_00', 'scene0231_00', 'scene0276_00', 'scene0515_00', 'scene0673_00', # apartment
'scene0012_00', 'scene0022_00', 'scene0151_00', 'scene0160_00', 'scene0192_00', 'scene0247_00', 'scene0281_00', 'scene0294_00', 'scene0297_00', 'scene0588_00', 'scene0603_00', 'scene0672_00', 'scene0694_00', # lounge
'scene0005_00', 'scene0370_00', 'scene0261_00', 'scene0678_00', # misc
'scene0006_00', 'scene0051_00', 'scene0137_00', 'scene0296_00', 'scene0435_00', 'scene0477_00', 'scene0640_00', 'scene0645_00', 'scene0698_00', # bedroom
'scene0008_00', 'scene0132_00', 'scene0134_00', 'scene0145_00', 'scene0199_00', 'scene0202_00', 'scene0269_00', 'scene0317_00', 'scene0363_00', 'scene0536_00', 'scene0549_00', 'scene0637_00', 'scene0641_00', # lobby
'scene0025_00', 'scene0040_00', 'scene0114_00', 'scene0142_00', 'scene0309_00', 'scene0505_00', 'scene0626_00', 'scene0653_00',  # office
'scene0030_00', 'scene0420_00', 'scene0621_00', # classromm
'scene0064_00', 'scene0187_00', 'scene0667_00', # bookstore
'scene0122_00', 'scene0403_00', 'scene0634_00', # conferenceroom
]


class Annotator(object):
    def __init__(self, empty_area, object_area, scan) -> None:
        self.empty = empty_area[:, 0:2]
        self.object = object_area[:, 0:2]
        self.scan = scan

        self.G = nx.Graph()
        self.path = []
        self.position = {}

    def _path(self):
        for i in range(len(self.path) - 1):
            self.G.add_edge(self.path[i], self.path[i + 1])

        self.path = []
        print('create path\n')

    def _done(self):

        for key in self.position:
            assert key in self.G.nodes
            self.G.nodes[key]['position'] = self.position[key]

        for (u, v) in self.G.edges:
            self.G[u][v]['weight'] = np.linalg.norm(self.G.nodes[u]['position'] - self.G.nodes[v]['position'])

        ## save and exit
        Gdata = json_graph.node_link_data(self.G)
        save_p = f'./graph/{scan}.pkl'
        os.makedirs(os.path.dirname(save_p), exist_ok=True)
        with open(save_p, 'wb') as fp:
            pickle.dump(Gdata, fp)
        
        print('save annotated graph..\n')
        ## close fig
        plt.close(self.fig)

    def _get_node_id(self, node_pos, threshold=0.1):
        for k in self.position:
            if np.linalg.norm(self.position[k] - node_pos) < threshold:
                return k
        
        return random_str(20)

    def _update_canvas(self):
        plt.axis("equal")

        plt.scatter(self.empty[:, 0], self.empty[:, 1], s=10, color='aquamarine')
        plt.scatter(self.object[:, 0], self.object[:, 1], s=10, color='royalblue')

        path_nodes = np.array([self.position[p] for p in self.path])
        if len(path_nodes) == 0:
            return

        ## draw path
        plt.scatter(path_nodes[:, 0], path_nodes[:, 1], s=50, color='red', marker='s')
        for i in range(len(path_nodes) - 1):
            x1, y1 = path_nodes[i]
            x2, y2 = path_nodes[i + 1]
            plt.plot([x1, x2], [y1, y2], color='red')
        
        ## draw graph
        for node in self.G.nodes:
            plt.scatter(self.position[node][0], self.position[node][1], s=50, color='red', marker='s')
        for u, v in self.G.edges:
            x1, y1 = self.position[u]
            x2, y2 = self.position[v]
            plt.plot([x1, x2], [y1, y2], color='red')
        
    def add_node(self, event):
        node_pos = np.array([event.xdata, event.ydata], dtype=np.float32)
        node_id = self._get_node_id(node_pos)

        if node_id not in self.position:
            self.position[node_id] = node_pos
        
        self.path.append(node_id)
        print('add node', node_id, node_pos)
        print(self.path)
        print()

        ## update canvas
        plt.clf()

        self._update_canvas()

        plt.draw()
    
    def key_press(self, event):
        if event.key == 'd':
            self._done()
        elif event.key == ' ':
            self._path()
        else:
            print(f'press key {event.key}')

    def showScene(self, ):
        self.fig = plt.figure()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        self.fig.canvas.mpl_connect('button_press_event', self.add_node)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)

        self._update_canvas()

        plt.show()
        plt.draw()

def annotate_nav_graph(scan_id):
    if os.path.exists(f'./graph/{scan_id}.pkl'):
        print(f'{scan_id} graph already exists.')
        return
    
    points = np.load(os.path.join(scannet_dir, f'{scan_id}.npy'))
    ceiling_mask = points[:, -1]
    
    mask = points[:, -1] == 0 # floor mask, see lable id https://github.com/Silverster98/point_transformer.scannet/blob/45b91444df2854d1bb1b63222b01ece95294289d/utils/config.py#L25
    
    # verts = points[:, 0:3]
    # color = np.ones((len(verts), 4), dtype=np.uint8) * 255
    # color[:, 0:3] = np.array([255, 0, 0], dtype=np.uint8)
    # color[mask, 0:3] = np.array([0, 255, 0], dtype=np.uint8)
    # S = trimesh.Scene()
    # S.add_geometry(trimesh.PointCloud(vertices=verts, colors=color))
    # S.add_geometry(trimesh.creation.axis())
    # S.show()

    empty_area = points[mask, :]
    object_area = points[~mask, :]

    anno = Annotator(empty_area, object_area, scan_id)
    anno.showScene()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default='all')
    args = parser.parse_args()

    if args.scene == 'all':
        for scan in scans:
            annotate_nav_graph(scan)
    else:
        if args.scene in scans:
            annotate_nav_graph(args.scene)
        else:
            raise Exception('Unknown scans.')


