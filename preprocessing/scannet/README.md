# Preprocess ScanNet data for path planning in indoor scenes

You can use our [pre-processed data](https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing) if you don't want to process it by yourself.

The data is organized as follows:

```bash
-| scannet_path_planning
---| height
-----| scene0000_00.pkl
-----| ...
---| mesh
-----| scene0000_00_vh_clean_2.ply # original ScanNet-V2 mesh
-----| ...
---| path
-----| scene0000_00.pkl
-----| ...
---| scene
-----| scene0000_00.npy
-----| ...
```

## Data preparation

Download scannet dataset, and use the scripts in [https://github.com/Silverster98/point_transformer.scannet](https://github.com/Silverster98/point_transformer.scannet) to preprocess the scene.

Notes: you can only preprocess the selected scene, which is listed in `preprocessing/scannet/scannet_annotate.py`.

## Annotate navigation graph mannually

By running:

```bash
python scannet_annotate.py
```

This script will provide a UI for user to annotate graph. Following the below tips to use this tool.

Tips:

1. Click the canvas created by Matportlib, and this program will add a graph node in the scene.

2. Continue to click, the nodes will connect in sequence, which means the progress will add edges.

3. Press 'space' key, the progress will start a new path, the previous path will be stored. After creating several paths, all the paths will grouped as a graph.

    Note: if you click the mouse at a position which is very close to a previous node's position, then this click will use the previous node.

4. Press 'd' to finish current scene

The annotated results will be saved in `./preprocessing/scannet/graph/` by default. Each `*.pkl` contains a dict data can be used to recover the graph. We provide the pre-annotated scene graph.

## Generate height map for each scene

Execute:

```bash
python scannet_height.py --out_dir PATH_TO_SAVE
```

The height map are used for computing collision.

Each `height/{scan_id}.pkl` contains a floor height map used for computing collision. The format is 

```bash
{
    'dim': int,             # height map dimension
    'height': np.ndarray,   # height map
    'minx': float,          # bounding box minx
    'maxx': float,          # bounding box maxx
    'miny': float,          # bounding box miny
    'maxy': float,          # bounding box maxy
}
```

## Generate path for constructing training data

Execute:

```bash
python scannet_gen_path.py --out_dir PATH_TO_SAVE
```

This part will generate training data, i.e., path in scene, for path planning task.
The generated data are formated as following:

```bash
- PATH_TO_SAVE/
    - path/
        - {scan_id}.pkl
        - ...
```

Each `{scan_id}.pkl` has the following format, storing a path list.
Each element in this list is a tuple consisting a coarse path list and a numpy array.
The coarse path list is a node id list within the original navigation graph. 
The numpy array with shape <N, 3> stores a refined path (position in space) that is processed from the coarse path.
We can only use the refined path for training.

```bash
[
    ([node_id_1, node_id_2, ...], np.ndarray),
    ([node_id_1, node_id_2, ...], np.ndarray),
    ...
]
```

## Train and test split

Train split: scene id less than 600

Test split: scene id greater than 600



