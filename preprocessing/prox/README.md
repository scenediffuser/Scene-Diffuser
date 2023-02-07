# Preprocess PROX data for pose and motion generation in indoor scenes

## Data preparation

1. Download original [PROX](https://prox.is.tue.mpg.de/) data, contaning scene meshes, camera data, and so on. And they are organized as follows:

    ```bash
    -| $YOUR_PATH/PROX/
    ---| body_segments/
    ---| cam2world/
    ---| scenes/
    ---| sdf/
    ---| preprocess_scenes/
    ```

    - `preprocess_scenes/` are pre-processed scenes that are used for models. You can generate these with the following instruction or just use our [pre-processed results](https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing).
    - Other folders are directly downloaded from [PROX](https://prox.is.tue.mpg.de/).
    - The path of PROX is specified by `dataset.prox_dir` in `./configs/task/pose_gen.yaml` or `./configs/task/motion_gen.yaml`.

2. Download [LEMO](https://sanweiliti.github.io/LEMO/LEMO.html) data, which will be used as ground truth of the pose and motion.
    - The path of LEMO is specified by `dataset.data_dir` in `./configs/task/pose_gen.yaml` or `./configs/task/motion_gen.yaml`.

3. Download [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php) weights.
    - The path of SMPL-X is specified by `dataset.smpl_dir` in `./configs/task/pose_gen.yaml` or `./configs/task/motion_gen.yaml`.

4. Download [VPoser](https://smpl-x.is.tue.mpg.de/download.php) weights. (Optional)
    - VPoser is optional for optimization guidance. We don't use this in our default setting.
    - The path of VPoser is specified by `dataset.vposer_dir` in `./configs/task/pose_gen.yaml` or `./configs/task/motion_gen.yaml`.

## Preprocess the scene to point cloud

1. Change the folder path configuration in `preprocessing/prox/prox_scene.py`, i.e., `scene_dir` of original PROX data folder and `preprocess_scenes_dir` used to save processed scene point cloud.

    Notes: you also need to change the data path in configuration, i.e., `*.yaml` files, for training and evaluation.

2. Execute to process prox scenes:

    ```bash
    cd preprocessing/prox
    python prox_scene.py
    ```