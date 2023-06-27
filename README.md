# Diffusion-based Generation, Optimization, and Planning in 3D Scenes

<p align="left">
    <a href='https://scenediffuser.github.io/paper.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://arxiv.org/abs/2301.06015'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://scenediffuser.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://huggingface.co/spaces/SceneDiffuser/SceneDiffuserDemo'>
      <img src='https://img.shields.io/badge/Demo-HuggingFace-yellow?style=plastic&logo=AirPlay%20Video&logoColor=yellow' alt='HuggingFace'>
    </a>
    <a href='https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing'>
      <img src='https://img.shields.io/badge/Model-Checkpoints-orange?style=plastic&logo=Google%20Drive&logoColor=orange' alt='Checkpoints'>
    </a>
</p>

[Siyuan Huang*](https://siyuanhuang.com/),
[Zan Wang*](https://silvester.wang),
[Puhao Li](https://xiaoyao-li.github.io/),
[Baoxiong Jia](https://buzz-beater.github.io/),
[Tengyu Liu](http://tengyu.ai/),
[Yixin Zhu](https://yzhu.io/),
[Wei Liang](https://liangwei-bit.github.io/web/),
[Song-Chun Zhu](http://www.stat.ucla.edu/~sczhu/)

This repository is the official implementation of paper "Diffusion-based Generation, Optimization, and Planning in 3D Scenes".

We introduce SceneDiffuser, a conditional generative model for 3D scene understanding. SceneDiffuser provides a unified model for solving scene-conditioned generation, optimization, and planning. In contrast to prior work, SceneDiffuser is intrinsically scene-aware, physics-based, and goal-oriented.

[Paper](https://scenediffuser.github.io/paper.pdf) |
[arXiv](https://arxiv.org/abs/2301.06015) |
[Project](https://scenediffuser.github.io/) |
[HuggingFace Demo](https://huggingface.co/spaces/SceneDiffuser/SceneDiffuserDemo) |
[Checkpoints](https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing)

<div align=center>
<img src='./figures/teaser.png' width=60%>
</div>

## Abstract

We introduce SceneDiffuser, a conditional generative model for 3D scene understanding. SceneDiffuser provides a unified model for solving scene-conditioned generation, optimization, and planning. In contrast to prior works, SceneDiffuser is intrinsically scene-aware, physics-based, and goal-oriented. With an iterative sampling strategy, SceneDiffuser jointly formulates the scene-aware generation, physics-based optimization, and goal-oriented planning via a diffusion-based denoising process in a fully differentiable fashion. Such a design alleviates the discrepancies among different modules and the posterior collapse of previous scene-conditioned generative models. We evaluate SceneDiffuser with various 3D scene understanding tasks, including human pose and motion generation, dexterous grasp generation, path planning for 3D navigation, and motion planning for robot arms. The results show significant improvements compared with previous models, demonstrating the tremendous potential of SceneDiffuser for the broad community of 3D scene understanding.

## News

- [ 2023.04 ] We release the code for grasp generation and arm motion planning!

## Setup

1. Create a new `conda` environemnt and activate it.

    ```bash
    conda create -n 3d python=3.8
    conda activate 3d
    ```

2. Install dependent libraries with `pip`.

    ```bash
    pip install -r pre-requirements.txt
    pip install -r requirements.txt
    ```

    - We use `pytorch1.11` and `cuda11.3`, modify `pre-requirements.txt` to install [other versions](https://pytorch.org/get-started/previous-versions/) of `pytorch`.

3. Install [Isaac Gym](https://developer.nvidia.com/isaac-gym) and install [pointnet2](https://github.com/daveredrum/Pointnet2.ScanNet) by executing the following command (optional for grasp generation and arm motion planning).

    ```bash
    pip install git+https://github.com/daveredrum/Pointnet2.ScanNet.git#subdirectory=pointnet2
    ```

## Data & Checkpoints

### 1. Data

You can use our [pre-processed data](https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing) or process the data by yourself following the [instructions](./preprocessing/README.md).

But, you also need to download some official released data assets which are not processed, see [instructions](./preprocessing/README.md). Please remember to use your own data path by modifying the path configuration in:

- `scene_model.pretrained_weights` in `model/*.yaml` for the path of pre-trained scene encoder (if you use a pre-trained scene encoder)

- `dataset.*_dir`/`dataset.*_path` configurations in `task/*.yaml` for the path of data assets

### 2. Checkpoints

Download our [pre-trained model](https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing) and unzip them into a folder, e.g., `./outputs/`.

task|checkpoints|desc
-|-|-
Pretrained Point Transformer|2022-04-13_18-29-56_POINTTRANS_C_32768|
Pose Generation|2022-11-09_11-22-52_PoseGen_ddm4_lr1e-4_ep100|
Motion Generation|2022-11-09_12-54-50_MotionGen_ddm_T200_lr1e-4_ep300|w/o start position
Motion Generation|2022-11-09_14-28-12_MotionGen_ddm_T200_lr1e-4_ep300_obser|w/ start position
Path Planning|2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL|
Grasp Generation|2022-11-15_18-07-50_GPUR_l1_pn2_T100|
Arm Motion Planning|2022-11-11_14-28-30_FK2Plan_ptr_T30_4|denoising step is 30


## Task-1: Human Pose Generation in 3D Scenes

### Train

- Train with single gpu

    ```bash
    bash scripts/pose_gen/train.sh ${EXP_NAME}
    ```

- Train with 4 GPUs (modify `scripts/pose_gen/train_ddm.sh` to specify the visible GPUs)

    ```bash
    bash scripts/pose_gen/train_ddm.sh ${EXP_NAME}
    ```

### Test (Quantitative Evaluation)

```bash
bash scripts/pose_gen/test.sh ${CKPT} [OPT]
# e.g., bash scripts/pose_gen/test.sh ./outputs/2022-11-09_11-22-52_PoseGen_ddm4_lr1e-4_ep100/ OPT
```

- `[OPT]` is optional for optimization-guided sampling.

### Sample (Qualitative Visualization)

```bash
bash scripts/pose_gen/sample.sh ${CKPT} [OPT]
# e.g., bash scripts/pose_gen/sample.sh ./outputs/2022-11-09_11-22-52_PoseGen_ddm4_lr1e-4_ep100/ OPT
```

- `[OPT]` is optional for optimization-guided sampling.

## Task-2: Human Motion Generation in 3D Scenes

**The default configuration is motion generation without observation. If you want to explore the setting of motion generation with start observation, please change the `task.has_observation` to `true` in all the scripts in folder `./scripts/motion_gen/`.**

### Train

- Train with single gpu

    ```bash
    bash scripts/motion_gen/train.sh ${EXP_NAME}
    ```

- Train with 4 GPUs (modify `scripts/motion_gen/train_ddm.sh` to specify the visible GPUs)

    ```bash
    bash scripts/motion_gen/train_ddm.sh ${EXP_NAME}
    ```

### Test (Quantitative Evaluation)

```bash
bash scripts/motion_gen/test.sh ${CKPT} [OPT]
# e.g., bash scripts/motion_gen/test.sh ./outputs/2022-11-09_12-54-50_MotionGen_ddm_T200_lr1e-4_ep300/ OPT
```

- `[OPT]` is optional for optimization-guided sampling.

### Sample (Qualitative Visualization)

```bash
bash scripts/motion_gen/sample.sh ${CKPT} [OPT]
# e.g., bash scripts/motion_gen/sample.sh ./outputs/2022-11-09_12-54-50_MotionGen_ddm_T200_lr1e-4_ep300/ OPT
```

- `[OPT]` is optional for optimization-guided sampling.

## Task-3: Dexterous Grasp Generation for 3D Objects

To run this code, you first need to change the git branch to `obj` by executing

```bash
git checkout obj
```

Make sure you have installed [Isaac Gym](https://developer.nvidia.com/isaac-gym) and [pointnet2](https://github.com/daveredrum/Pointnet2.ScanNet). See [Setup](#setup) section.

### Train

- Train with single gpu (one gpu is enough)

    ```bash
    bash scripts/grasp_gen_ur/train.sh ${EXP_NAME}
    ```

### Sample (Qualitative Visualization)

```bash
bash scripts/grasp_gen_ur/sample.sh ${CKPT} [OPT]
# e.g., bash scripts/grasp_gen_ur/sample.sh ./outputs/2022-11-15_18-07-50_GPUR_l1_pn2_T100/ OPT
```

- `[OPT]` is optional for optimization-guided sampling.

### Test (Quantitative Evaluation)

You first need to run `scripts/grasp_gen_ur/sample.sh` to sample some results. Then we will compute quantitative metrics with these sampled results.

```bash
bash scripts/grasp_gen_ur/test.sh ${EVAL_DIR} ${DATASET_DIR}
# e.g., bash scripts/grasp_gen_ur/test.sh outputs/2022-11-15_18-07-50_GPUR_l1_pn2_T100/eval/final/2023-04-20_13-06-44 YOUR_PATH/data/MultiDex_UR
```

## Task-4: Path Planning in 3D Scenes

### Train

- Train with single gpu

    ```bash
    bash scripts/path_planning/train.sh ${EXP_NAME}
    ```

- Train with 4 GPUs (modify `scripts/path_planning/train_ddm.sh` to specify the visible GPUs)

    ```bash
    bash scripts/path_planning/train_ddm.sh ${EXP_NAME}
    ```

### Test (Quantitative Evaluation)

```bash
bash scripts/path_planning/plan.sh ${CKPT}
# e.g., bash scripts/path_planning/plan.sh ./outputs/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/
```

### Sample (Qualitative Visualization)

```bash
bash scripts/path_planning/sample.sh ${CKPT} [OPT] [PLA]
# e.g., bash scripts/path_planning/sample.sh ./outputs/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/ OPT PLA
```

- The program will generate trajectories with given start position and scene; rendering the results into images. (The results not the planning results, just use diffuser to generate diverse trajectories.)
- `[OPT]` is optional for optimization-guided sampling.
- `[PLA]` is optional for planner-guided sampling.

## Task-5: Motion Planning for Robot Arms

To run this code, you first need to change the git branch to `obj` by executing

```bash
git checkout obj
```

Make sure you have installed [Isaac Gym](https://developer.nvidia.com/isaac-gym) and [pointnet2](https://github.com/daveredrum/Pointnet2.ScanNet). See [Setup](#setup) section.

### Train

- Train with single gpu

    ```bash
    bash scripts/franka_planning/train.sh ${EXP_NAME}
    ```

- Train with 4 GPUs (modify `scripts/path_planning/train_ddm.sh` to specify the visible GPUs)

    ```bash
    bash scripts/franka_planning/train_ddm.sh ${EXP_NAME}
    ```

### Test (Quantitative Evaluation)

```bash
bash scripts/franka_planning/plan.sh ${CKPT}
# e.g., bash scripts/franka_planning/plan.sh outputs/2022-11-11_14-28-30_FK2Plan_ptr_T30_4/
```

## Citation

If you find our project useful, please consider citing us:

```tex
@article{huang2023diffusion,
  title={Diffusion-based Generation, Optimization, and Planning in 3D Scenes},
  author={Huang, Siyuan and Wang, Zan and Li, Puhao and Jia, Baoxiong and Liu, Tengyu and Zhu, Yixin and Liang, Wei and Zhu, Song-Chun},
  journal={arXiv preprint arXiv:2301.06015},
  year={2023}
}
```

## Acknowledgments

Some codes are borrowed from [latent-diffusion](https://github.com/CompVis/latent-diffusion), [PSI-release](https://github.com/yz-cnsdqz/PSI-release), [Pointnet2.ScanNet](https://github.com/daveredrum/Pointnet2.ScanNet), [point-transformer](https://github.com/POSTECH-CVLab/point-transformer), and [diffuser](https://github.com/jannerm/diffuser).

### License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details. The following datasets are used in this project and are subject to their respective licenses:

- PROX is under the [Software Copyright License for non-commercial scientific research purposes](https://prox.is.tue.mpg.de/license.html).
- LEMO is under the [MIT License](https://github.com/sanweiliti/LEMO/blob/main/LICENSE).
- ScanNet V2 is under the [ScanNet Terms of Use](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf).
