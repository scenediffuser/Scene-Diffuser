#!/bin/bash
#SBATCH -J gpg_diffuser
#SBATCH --comment "3D-conditioned diffusion"

#SBATCH --qos singlegpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00

#SBATCH -o "/home/lipuhao/dev/SceneDiffuser/3d_diffusion/outputs/slurm/%x.%j.out"
#SBATCH -e "/home/lipuhao/dev/SceneDiffuser/3d_diffusion/outputs/slurm/%x.%j.err"

EXP_NAME=$1
BATCH_SIZE=$2
LR=$3

SLURM=1 python train.py hydra/job_logging=none hydra/hydra_logging=none \
                output_dir=outputs/exps/ \
                exp_name=${EXP_NAME} \
                diffuser=ddpm_l1 \
                model=unet_grasp \
                task=grasp_gen \
                task.visualizer.visualize=false \
                task.train.batch_size=${BATCH_SIZE} \
                task.lr=${LR}