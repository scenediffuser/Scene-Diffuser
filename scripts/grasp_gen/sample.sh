CKPT=$1

python sample.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${CKPT} \
                diffuser=ddpm_l1 \
                model=unet_grasp \
                task=grasp_gen \
                task.visualizer.ksample=10
