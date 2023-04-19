EXP_NAME=$1

python train.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_name=${EXP_NAME} \
                diffuser=ddpm \
                diffuser.loss_type=l1 \
                diffuser.steps=100 \
                model=unet_grasp \
                task=grasp_gen_ur \
                task.dataset.normalize_x=true \
                task.dataset.normalize_x_trans=true