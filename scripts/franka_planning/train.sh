EXP_NAME=$1

python train.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_name=${EXP_NAME} \
                diffuser=ddpm \
                diffuser.loss_type=l1 \
                diffuser.steps=30 \
                model=unet_fk2 \
                model.use_position_embedding=true \
                task=franka_planning \
                task.dataset.normalize_x=true