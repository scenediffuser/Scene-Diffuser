EXP_NAME=$1

python train.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_name=${EXP_NAME} \
                diffuser=ddpm \
                model=unet \
                model.use_position_embedding=true \
                task=path_planning \
                task.dataset.repr_type=relative