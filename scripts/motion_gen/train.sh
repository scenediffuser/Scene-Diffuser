EXP_NAME=$1

python train.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_name=${EXP_NAME} \
                diffuser=ddpm \
                diffuser.steps=200 \
                model=unet \
                model.use_position_embedding=true \
                task=motion_gen \
                task.dataset.repr_type=absolute \
                task.has_observation=false