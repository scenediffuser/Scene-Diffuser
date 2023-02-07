CKPT=$1
OPT=$2

if [ -z ${CKPT} ]
then
    echo "No ckpt input."
    exit
fi


if [ -z ${OPT} ] || [ ${OPT} != "OPT" ]
then
    echo "Without optimizer guidance."
    python test.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${CKPT} \
                diffuser=ddpm \
                diffuser.steps=200 \
                model=unet \
                model.use_position_embedding=true \
                task=motion_gen \
                task.dataset.repr_type=absolute \
                task.has_observation=false
else
    echo "With optimizer guidance."
    python test.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${CKPT} \
                diffuser=ddpm \
                diffuser.steps=200 \
                model=unet \
                model.use_position_embedding=true \
                task=motion_gen \
                task.dataset.repr_type=absolute \
                task.has_observation=false \
                optimizer=motion_in_scene \
                optimizer.scale_type=div_var \
                optimizer.scale=1.0 \
                optimizer.vposer=false \
                optimizer.contact_weight=0.02 \
                optimizer.collision_weight=1.0 \
                optimizer.smoothness_weight=0.001 \
                optimizer.frame_interval=1
fi