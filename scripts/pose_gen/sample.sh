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
    python sample.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${CKPT} \
                diffuser=ddpm \
                model=unet \
                task=pose_gen \
                task.visualizer.ksample=10
else
    echo "With optimizer guidance."
    python sample.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${CKPT} \
                diffuser=ddpm \
                model=unet \
                task=pose_gen \
                task.visualizer.ksample=10 \
                optimizer=pose_in_scene \
                optimizer.scale_type=div_var \
                optimizer.scale=2.5 \
                optimizer.vposer=false \
                optimizer.contact_weight=0.02 \
                optimizer.collision_weight=1.0
fi