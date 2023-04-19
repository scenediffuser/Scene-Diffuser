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
                diffuser.loss_type=l1 \
                diffuser.steps=100 \
                model=unet_grasp \
                task=grasp_gen_ur \
                task.dataset.normalize_x=true \
                task.dataset.normalize_x_trans=true
else
    echo "With optimizer guidance."
    python sample.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${CKPT} \
                diffuser=ddpm \
                diffuser.loss_type=l1 \
                diffuser.steps=100 \
                model=unet_grasp \
                task=grasp_gen_ur \
                task.dataset.normalize_x=true \
                task.dataset.normalize_x_trans=true \
                optimizer=grasp_with_object \
                optimizer.scale=0.1
fi