CKPT=$1
GUIDANCE=($2 $3)

if [ -z ${CKPT} ]
then
    echo "No ckpt input."
    exit
fi

if [[ "${GUIDANCE[@]}" =~ "OPT" ]]
then
    echo "With optimizer guidance."
    OPT_ARGS="optimizer=path_in_scene optimizer.scale_type=div_var optimizer.continuity=false"
else
    echo "Without optimizer guidance."
    OPT_ARGS=""
fi

if [[ "${GUIDANCE[@]}" =~ "PLA" ]]
then
    echo "With planner guidance."
    PLA_ARGS="planner=greedy_path_planning planner.scale=0.2 planner.scale_type=div_var planner.greedy_type=all_frame_exp"
else
    echo "Without planner guidance."
    PLA_ARGS=""
fi

python sample.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${CKPT} \
                diffuser=ddpm \
                model=unet \
                model.use_position_embedding=true \
                task=path_planning \
                task.dataset.repr_type=relative \
                task.visualizer.ksample=10 ${OPT_ARGS} ${PLA_ARGS}