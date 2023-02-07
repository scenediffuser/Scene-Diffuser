CKPT=$1

python plan.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${CKPT} \
                diffuser=ddpm \
                model=unet \
                model.use_position_embedding=true \
                task=path_planning \
                task.dataset.repr_type=relative \
                task.env.inpainting_horizon=16 \
                task.env.eval_case_num=64 \
                task.env.robot_top=3.0 \
                task.env.env_adaption=false \
                optimizer=path_in_scene \
                optimizer.scale_type=div_var \
                optimizer.continuity=false \
                planner=greedy_path_planning \
                planner.scale=0.2 \
                planner.scale_type=div_var \
                planner.greedy_type=all_frame_exp