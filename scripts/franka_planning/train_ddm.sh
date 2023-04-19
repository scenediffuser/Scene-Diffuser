EXP_NAME=$1

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --use_env train_ddm.py \
                hydra/job_logging=none hydra/hydra_logging=none \
                exp_name=${EXP_NAME} \
                diffuser=ddpm \
                diffuser.loss_type=l1 \
                diffuser.steps=30 \
                model=unet_fk2 \
                model.use_position_embedding=true \
                task=franka_planning \
                task.dataset.normalize_x=true