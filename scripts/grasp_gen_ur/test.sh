EVAL_DIR=$1
DATASET_DIR=$2

# eval_dir: path to the directory where the evaluation results be sampled 
# (e.g., "outputs/2022-11-15_18-07-50_GPUR_l1_pn2_T100/eval/final/2023-04-20_13-06-44")
# dataset_dir: path to the directory where the MultiDex_UR dataset is stored
# (e.g., "/home/puhao/data/MultiDex_UR")
python ./scripts/grasp_gen_ur/test.py --eval_dir=${EVAL_DIR} \
                                      --dataset_dir=${DATASET_DIR} \
                                      --stability_config='envs/tasks/grasp_test_force.yaml' \
                                      --seed=42
