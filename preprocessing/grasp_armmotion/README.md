# Prepare data&assets for grasp generation and arm motion planning

1. Download the [ignore-file.zip](https://drive.google.com/drive/folders/1Kg2Gt7nQ326yYEBiQcS2DW2ZYhdHp3uk?usp=sharing) and upzip the file.

2. Put the contents in `ignore-file/assets/` into `${YOUR_PATH}/Scene-Diffuser/assets/` and put the contents in `ignore-file/envs/assets/` into `${YOUR_PATH}/Scene-Diffuser/envs/assets/`.

3. Download the [MultiDex_UR](https://drive.google.com/drive/folders/1Kg2Gt7nQ326yYEBiQcS2DW2ZYhdHp3uk?usp=sharing) dataset, then specify `asset_dir` `configs/task/grasp_gen_ur.yaml`. You can also change them in the bash files, e.g., `scripts/grasp_gen_ur/train.sh`.

4. Download the [FK2PlanDataset](https://drive.google.com/drive/folders/1Kg2Gt7nQ326yYEBiQcS2DW2ZYhdHp3uk?usp=sharing) dataset, then specify `data_dir` in `configs/task/franka_planning.yaml`. You can also change them in the bash files, e.g., `scripts/franka_planning/train.sh`.
