import os
import hydra
import torch
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from utils.misc import timestamp_str, compute_model_dim
from utils.io import mkdir_if_not_exists
from datasets.base import create_dataset
from datasets.misc import collate_fn_general, collate_fn_squeeze_pcd_batch
from models.base import create_model
from models.visualizer import create_visualizer

def load_ckpt(model: torch.nn.Module, path: str) -> None:
    """ load ckpt for current model

    Args:
        model: current model
        path: save path
    """
    assert os.path.exists(path), 'Can\'t find provided ckpt.'

    saved_state_dict = torch.load(path)['model']
    model_state_dict = model.state_dict()

    for key in model_state_dict:
        if key in saved_state_dict:
            model_state_dict[key] = saved_state_dict[key]
            logger.info(f'Load parameter {key} for current model.')
        ## model is trained with ddm
        if 'module.'+key in saved_state_dict:
            model_state_dict[key] = saved_state_dict['module.'+key]
            logger.info(f'Load parameter {key} for current model [Trained on multi GPUs].')
    
    model.load_state_dict(model_state_dict)

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    ## compute modeling dimension according to task
    cfg.model.d_x = compute_model_dim(cfg.task)
    if os.environ.get('SLURM') is not None:
        cfg.slurm = True # update slurm config
    
    ## set output dir
    eval_dir = os.path.join(cfg.exp_dir, 'eval')
    mkdir_if_not_exists(eval_dir)
    vis_dir = os.path.join(eval_dir, 
        'series' if cfg.task.visualizer.vis_denoising else 'final', timestamp_str())

    logger.add(vis_dir + '/sample.log') # set logger file
    logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg)) # record configuration

    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'
    
    ## prepare dataset for visual evaluation
    ## only load scene
    datasets = {
        'test': create_dataset(cfg.task.dataset, 'test', cfg.slurm, case_only=True),
    }
    for subset, dataset in datasets.items():
        logger.info(f'Load {subset} dataset size: {len(dataset)}')
    
    if cfg.model.scene_model.name == 'PointTransformer':
        collate_fn = collate_fn_squeeze_pcd_batch
    else:
        collate_fn = collate_fn_general
    
    dataloaders = {
        'test': datasets['test'].get_dataloader(
            batch_size=cfg.task.test.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.task.test.num_workers,
            pin_memory=True,
            shuffle=True,
        )
    }
    
    ## create model and load ckpt
    model = create_model(cfg, slurm=cfg.slurm, device=device)
    model.to(device=device)
    ## if your models are seperately saved in each epoch, you need to change the model path manually
    load_ckpt(model, path=os.path.join(cfg.ckpt_dir, 'model.pth'))
    
    ## create visualizer and visualize
    visualizer = create_visualizer(cfg.task.visualizer)
    visualizer.visualize(model, dataloaders['test'], vis_dir)

if __name__ == '__main__':
    ## set random seed
    seed = 0
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()