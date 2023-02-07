from typing import Dict, List
import torch
from einops import rearrange

def collate_fn_general(batch: List) -> Dict:
    """ General collate function used for dataloader.
    """
    batch_data = {key: [d[key] for d in batch] for key in batch[0]}
    
    for key in batch_data:
        if torch.is_tensor(batch_data[key][0]):
            batch_data[key] = torch.stack(batch_data[key])
    return batch_data

def collate_fn_squeeze_pcd_batch(batch: List) -> Dict:
    """ General collate function used for dataloader.
    This collate function is used for point-transformer
    """
    batch_data = {key: [d[key] for d in batch] for key in batch[0]}
    
    for key in batch_data:
        if torch.is_tensor(batch_data[key][0]):
            batch_data[key] = torch.stack(batch_data[key])
    
    ## squeeze the first dimension of pos and feat
    offset, count = [], 0
    for item in batch_data['pos']:
        count += item.shape[0]
        offset.append(count)
    offset = torch.IntTensor(offset)
    batch_data['offset'] = offset

    batch_data['pos'] = rearrange(batch_data['pos'], 'b n c -> (b n) c')
    batch_data['feat'] = rearrange(batch_data['feat'], 'b n c -> (b n) c')
    
    return batch_data


def collate_fn_squeeze_pcd_batch_grasp(batch: List) -> Dict:
    """ General collate function used for dataloader.
    This collate function is used for point-transformer
    """
    batch_data = {key: [d[key] for d in batch] for key in batch[0]}

    for key in batch_data:
        if torch.is_tensor(batch_data[key][0]):
            batch_data[key] = torch.stack(batch_data[key])

    ## squeeze the first dimension of pos and feat
    offset, count = [], 0
    for item in batch_data['pos']:
        count += item.shape[0]
        offset.append(count)
    offset = torch.IntTensor(offset)
    batch_data['offset'] = offset

    batch_data['pos'] = rearrange(batch_data['pos'], 'b n c -> (b n) c')

    return batch_data
