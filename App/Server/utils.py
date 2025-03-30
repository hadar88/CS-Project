import torch

def merge_ids_and_amounts(ids, amounts):
    return torch.stack((ids, amounts), dim=-1)
