import torch
from torch import FloatTensor

def rmsd_loss(pred: FloatTensor, target: FloatTensor) -> FloatTensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))