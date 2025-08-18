import torch
import random
from itertools import combinations
from torch import FloatTensor, IntTensor
from typing import Tuple


def get_spring_ff_param(max_Za: int):
    k = torch.rand(max_Za + 1, max_Za + 1) / 2 # symmetric
    k = (k + k.T) / 2
    r0 = torch.rand(max_Za + 1, max_Za + 1) / 2 # symmetric
    r0 = (r0 + r0.T) / 2
    return k, r0

def get_spring_system(N: int, N_pairs: int, max_Za: int, start_Za: int=0) -> Tuple[FloatTensor, IntTensor, IntTensor, IntTensor]:
    Ra = torch.randn(N, 3)
    Ra.requires_grad_()
    Za = torch.randint(start_Za, max_Za + 1, (N,))
    full_combinations = list(combinations(range(N), 2))
    N_pairs = N * (N - 1) // 4
    edge_index_undirected = torch.tensor(random.sample(full_combinations, N_pairs))
    idx_i = torch.cat([edge_index_undirected[:, 0], edge_index_undirected[:, 1]])
    idx_j = torch.cat([edge_index_undirected[:, 1], edge_index_undirected[:, 0]])
    return Ra, Za, idx_i, idx_j