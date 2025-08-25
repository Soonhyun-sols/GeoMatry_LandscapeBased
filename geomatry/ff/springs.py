import torch
from torch import IntTensor, FloatTensor
from torch.nn.functional import pairwise_distance
from .utils import gather_nd
from .baseFF import baseFF
import random
from itertools import combinations
from typing import Tuple    

class SpringFF(baseFF):
    '''
    Spring force field model between different atom types.
    The force field is defined by:
    E = 1/2 * sum_{i,j} k_{ij} * (r_{ij} - r0_{ij}) ** 2
    where r_{ij} is the distance between atom i and j, and r0_{ij} is the equilibrium distance between atom i and j.
    k_{ij} is the force constant between atom i and j.
    The force field is symmetric, i.e. k_{ij} = k_{ji} and r0_{ij} = r0_{ji}.
    The force field is defined for all atom types up to max_Za.
    '''
    def __init__(self, max_Za: int):
        super().__init__(max_Za)
        self.register_parameter("k", torch.nn.Parameter(torch.ones(max_Za + 1, max_Za + 1)))
        self.register_parameter("r0", torch.nn.Parameter(torch.ones(max_Za + 1, max_Za + 1)))

    def reset_parameters(self, k: FloatTensor, r0: FloatTensor, symmetrize: bool=True) -> None:
        assert k.shape == (self.max_Za + 1, self.max_Za + 1)
        assert r0.shape == (self.max_Za + 1, self.max_Za + 1)
        if symmetrize:
            k = (k + k.T) / 2
            r0 = (r0 + r0.T) / 2
        else:
            assert torch.allclose(k, k.T)
            assert torch.allclose(r0, r0.T)
        self.k.data = k
        self.r0.data = r0

    def get_E(self, Ra: FloatTensor, Za: IntTensor, idx_i: IntTensor, idx_j: IntTensor) -> FloatTensor:
        if Ra.device == torch.device("cpu"):
            Ri = Ra[idx_i]
            Rj = Ra[idx_j]
            Zi = Za[idx_i]
            Zj = Za[idx_j]
        else:
            Ri = Ra.gather(0, idx_i.view(-1, 1).expand(-1, 3))
            Rj = Ra.gather(0, idx_j.view(-1, 1).expand(-1, 3))
            Zi = Za.gather(0, idx_i)
            Zj = Za.gather(0, idx_j)
        rij = pairwise_distance(Ri, Rj)
        Zij = torch.stack([Zi, Zj], dim=1)
        kij = gather_nd(self.k, Zij)
        r0ij = gather_nd(self.r0, Zij)
        return torch.sum(kij * (rij - r0ij) ** 2 / 2) / 2 # divide by 2 because we have two edges for each pair


def _random_spring_ff_param(max_Za: int, k_max: float=1.0, r0_max: float=1.0) -> Tuple[FloatTensor, FloatTensor]:
    k = torch.rand(max_Za + 1, max_Za + 1, dtype=torch.float64) * k_max # symmetric
    k = (k + k.T) / 2
    r0 = torch.rand(max_Za + 1, max_Za + 1, dtype=torch.float64) * r0_max # symmetric
    r0 = (r0 + r0.T) / 2
    return k, r0


def _random_spring_system(N: int, N_pairs: int, max_Za: int, start_Za: int=0) -> Tuple[FloatTensor, IntTensor, IntTensor, IntTensor]:
    Ra = torch.randn(N, 3, dtype=torch.float64)
    Ra.requires_grad_()
    Za = torch.randint(start_Za, max_Za + 1, (N,))
    full_combinations = list(combinations(range(N), 2))
    edge_index_undirected = torch.tensor(random.sample(full_combinations, N_pairs))
    idx_i = torch.cat([edge_index_undirected[:, 0], edge_index_undirected[:, 1]])
    idx_j = torch.cat([edge_index_undirected[:, 1], edge_index_undirected[:, 0]])
    return Ra, Za, idx_i, idx_j
