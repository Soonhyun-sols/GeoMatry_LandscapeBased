import sys
sys.path.append("..")

from torch.testing import assert_close
import torch
import random
from itertools import combinations


def test_spring_ff():
    # model preparation
    from ff.springs import SpringFF
    max_Za = 10
    spring_ff = SpringFF(max_Za=max_Za)
    k = torch.rand(max_Za + 1, max_Za + 1) / 2 # symmetric
    k = k + k.T
    r0 = torch.rand(max_Za + 1, max_Za + 1) / 2 # symmetric
    r0 = r0 + r0.T
    spring_ff.reset_parameters(k=k, r0=r0)

    # data preparation
    N = 10
    Ra = torch.randn(N, 3)
    Ra.requires_grad_()
    Za = torch.randint(0, max_Za + 1, (N,))
    full_combinations = list(combinations(range(N), 2))
    N_pairs = N * (N - 1) // 4
    edge_index_undirected = torch.tensor(random.sample(full_combinations, N_pairs))
    idx_i = torch.cat([edge_index_undirected[:, 0], edge_index_undirected[:, 1]])
    idx_j = torch.cat([edge_index_undirected[:, 1], edge_index_undirected[:, 0]])
    E = spring_ff.get_E(Ra, Za, idx_i, idx_j)
    Fa = spring_ff.get_Fa(E, Ra)
    
    E_ref = 0
    Fa_ref = torch.zeros_like(Ra)
    for l in range(N_pairs * 2):
        i = idx_i[l]
        j = idx_j[l]
        rij = (Ra[i] - Ra[j]).norm()
        kij = k[Za[i], Za[j]]
        r0ij = r0[Za[i], Za[j]]
        E_ref += kij * (rij - r0ij) ** 2 / 2 / 2 # divide by 2 because we have two edges for each pair
        Fa_ref[i] += -kij * (rij - r0ij) * (Ra[i] - Ra[j]) / rij

    assert_close(E, E_ref)
    assert_close(Fa, Fa_ref)
