import sys
sys.path.append("..")
from torch.testing import assert_close
import torch


def test_spring_ff():
    # model preparation
    from geomatry.ff.springs import SpringFF
    from config import get_spring_ff_param, get_spring_system
    max_Za = 10
    spring_ff = SpringFF(max_Za=max_Za)
    k, r0 = get_spring_ff_param(max_Za)
    spring_ff.reset_parameters(k=k, r0=r0)

    # data preparation
    N = 10
    N_pairs = N * (N - 1) // 4
    Ra, Za, idx_i, idx_j = get_spring_system(N, N_pairs, max_Za)
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
