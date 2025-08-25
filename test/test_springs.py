import sys
sys.path.append("..")
import torch
from geomatry.ff.springs import SpringFF
from geomatry.ff.springs import _random_spring_ff_param, _random_spring_system


def test_spring_ff():
    # model preparation
    from torch.testing import assert_close
    max_Za = 10
    spring_ff = SpringFF(max_Za=max_Za)
    k, r0 = _random_spring_ff_param(max_Za)
    spring_ff.reset_parameters(k=k, r0=r0)

    # data preparation
    N = 10
    N_pairs = N * (N - 1) // 4
    Ra, Za, idx_i, idx_j = _random_spring_system(N, N_pairs, max_Za)
    E = spring_ff.get_E(Ra, Za, idx_i, idx_j)
    Fa = spring_ff.get_Fa(E, Ra)
    Hessian = spring_ff.get_Hessian(Fa, Ra)
    E_ref = 0
    Fa_ref = torch.zeros((N, 3), dtype=torch.float64)
    Hessian_ref = torch.zeros((N * 3, N * 3), dtype=torch.float64)
    for l in range(N_pairs * 2):
        i = idx_i[l]
        j = idx_j[l]
        Rij = Ra[i] - Ra[j]
        rij = Rij.norm()
        kij = k[Za[i], Za[j]]
        r0ij = r0[Za[i], Za[j]]
        E_ref += kij * (rij - r0ij) ** 2 / 2 / 2 # divide by 2 because we have two edges for each pair
        Fa_ref[i] += -kij * (rij - r0ij) * Rij / rij
        Hessian_ref[i * 3: (i + 1) * 3, i * 3: (i + 1) * 3] += kij * (
            (1 - r0ij / rij) * torch.eye(3) + r0ij * torch.outer(Rij, Rij) / rij ** 3
        )
        Hessian_ref[i * 3: (i + 1) * 3, j * 3: (j + 1) * 3] += kij * (
            -(1 - r0ij / rij) * torch.eye(3) - r0ij * torch.outer(Rij, Rij) / rij ** 3
        )
    
    assert_close(E, E_ref)
    assert_close(Fa, Fa_ref)
    assert_close(Hessian, Hessian_ref)



