import sys
sys.path.append("..")
import torch
from geomatry.ff.springs import SpringFF


def test_spring_ff():
    # model preparation
    from config import get_spring_ff_param, get_spring_system
    from torch.testing import assert_close
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
    Haa = spring_ff.get_Haa(Fa, Ra)
    E_ref = 0
    Fa_ref = torch.zeros((N, 3))
    Haa_ref = torch.zeros((N * 3, N * 3))
    for l in range(N_pairs * 2):
        i = idx_i[l]
        j = idx_j[l]
        Rij = Ra[i] - Ra[j]
        rij = Rij.norm()
        kij = k[Za[i], Za[j]]
        r0ij = r0[Za[i], Za[j]]
        E_ref += kij * (rij - r0ij) ** 2 / 2 / 2 # divide by 2 because we have two edges for each pair
        Fa_ref[i] += -kij * (rij - r0ij) * Rij / rij
        Haa_ref[i * 3: (i + 1) * 3, i * 3: (i + 1) * 3] += kij * (
            (1 - r0ij / rij) * torch.eye(3) + r0ij * torch.outer(Rij, Rij) / rij ** 3
        )
        Haa_ref[i * 3: (i + 1) * 3, j * 3: (j + 1) * 3] += kij * (
            -(1 - r0ij / rij) * torch.eye(3) - r0ij * torch.outer(Rij, Rij) / rij ** 3
        )
    
    assert_close(E, E_ref)
    assert_close(Fa, Fa_ref)
    assert_close(Haa, Haa_ref)


def test_optimization():
    from ase.optimize import BFGS
    from ase import Atoms
    from geomatry.opt.calculator import NNFFCalculator
    from geomatry.ff.graph import get_given_graph_builder
    from random import random
    import numpy as np
    model = SpringFF(max_Za=2)
    model.reset_parameters(
        k=torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), 
        r0=torch.tensor([[2.0, 2.0, 2.0], [2.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    )
    atoms = Atoms(numbers=[1, 2, 2], positions=[[0.0, 0.0, 0.0], [1 + random() * 2, 0.0, 0.0], [random() * 2, 1 + random() * 2, 0.0]])
    idx_i = torch.tensor([0, 0, 1, 1, 2, 2])
    idx_j = torch.tensor([1, 2, 0, 2, 0, 1])
    atoms.calc = NNFFCalculator(model=model, graph_builder=get_given_graph_builder(idx_i, idx_j))
    optimizer = BFGS(atoms)
    optimizer.run(fmax=1e-6)
    dist_ab = np.linalg.norm(atoms.positions[0] - atoms.positions[1])
    dist_ac = np.linalg.norm(atoms.positions[0] - atoms.positions[2])
    dist_bc = np.linalg.norm(atoms.positions[1] - atoms.positions[2])
    assert np.allclose(dist_ab, 1.0)
    assert np.allclose(dist_ac, 1.0)
    assert np.allclose(dist_bc, 1.0)
