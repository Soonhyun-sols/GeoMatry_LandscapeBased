import sys
sys.path.append("..")
import torch
from geomatry.ff.springs import SpringFF


def test_BFGS():
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
    atoms.calc = NNFFCalculator(ff=model, graph_builder=get_given_graph_builder(idx_i, idx_j))
    optimizer = BFGS(atoms)
    optimizer.run(fmax=1e-6)
    dist_ab = np.linalg.norm(atoms.positions[0] - atoms.positions[1])
    dist_ac = np.linalg.norm(atoms.positions[0] - atoms.positions[2])
    dist_bc = np.linalg.norm(atoms.positions[1] - atoms.positions[2])
    assert np.allclose(dist_ab, 1.0)
    assert np.allclose(dist_ac, 1.0)
    assert np.allclose(dist_bc, 1.0)