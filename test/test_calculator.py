import sys
sys.path.append("..")
from geomatry.opt.calculator import NNFFCalculator
from geomatry.ff.springs import SpringFF
from geomatry.ff.graph import get_given_graph_builder
from config import get_spring_system, get_spring_ff_param
from ase import Atoms
import numpy as np


def test_calculator():
    N = 10
    N_pairs = N * (N - 1) // 4
    max_Za = 10
    model = SpringFF(max_Za=max_Za)
    k, r0 = get_spring_ff_param(max_Za)
    model.reset_parameters(k=k, r0=r0)
    Ra, Za, idx_i, idx_j = get_spring_system(N, N_pairs, max_Za, start_Za=1)
    atoms = Atoms(numbers=Za.detach().numpy(), positions=Ra.detach().numpy())
    calculator = NNFFCalculator(model=model, graph_builder=get_given_graph_builder(idx_i, idx_j))
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    E = model.get_E(Ra, Za, idx_i, idx_j)
    Fa = model.get_Fa(E, Ra)
    assert np.allclose(energy, E.detach().numpy())
    assert np.allclose(forces, -Fa.detach().numpy())