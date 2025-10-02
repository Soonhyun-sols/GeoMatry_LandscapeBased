import torch
from torch import IntTensor, FloatTensor
from torch.nn.functional import pairwise_distance
from .utils import gather_nd
from .baseFF import baseFF
import random
from itertools import combinations
from typing import Tuple
from ..opt_Energy_approximated.calculator import get_ase_atoms

class LennardJonesFF(baseFF):
    '''
    Lennard-Jones force field model between different atom types.
    The force field is defined by:
    E = sum_{i<j} 4*k_{ij} * ((r0_{ij}/r_{ij})**12 - (r0_{ij}/r_{ij})**6)
    where r_{ij} is the distance between atom i and j.
    k_{ij} is the well depth and r0_{ij} is the distance at which the potential is zero.
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
        k_ij = gather_nd(self.k, Zij)
        r0_ij = gather_nd(self.r0, Zij)


        # Calculate Lennard-Jones potential
        rc_ij = r0_ij / 2
        
        mask = rij > rc_ij
        rc_over_r0_inv = r0_ij / rc_ij
        E_rc = 4 * k_ij * (rc_over_r0_inv**12 - rc_over_r0_inv**6)
        E_prime_rc = 4 * k_ij * ((-12 * (r0_ij**12) / (rc_ij**13)) + (6 * (r0_ij**6) / (rc_ij**7)))
        E_double_prime_rc = 4 * k_ij * ((156 * (r0_ij**12) / (rc_ij**14)) - (42 * (r0_ij**6) / (rc_ij**8)))

        # Solve for the cubic polynomial coefficients
        # P(r) = A*r^3 + B*r^2 + D (since P'(0)=0, C=0)
        # Using the boundary conditions:
        # A = (r_c * E''_rc - E'_rc) / (3 * r_c**2)
        A = (rc_ij * E_double_prime_rc - E_prime_rc) / (3 * rc_ij**2)
        # B = 0.5 * E''_rc - 3 * A * r_c
        B = 0.5 * E_double_prime_rc - 3 * A * rc_ij
        # D = E_rc - A * r_c**3 - B * r_c**2
        D = E_rc - A * rc_ij**3 - B * rc_ij**2

        # Calculate polynomial potential
        poly_potential = (A * rij**3 + B * rij**2 + D) * (~mask)

        r0_over_r = r0_ij / rij
        lj_potential = 4 * k_ij * ((r0_over_r ** 12) - (r0_over_r ** 6)) * mask
        # Combine the two potentials
        total_potential = lj_potential + poly_potential


        return torch.sum(total_potential) / 2 # Sum over all pairs and divide by 2 since idx_i and idx_j are symmetric
    def getRdistribution(self,e_size,Ra_stars,Zas,graphbuilders):
        distributions={}
        for Ra, Za, graph_builder in zip(Ra_stars,Zas,graphbuilders):
            idx_i, idx_j=graph_builder(get_ase_atoms(Ra, Za))
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

            for r, zi, zj in zip(rij, Zi, Zj):
                zi=int(zi)
                zj=int(zj)
                if (zi<zj):
                    if (not f'{zi}_{zj}' in distributions.keys()):
                        distributions[f'{zi}_{zj}']=[r]
                    else:
                        distributions[f'{zi}_{zj}'].append(r)
                else:
                    if (not f'{zj}_{zi}' in distributions.keys()):
                        distributions[f'{zj}_{zi}']=[r]
                    else:
                        distributions[f'{zj}_{zi}'].append(r)
        import matplotlib.pyplot as plt
        plt.hist(distributions['1_1'])
        plt.show()
        import numpy as np
        for key in distributions.keys():
            mean=np.mean(distributions[key])
            if (len(distributions[key])==1):
                std=0
            else:
                std=np.std(distributions[key])
            std=(std**2+e_size**2)**0.5
            print(distributions)
            print(mean,std)



def _random_lj_ff_param(max_Za: int, k_max: float=1.0, r0_max: float=1.0) -> Tuple[FloatTensor, FloatTensor]:
    k = torch.rand(max_Za + 1, max_Za + 1, dtype=torch.float64) * k_max
    k = (k + k.T) / 2
    r0 = torch.rand(max_Za + 1, max_Za + 1, dtype=torch.float64) * r0_max
    r0 = (r0 + r0.T) / 2
    return k, r0


def _random_lj_system(N: int, N_pairs: int, max_Za: int, start_Za: int=0) -> Tuple[FloatTensor, IntTensor, IntTensor, IntTensor]:
    Ra = torch.rand(N, 3, dtype=torch.float64)*5
    Ra.requires_grad_()
    Za = torch.randint(start_Za, max_Za + 1, (N,))
    full_combinations = list(combinations(range(N), 2))
    edge_index_undirected = torch.tensor(random.sample(full_combinations, N_pairs))
    idx_i = torch.cat([edge_index_undirected[:, 0], edge_index_undirected[:, 1]])
    idx_j = torch.cat([edge_index_undirected[:, 1], edge_index_undirected[:, 0]])
    return Ra, Za, idx_i, idx_j

def _random_lj_systems(systemN: int, N: int, N_pairs: int, max_Za: int, start_Za: int=0) -> Tuple[FloatTensor, IntTensor, IntTensor, IntTensor]:
    Ras = []
    Zas = []
    idx_is = []
    idx_js = []
    for i in range(systemN):
        Ra, Za, idx_i, idx_j = _random_lj_system(N, N_pairs, max_Za, start_Za)
        Ras.append(Ra)
        Zas.append(Za)
        idx_is.append(idx_i)
        idx_js.append(idx_j)
    return Ras, Zas, idx_is, idx_js