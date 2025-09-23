import torch
import torch.nn as nn
from torch import IntTensor, FloatTensor
from torch.nn.functional import pairwise_distance
from .utils import gather_nd
from .baseFF import baseFF
from itertools import combinations
from typing import Tuple, List, Dict
import random
from ..opt_Energy_approximated.calculator import get_ase_atoms


class PairNN(nn.Module):
    """Small feedforward network: r -> energy"""
    def __init__(self, hidden_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim, dtype=torch.float32),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1, dtype=torch.float32),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')


    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r.unsqueeze(-1)).squeeze(-1)


class NNPotentialFF(baseFF):
    """
    Neural-network-based force field model between different atom types.
    Each atom-type pair (i,j) has its own small neural network taking r_ij -> energy.
    """
    def __init__(self, max_Za: int, hidden_dim: int = 10):
        super().__init__(max_Za)
        self.max_Za = max_Za
        # make a dictionary of networks for each (i,j) combination (symmetric)
        self.nets = nn.ModuleDict()
        for i in range(max_Za + 1):
            for j in range(i, max_Za + 1):
                key = f"{i}_{j}"
                self.nets[key] = PairNN(hidden_dim=hidden_dim)
    
    def changeParamByInputDistribution(self,e_size,Ra_stars,Zas,graphbuilders):
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
            self.nets[key].state_dict()['net.0.weight']/=std
            self.nets[key].state_dict()['net.0.bias']-=mean*self.nets[key].state_dict()['net.0.weight'].reshape(-1)



    def reset_parameters(self, nets: dict = None) -> None:
        """
        Reset parameters of all neural networks, or load from provided nets.

        Args:
            nets (dict): Optional dictionary of pre-initialized PairNN networks.
                         Keys must be strings "i_j" with i <= j.
            symmetrize (bool): If True, ensure i_j and j_i share the same parameters.
        """
        if nets is not None:
            self.load_state_dict(nets)

        else:
            print('nets is None')
            '''
            # Randomly reinitialize all nets
            for i in range(self.max_Za + 1):
                for j in range(i, self.max_Za + 1):
                    key = f"{i}_{j}"
                    net = self.nets[key]
                    for layer in net.net:
                        if isinstance(layer, torch.nn.Linear):
                            torch.nn.init.xavier_uniform_(layer.weight)
                            torch.nn.init.zeros_(layer.bias)
            '''

    def forward_single_pair(self, r: torch.Tensor, Zi: int, Zj: int) -> torch.Tensor:
        key = f"{min(Zi,Zj)}_{max(Zi,Zj)}"
        return self.nets[key](r)

    def get_E(self, Ra: FloatTensor, Za: IntTensor, idx_i: IntTensor, idx_j: IntTensor, printFlag=False) -> FloatTensor:
        # Gather atom positions and types
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

        # Compute energy pair by pair
        E_pairs = []
        for r, zi, zj in zip(rij, Zi, Zj):
            if (printFlag): print(r,zi,zj)
            E_pairs.append(self.forward_single_pair(r, int(zi), int(zj)))
        E_pairs = torch.stack(E_pairs)
        return E_pairs.sum() / 2  # divide by 2 to avoid double-counting edges


def _random_nn_ff_param(max_Za: int, hidden_dim: int = 10) -> Dict[str, nn.Module]:
    """
    Initialize a dictionary of randomly-initialized PairNN networks for all (i,j) pairs.
    Keys are strings "i_j" with i <= j.
    """
    nets = {}
    for i in range(max_Za + 1):
        for j in range(i, max_Za + 1):
            key = f"{i}_{j}"
            net = PairNN(hidden_dim=hidden_dim)
            '''
            # Initialize weights: small random values for smooth starting potential
            for layer in net.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            '''
            for net_key in net.state_dict().keys():
                nets['nets.'+key+'.'+net_key] = net.state_dict()[net_key]
    return nets

def _random_nn_system(N: int, N_pairs: int, max_Za: int, start_Za: int = 0) -> Tuple[torch.FloatTensor, torch.IntTensor, torch.IntTensor, torch.IntTensor]:
    """
    Create a single random atomic system with positions, atomic numbers, and pair indices.
    """
    #Ra = torch.rand(N, 3, dtype=torch.float64)
    Ra = torch.randn(N, 3, dtype=torch.float64)
    Ra.requires_grad_()
    Za = torch.randint(start_Za, max_Za + 1, (N,))
    full_combinations = list(combinations(range(N), 2))
    edge_index_undirected = torch.tensor(random.sample(full_combinations, N_pairs))
    idx_i = torch.cat([edge_index_undirected[:, 0], edge_index_undirected[:, 1]])
    idx_j = torch.cat([edge_index_undirected[:, 1], edge_index_undirected[:, 0]])
    return Ra, Za, idx_i, idx_j

def _random_nn_systems(systemN: int, N: int, N_pairs: int, max_Za: int, start_Za: int = 0) -> Tuple[
    List[torch.FloatTensor], List[torch.IntTensor], List[torch.IntTensor], List[torch.IntTensor]]:
    """
    Create multiple random atomic systems.
    Returns lists of Ra, Za, idx_i, idx_j for each system.
    """
    Ras, Zas, idx_is, idx_js = [], [], [], []
    for _ in range(systemN):
        Ra, Za, idx_i, idx_j = _random_nn_system(N, N_pairs, max_Za, start_Za)
        Ras.append(Ra)
        Zas.append(Za)
        idx_is.append(idx_i)
        idx_js.append(idx_j)
    return Ras, Zas, idx_is, idx_js
