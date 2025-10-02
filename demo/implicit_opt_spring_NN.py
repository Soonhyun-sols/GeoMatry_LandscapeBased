import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from geomatry.opt.scipy_optimizer import SingleSystemOptimizer
from geomatry.opt.loss import rmsd_loss
from geomatry.ff.graph import get_given_graph_builder
from geomatry.ff.NN import NNPotentialFF, _random_nn_system, _random_nn_ff_param
from geomatry.ff.springs import SpringFF, _random_spring_system, _random_spring_ff_param
import torch
import copy

if __name__ == "__main__":
    N = 7
    N_pairs = 21
    max_Za = 1

    Ra, Za, idx_i, idx_j = _random_spring_system(N, N_pairs, max_Za, start_Za=1)
    params = _random_spring_ff_param(max_Za)
    ff = SpringFF(max_Za)
    ff.reset_parameters(*params)

    fixed_atom_indices = [0,1,2,3,4,5]
    graph_builder=get_given_graph_builder(idx_i, idx_j)
    k, r0 = _random_spring_ff_param(max_Za,k_max=10,r0_max=2)

    r0[1,1] = 0.7
    print(k, r0)
    ff_spring = SpringFF(max_Za)
    ff_spring.reset_parameters(k, r0)
    import copy
    params_spring = copy.deepcopy(ff_spring.state_dict())
    optimizer_spring = SingleSystemOptimizer(
        Ra, Za, graph_builder, ff_spring, rmsd_loss, 
        params_star=params_spring,
        fixed_atom_indices=fixed_atom_indices,
        fmax=1e-5, reoptimize=True
    )

    optimizer = SingleSystemOptimizer(
        optimizer_spring.Ra_star, Za, graph_builder, ff, rmsd_loss, 
        params_star=params,
        fixed_atom_indices=fixed_atom_indices,
        fmax=1e-5, reoptimize=False
    )

    # start implicit optimization
    optimizer.optimize(params)