import sys
sys.path.append("..")
from geomatry.opt.scipy_optimizer import SingleSystemOptimizer
from geomatry.opt.loss import rmsd_loss
from geomatry.ff.graph import get_given_graph_builder
from geomatry.ff.springs import SpringFF, _random_spring_system, _random_spring_ff_param
import torch

if __name__ == "__main__":
    N = 50
    N_pairs = 200
    max_Za = 3
    Ra, Za, idx_i, idx_j = _random_spring_system(N, N_pairs, max_Za, start_Za=1)
    k, r0 = _random_spring_ff_param(max_Za, r0_max=5, k_max=5)
    ff = SpringFF(max_Za)
    ff.reset_parameters(k, r0)
    
    fixed_atom_indices = [0, 1, 2]
    graph_builder = get_given_graph_builder(idx_i, idx_j)

    # start implicit optimization
    params_star = ff.state_dict()
    print("params_star", params_star)
    optimizer = SingleSystemOptimizer(
        Ra, Za, graph_builder, ff, rmsd_loss, 
        params_star=params_star,
        fixed_atom_indices=fixed_atom_indices,
        fmax=1e-5
    )

    k_perturbed = torch.clamp(k.clone() + torch.randn_like(k) * 0.1, min=0)
    r0_perturbed = torch.clamp(r0.clone() + torch.randn_like(r0) * 0.1, min=0)
    ff.reset_parameters(k_perturbed, r0_perturbed)
    params_0 = ff.state_dict()
    print("params_0", params_0)
    print("params_optimized",optimizer.optimize(params_0))