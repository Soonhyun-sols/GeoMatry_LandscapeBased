import sys
sys.path.append("..")
from geomatry.ff.springs import SpringFF
from geomatry.ff.springs import _random_spring_ff_param, _random_spring_system
import torch

def test_dFa_div_dtheta_vmap():
    from geomatry.opt.implicit_gradient import get_dFa_div_dtheta_vmap, get_dFa_div_dtheta_loop
    N = 10
    N_pairs = N * (N - 1) // 4
    max_Za = 10
    Ra, Za, idx_i, idx_j = _random_spring_system(N=N, N_pairs=N_pairs, max_Za=max_Za)
    params = _random_spring_ff_param(max_Za=max_Za)
    model = SpringFF(max_Za=max_Za)
    model.reset_parameters(*params)
    model = model.to(dtype=torch.float64)
    E = model.get_E(Ra, Za, idx_i, idx_j)
    Fa = model.get_Fa(E, Ra)

    dFa_div_dtheta_vmap = get_dFa_div_dtheta_vmap(model, Fa.view(-1))
    dFa_div_dtheta_loop = get_dFa_div_dtheta_loop(model, Fa.view(-1))

    assert dFa_div_dtheta_vmap.keys() == dFa_div_dtheta_loop.keys()
    for name in dFa_div_dtheta_vmap.keys():
        assert dFa_div_dtheta_vmap[name].shape == dFa_div_dtheta_loop[name].shape
        assert torch.allclose(dFa_div_dtheta_vmap[name], dFa_div_dtheta_loop[name])