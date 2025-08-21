from typing import Optional
from collections import OrderedDict
import torch
from torch import FloatTensor, IntTensor
from ..ff.baseFF import baseFF
from ..ff.constraints import reduce_force, reduce_hessian


def get_dFa_div_dtheta_loop(model: baseFF, Fa: FloatTensor) -> FloatTensor:
    '''
    Compute the gradient of the force with respect to the model parameters by looping over the parameters and force components.
    '''
    dFa_div_dtheta = OrderedDict()
    forces_flatten = Fa.view(-1)
    for name, param in model.named_parameters():
        dFa_div_dtheta[name] = torch.empty((*param.shape, forces_flatten.shape[0]), dtype=Fa.dtype, device=Fa.device)
        for i in range(forces_flatten.shape[0]):
            grad_component = torch.autograd.grad(
                forces_flatten[i], param, create_graph=False, retain_graph=True
            )[0]
            dFa_div_dtheta[name][..., i] = grad_component
    return dFa_div_dtheta


def get_dFa_div_dtheta_vmap(model: baseFF, Fa: FloatTensor) -> FloatTensor:
    '''
    Compute the gradient of the force with respect to the model parameters by vmap.
    '''
    dFa_div_dtheta = OrderedDict()
    forces_flatten = Fa.view(-1)
    num_elements = forces_flatten.shape[0]
    for name, param in model.named_parameters():
        def get_vjp(v):
            # Compute gradient of forces with respect to the parameter
            gradient = torch.autograd.grad(
                forces_flatten,
                param,  # Use the original parameter, not flattened
                v,
                retain_graph=True,
                create_graph=False, 
                allow_unused=True,
            )
            return gradient
        I_N = torch.eye(num_elements, dtype=Fa.dtype, device=Fa.device)
        chunk_size = 1 if num_elements < 64 else 16
        component = torch.vmap(get_vjp, in_dims=0, out_dims=-1, chunk_size=chunk_size)(
            I_N
        )[0]
        dFa_div_dtheta[name] = component
    return dFa_div_dtheta


def get_dFa_div_dtheta_backprop(model: baseFF, Fa: FloatTensor) -> FloatTensor:
    '''
    Compute the gradient of the force with respect to the model parameters by backpropagation.
    '''
    dFa_div_dtheta = OrderedDict()
    forces_flatten = Fa.view(-1)
    for name, param in model.named_parameters():
        dFa_div_dtheta[name] = torch.empty((*param.shape, forces_flatten.shape[0]), dtype=Fa.dtype, device=Fa.device)
        param.grad = None
    for i, force_component in enumerate(forces_flatten):
        force_component.backward(retain_graph=True)
        for name, param in model.named_parameters():
            dFa_div_dtheta[name][..., i] = param.grad
            param.grad = None
    return dFa_div_dtheta
