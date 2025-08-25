from typing import Optional
import warnings
from collections import OrderedDict
import torch
from torch import FloatTensor, IntTensor
from ..ff.baseFF import baseFF
from ..ff.constraints import reduce_force, reduce_hessian


def get_dFa_div_dtheta_loop(model: baseFF, forces_flattened: FloatTensor) -> FloatTensor:
    '''
    Compute the gradient of the force with respect to the model parameters by looping over the parameters and force components.
    '''
    dFa_div_dtheta = OrderedDict()
    for name, param in model.named_parameters():
        dFa_div_dtheta[name] = torch.empty((*param.shape, forces_flattened.shape[0]), dtype=forces_flattened.dtype, device=forces_flattened.device)
        for i in range(forces_flattened.shape[0]):
            grad_component = torch.autograd.grad(
                forces_flattened[i], param, create_graph=False, retain_graph=True
            )[0]
            dFa_div_dtheta[name][..., i] = grad_component
    return dFa_div_dtheta


def get_dFa_div_dtheta_vmap(model: baseFF, forces_flattened: FloatTensor) -> FloatTensor:
    '''
    Compute the gradient of the force with respect to the model parameters by vmap.
    '''
    dFa_div_dtheta = OrderedDict()
    num_elements = forces_flattened.shape[0]
    for name, param in model.named_parameters():
        def get_vjp(v):
            # Compute gradient of forces with respect to the parameter
            gradient = torch.autograd.grad(
                forces_flattened,
                param,  # Use the original parameter, not flattened
                v,
                retain_graph=True,
                create_graph=False, 
                allow_unused=True,
            )
            return gradient
        I_N = torch.eye(num_elements, dtype=forces_flattened.dtype, device=forces_flattened.device)
        chunk_size = 1 if num_elements < 64 else 16
        component = torch.vmap(get_vjp, in_dims=0, out_dims=-1, chunk_size=chunk_size)(
            I_N
        )[0]
        dFa_div_dtheta[name] = component
    return dFa_div_dtheta


def get_dFa_div_dtheta_backprop(model: baseFF, forces_flattened: FloatTensor) -> FloatTensor:
    '''
    Compute the gradient of the force with respect to the model parameters by backpropagation.
    '''
    dFa_div_dtheta = OrderedDict()
    for name, param in model.named_parameters():
        dFa_div_dtheta[name] = torch.empty((*param.shape, forces_flattened.shape[0]), dtype=forces_flattened.dtype, device=forces_flattened.device)
        param.grad = None
    for i, force_component in enumerate(forces_flattened):
        force_component.backward(retain_graph=True)
        for name, param in model.named_parameters():
            dFa_div_dtheta[name][..., i] = param.grad
            param.grad = None
    return dFa_div_dtheta


def get_dRa_star_div_dtheta(model: baseFF, forces_flattened: FloatTensor, Hessian: FloatTensor, fixed_indices: Optional[IntTensor]=None) -> FloatTensor:
    '''
    Compute the gradient of the Ra* with respect to the model parameters.
    '''
    if fixed_indices is None or len(fixed_indices) <= 6:
        warnings.warn("Fixing more than 6 degrees of freedom are recommended to avoid numerical instability.")
    if fixed_indices is not None:
        Hessian = reduce_hessian(Hessian, fixed_indices)
        forces_flattened = reduce_force(forces_flattened, fixed_indices)

    dRa_star_div_dtheta = OrderedDict()
    dFa_div_dtheta = get_dFa_div_dtheta_vmap(model, forces_flattened)
    for name, grad in dFa_div_dtheta.items():
        dRa_star_div_dtheta[name] = grad @ torch.linalg.inv(Hessian)
    return dRa_star_div_dtheta
