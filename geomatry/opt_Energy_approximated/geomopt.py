from typing import Dict, Callable, Optional, Sequence
from collections import OrderedDict
import torch
from torch import FloatTensor, IntTensor
import ase.io
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize.bfgs import BFGS
from .calculator import NNFFCalculator, get_ase_atoms
from ..ff.baseFF import baseFF
from ..ff.graph import GRAPH_BUILDER_TYPE
from ..ff.constraints import reduce_coordinates
from ..opt.implicit_gradient import get_dRa_star_div_dtheta

class GeometryFitting:
    def __init__(self, ff: baseFF, loss_function: Callable[[FloatTensor, FloatTensor], FloatTensor], fmax: float=1e-4):
        self.ff = ff
        self.loss_function = loss_function
        self.fmax = fmax

    def reset_parameters(self, state_dict: Optional[Dict[str, FloatTensor]]=None, **params) -> None:
        if state_dict is not None:
            self.ff.load_state_dict(state_dict)
        else:
            self.ff.reset_parameters(**params)

    def _unflatten(self, flattened_params: FloatTensor) -> Dict[str, FloatTensor]:
        unflattened_params = OrderedDict()
        numel = 0
        for name, param in self.ff.named_parameters():
            param_size = param.numel()
            unflattened_params[name] = flattened_params[numel:numel+param_size].view(param.shape)
            numel += param_size
        return unflattened_params

    def _flatten(self, gradients: Dict[str, FloatTensor]) -> FloatTensor:
        flattened_gradients = []
        for gradient in gradients.values():
            flattened_gradients.append(gradient.view(-1))
        return torch.cat(flattened_gradients)
    
    def _get_Ra_star(self, params: Dict[str, FloatTensor], graph_builder: GRAPH_BUILDER_TYPE, Ra: FloatTensor, Za: IntTensor, fixed_atom_indices: Optional[Sequence[int]]=None) -> FloatTensor:
        self.reset_parameters(state_dict=params)
        atoms = get_ase_atoms(Ra, Za)
        atoms.calc = NNFFCalculator(self.ff, graph_builder)
        if fixed_atom_indices is not None:
            atoms.set_constraint(FixAtoms(indices=fixed_atom_indices))
        optimizer = BFGS(atoms, logfile="/dev/null")
        optimizer.run(steps=1000, fmax=0.001)
        ase.io.write("Ra_star_traj.xyz", atoms, append=True)
        return torch.from_numpy(atoms.get_positions()).to(Ra.device, Ra.dtype).requires_grad_(True)

    def _get_loss_and_gradient(self, params: Dict[str, FloatTensor], graph_builder: GRAPH_BUILDER_TYPE, Ra_star_label: FloatTensor, Za: IntTensor, fixed_atom_indices: Optional[Sequence[int]]=None) -> FloatTensor:
        atoms = get_ase_atoms(Ra_star_label, Za)
        Ra_star_model = self._get_Ra_star(params, graph_builder, Ra_star_label, Za, fixed_atom_indices)

        fixed_cartesian_indices = torch.tensor([[i * 3, i * 3 + 1, i * 3 + 2] for i in fixed_atom_indices]).to(Za.device).view(-1)
        
        E = self.ff.get_E(Ra_star_model, Za, *graph_builder(atoms))
        Fa = self.ff.get_Fa(E, Ra_star_model)
        Hessian = self.ff.get_Hessian(Fa, Ra_star_model)
        dRa_star_div_dtheta = get_dRa_star_div_dtheta(self.ff, Fa.view(-1), Hessian, fixed_cartesian_indices) # (N_params, 3 * N_atoms - N_fixed_cart)

        loss = self.loss_function(Ra_star_label, Ra_star_model)
        dloss_div_dRa = reduce_coordinates(torch.autograd.grad(loss, Ra_star_model)[0], fixed_cartesian_indices) # (3 * N_atoms - N_fixed_cart)
        dloss_div_dtheta = OrderedDict()
        for name, grad in dRa_star_div_dtheta.items():
            dloss_div_dtheta[name] = grad @ dloss_div_dRa
        return loss, dloss_div_dtheta

    def get_loss_and_gradient_flat(self, flattened_params: FloatTensor, graph_builder: GRAPH_BUILDER_TYPE, Ra: FloatTensor, Za: IntTensor, fixed_atom_indices: Optional[Sequence[int]]=None) -> FloatTensor:
        params = self._unflatten(flattened_params)
        loss, gradients = self._get_loss_and_gradient(params, graph_builder, Ra, Za, fixed_atom_indices)
        return loss, self._flatten(gradients)
