import numpy as np
import ase.io
from scipy.optimize import minimize, OptimizeResult
from typing import Callable, Dict, Sequence, Optional
import torch
from torch import FloatTensor, IntTensor
from ..ff.graph import GRAPH_BUILDER_TYPE
from ..ff.baseFF import baseFF
from .calculator import get_ase_atoms
from .geomopt import GeometryFitting


def callback(intermediate_result: OptimizeResult):
    print("this is a callback")

class SingleSystemOptimizer:
    def __init__(self, 
        Ra: FloatTensor, Za: IntTensor, graph_builder: GRAPH_BUILDER_TYPE,
        ff: baseFF, loss_function: Callable[[FloatTensor, FloatTensor], FloatTensor], 
        Ra_star: Optional[FloatTensor]=None,
        params_star: Optional[Dict[str, FloatTensor]]=None,
        fixed_atom_indices: Optional[Sequence[int]]=None,
        device: torch.device=torch.device("cpu"),
        dtype: torch.dtype=torch.float64,
        fmax: float=1e-4
    ):
        self.Za = Za.to(device, dtype)
        self.graph_builder = graph_builder
        self.fixed_atom_indices = fixed_atom_indices
        self.fitting = GeometryFitting(ff.to(device, dtype), loss_function, fmax)
        self.device = device
        self.dtype = dtype
        if Ra_star is None:
            if params_star is None:
                raise ValueError("Either Ra_star or params_star must be provided")
            self.Ra_star = self.get_Ra_star(Ra, params_star)
            ase.io.write("Ra_star.xyz", get_ase_atoms(self.Ra_star, self.Za))
        else:
            self.Ra_star = Ra_star.to(device, dtype).requires_grad_(False)

    def objective_function(self, params: np.ndarray) -> float:
        flattened_params = torch.from_numpy(params).to(self.device, self.dtype)
        loss, flattened_gradient = self.fitting.get_loss_and_gradient_flat(
            flattened_params, self.graph_builder, 
            self.Ra_star, self.Za, 
            self.fixed_atom_indices
        )
        print("loss:", loss.item())
        return loss.item(), flattened_gradient.detach().cpu().numpy()

    def get_Ra_star(self, Ra: FloatTensor, params_star: Dict[str, FloatTensor]) -> FloatTensor:
        self.fitting.reset_parameters(state_dict=params_star)
        Ra_star = self.fitting._get_Ra_star(params_star, self.graph_builder, Ra, self.Za, self.fixed_atom_indices)
        return Ra_star.detach().clone()

    def optimize(self, params_0: Dict[str, FloatTensor]) -> FloatTensor:
        res = minimize(
            self.objective_function,
            x0=self.fitting._flatten(params_0),
            jac=True
        )
        return self.fitting._unflatten(torch.from_numpy(res.x).to(self.device, self.dtype))
