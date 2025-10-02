import numpy as np
import ase.io
from scipy.optimize import minimize, OptimizeResult
from typing import Callable, Dict, Sequence, Optional, List
import torch
from torch import FloatTensor, IntTensor
from ..ff.graph import GRAPH_BUILDER_TYPE
from ..ff.baseFF import baseFF
from .calculator import get_ase_atoms
from .geomopt import GeometryFitting

import torch.nn as nn
import torch.optim as optim

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
        fmax: float=1e-4, reoptimize=True
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
            if reoptimize:
                self.Ra_star = self.get_Ra_star(Ra, params_star)
            else:
                self.Ra_star = Ra.to(device, dtype)
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

class MultiSystemOptimizer:
    def __init__(self, 
        Ras: List[FloatTensor], Zas: List[IntTensor], graph_builders: GRAPH_BUILDER_TYPE,
        ff: baseFF, loss_function: Callable[[FloatTensor, FloatTensor], FloatTensor], 
        Ra_stars: Optional[List[FloatTensor]]=None,
        params_star: Optional[Dict[str, FloatTensor]]=None,
        fixed_atom_indices: Optional[Sequence[int]]=None,
        device: torch.device=torch.device("cpu"),
        dtype: torch.dtype=torch.float64,
        fmax: float=1e-4, reoptimize=True
    ):
        self.Zas = [Za.to(device, dtype) for Za in Zas]
        self.graph_builders = graph_builders
        self.fixed_atom_indices = fixed_atom_indices
        self.fitting = GeometryFitting(ff.to(device, dtype), loss_function, fmax)
        self.device = device
        self.dtype = dtype
        if Ra_stars is None:
            if params_star is None:
                raise ValueError("Either Ra_star or params_star must be provided")
            if reoptimize:
                self.Ra_stars = [self.get_Ra_star(graph_builder, Ra, Za, params_star) for graph_builder, Ra, Za in zip(graph_builders, Ras, Zas)]
            else:
                self.Ra_stars = [Ra.to(device, dtype) for Ra in Ras]
            for i, (Ra_star, Za) in enumerate(zip(self.Ra_stars, self.Zas)):
                ase.io.write(f"Ra_star_{i}.xyz", get_ase_atoms(Ra_star, Za))
        else:
            self.Ra_stars = [Ra_star.to(device, dtype).requires_grad_(False) for Ra_star in Ra_stars]

    def objective_function(self, params: np.ndarray) -> float:
        flattened_params = torch.from_numpy(params).to(self.device, self.dtype)
        loss_total=0
        flattened_gradient_total=0
        picked=np.random.choice(len(self.Ra_stars),[min(10,len(self.Ra_stars))],replace=False)
        for index in picked:
            Ra_star = self.Ra_stars[index]
            Za = self.Zas[index]
            graph_builder = self.graph_builders[index]
            loss, flattened_gradient = self.fitting.get_loss_and_gradient_flat(
                flattened_params, graph_builder, 
                Ra_star, Za, 
                self.fixed_atom_indices
            )
            loss_total+=loss
            flattened_gradient_total+=flattened_gradient
        print("loss:", loss_total.item())
        return loss_total.item(), flattened_gradient_total.detach().cpu().numpy()

    def get_Ra_star(self, graph_builder, Ra: FloatTensor, Za: IntTensor, params_star: Dict[str, FloatTensor]) -> FloatTensor:
        self.fitting.reset_parameters(state_dict=params_star)
        Ra_star = self.fitting._get_Ra_star(params_star, graph_builder, Ra, Za, self.fixed_atom_indices)
        return Ra_star.detach().clone()

    def optimize(self, params_0: Dict[str, FloatTensor]) -> FloatTensor:
        res = minimize(
            self.objective_function,
            x0=self.fitting._flatten(params_0),
            jac=True
        )
        return self.fitting._unflatten(torch.from_numpy(res.x).to(self.device, self.dtype))

    def optimize_Adam(self, epoch, params_0: Dict[str, FloatTensor]) -> FloatTensor:
        initial_params_flat = self.fitting._flatten(params_0)
        params = nn.Parameter(torch.tensor(initial_params_flat, dtype=torch.float32))

        # Define the Adam optimizer
        learning_rate = 0.001
        optimizer = optim.Adam([params], lr=learning_rate)
        for step in range(epoch):
            optimizer.zero_grad()
            params_numpy = params.data.numpy()
            scalar_loss, derivative_numpy = self.objective_function(
                x=params_numpy)
            derivative_tensor = torch.tensor(derivative_numpy, dtype=torch.float32)
            params.grad = derivative_tensor
            optimizer.step()
            if (step + 1) % 100 == 0:
                # Use .item() to get the Python scalar from the loss value
                print(f"Step {step + 1}/{epoch}, Loss: {scalar_loss:.6f}")
        return self.fitting._unflatten(params.to(self.device, self.dtype))
