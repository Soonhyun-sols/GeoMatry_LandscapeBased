import torch
from torch import IntTensor
from ase.calculators.calculator import Calculator
from typing import Callable, Tuple, List
from ase import Atoms
from ..ff import baseFF


class NNFFCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model: baseFF, graph_builder: Callable[[Atoms], Tuple[IntTensor, IntTensor]], **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.graph_builder = graph_builder

    def calculate(self, 
        atoms: Atoms, 
        properties: List[str]=["energy", "forces"], 
        system_changes: List[str]=["positions", "cell"]
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        self.model.eval()
        with torch.no_grad():
            idx_i, idx_j = self.graph_builder(atoms)
            energy = self.model.get_E(atoms.positions, atoms.numbers, idx_i, idx_j)
            forces = -self.model.get_Fa(energy, atoms.positions, atoms.numbers)
        self.results["energy"] = energy
        self.results["forces"] = forces
        
        