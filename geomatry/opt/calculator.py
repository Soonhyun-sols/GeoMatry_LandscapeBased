import torch
from ase.calculators.calculator import Calculator
from typing import List
from ase import Atoms
from ..ff import baseFF, GRAPH_BUILDER_TYPE

class NNFFCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model: baseFF, graph_builder: GRAPH_BUILDER_TYPE, **kwargs):
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
        idx_i, idx_j = self.graph_builder(atoms)
        Ra = torch.tensor(atoms.positions, requires_grad=True)
        Za = torch.tensor(atoms.numbers)
        energy = self.model.get_E(Ra, Za, idx_i, idx_j)
        forces = -self.model.get_Fa(energy, Ra)
        self.results["energy"] = energy.detach().numpy()
        self.results["forces"] = forces.detach().numpy()
        
        