import torch
from ase.calculators.calculator import Calculator
from typing import List
from ase import Atoms
from torch import FloatTensor, IntTensor
from ..ff import baseFF, GRAPH_BUILDER_TYPE


def get_ase_atoms(Ra: FloatTensor, Za: IntTensor) -> Atoms:
    return Atoms(
        positions=Ra.detach().cpu().numpy(),
        numbers=Za.detach().cpu().numpy()
    )


class NNFFCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, ff: baseFF, graph_builder: GRAPH_BUILDER_TYPE, **kwargs):
        super().__init__(**kwargs)
        self.ff = ff
        self.graph_builder = graph_builder

    def calculate(self, 
        atoms: Atoms, 
        properties: List[str]=["energy", "forces"], 
        system_changes: List[str]=["positions", "cell"]
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        self.ff.eval()
        idx_i, idx_j = self.graph_builder(atoms)
        Ra = torch.tensor(atoms.positions, requires_grad=True)
        Za = torch.tensor(atoms.numbers)
        energy = self.ff.get_E(Ra, Za, idx_i, idx_j)
        forces = self.ff.get_Fa(energy, Ra)
        self.results["energy"] = energy.detach().cpu().numpy()
        self.results["forces"] = forces.detach().cpu().numpy()
        