from typing import Callable, Tuple
from ase import Atoms
from torch import IntTensor

GRAPH_BUILDER_TYPE = Callable[[Atoms], Tuple[IntTensor, IntTensor]]

def get_given_graph_builder(idx_i: IntTensor, idx_j: IntTensor) -> GRAPH_BUILDER_TYPE:
    def graph_builder(atoms: Atoms) -> GRAPH_BUILDER_TYPE:
        return idx_i, idx_j
    return graph_builder
