from abc import ABC, abstractmethod
import torch
from torch.nn import Module
from torch import FloatTensor, IntTensor
from .hessian import compute_hessians_vmap

class baseFF(Module, ABC):
    def __init__(self, max_Za: int):
        super().__init__()
        self.max_Za = max_Za

    @abstractmethod
    def get_E(self, Ra: FloatTensor, Za: IntTensor, idx_i: IntTensor, idx_j: IntTensor) -> FloatTensor:
        pass
    
    def get_Fa(self, E: FloatTensor, Ra: FloatTensor) -> FloatTensor:
        Fa = -torch.autograd.grad(E, Ra, create_graph=True, retain_graph=True)[0]
        return Fa 
    
    def get_Haa(self, Fa: FloatTensor, Ra: FloatTensor) -> FloatTensor:
        return compute_hessians_vmap(Fa, Ra)
    