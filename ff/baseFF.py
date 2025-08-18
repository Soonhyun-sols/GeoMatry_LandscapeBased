from abc import ABC, abstractmethod
from torch.nn import Module
from torch import FloatTensor, IntTensor


class baseFF(Module, ABC):
    def __init__(self, max_Za: int):
        super().__init__()
        self.max_Za = max_Za

    @abstractmethod
    def get_E(self, Ra: FloatTensor, Za: IntTensor, idx_i: IntTensor, idx_j: IntTensor) -> FloatTensor:
        pass
    
    @abstractmethod
    def get_Fa(self, E: FloatTensor, Ra: FloatTensor) -> FloatTensor:
        pass
    