from typing import Any, List
from ...processing_utils import (
    ProcessorMixin,
)
from torch import Tensor

class PaliGemmaWMActionProcessor(ProcessorMixin):
    attributes = []
    
    def __init__(
            self,
            action_seq_length: int,
            ):
        self.action_seq_length = action_seq_length    
        super().__init__()
        
    def __call__(self, inputs: List[List[Tensor]]):
        # dummy action processing code. Can do something smarter here.
        return inputs, [len(input) for input in inputs]
    