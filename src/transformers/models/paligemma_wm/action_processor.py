from typing import Any, List
from ...processing_utils import (
    ProcessorMixin,
)
import torch

class PaliGemmaWMActionProcessor(ProcessorMixin):
    attributes = []
    
    def __init__(
            self,
            action_seq_length: int,
            ):
        self.action_seq_length = action_seq_length    
        super().__init__()
        
    def __call__(self, inputs: List[List[torch.Tensor]]):
        # flatten the actions and return the list of lengths
        lengths = [len(input) for input in inputs]
        # TODO optimize this? 
        flattened_action = [action if isinstance(action, torch.Tensor) else torch.from_numpy(action) for input in inputs for action in input]

        return torch.stack(flattened_action), lengths
    