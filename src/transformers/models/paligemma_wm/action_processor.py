from typing import Any, List
from ...processing_utils import (
    ProcessorMixin,
)
import torch
import numpy as np

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
        if sum(lengths) == 0:
            # we know that there is no actions, and it is a list of empty actions
            return None, lengths
        # TODO optimize this? 
        max_length = max(lengths)
        # pad with infinite actions
        inputs = [input + [torch.tensor([float('inf'), float('inf')]) for _ in range(max_length - len(input))] for input in inputs]
        
        # flattened_action = [action if isinstance(action, torch.Tensor) else torch.from_numpy(action) for input in inputs for action in input]
        # flattened_action = torch.stack(flattened_action)
        # print(flattened_action.shape)
        # print(lengths)
        return torch.from_numpy(np.array(inputs)), lengths
    