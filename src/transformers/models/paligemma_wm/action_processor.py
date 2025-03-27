from typing import Any, List
from ...processing_utils import (
    ProcessorMixin,
)
import torch
import numpy as np
import typing as tp
class PaliGemmaWMActionProcessor(ProcessorMixin):
    attributes = []
    
    def __init__(
            self,
            action_seq_length: int,
            ):
        self.action_seq_length = action_seq_length    
        super().__init__()
        
    def __call__(self, inputs: tp.Union[List[List[torch.Tensor]], torch.Tensor, np.ndarray]):
        # flatten the actions and return the list of lengths
        # Check if it is a numpy or tensor
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        
        if isinstance(inputs, torch.Tensor):
            assert len(inputs.shape) == 3, "If you pass a tensor, the shape is expected to be B, num_action, act_dim"
            lengths = [inputs.shape[1] for _ in range(len(inputs))]
            return inputs, lengths
            
        
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
    
__all__ = ["PaliGemmaWMActionProcessor"]