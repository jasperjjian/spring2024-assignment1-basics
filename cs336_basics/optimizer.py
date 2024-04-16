from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import torch.nn as nn

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)  

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        return loss