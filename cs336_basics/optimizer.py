from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import torch.nn as nn

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr, "betas":betas, "weight_decay":weight_decay, "eps":eps}
        
        super().__init__(params, defaults)  

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * torch.square(grad)
                m_b1 = m / (1 - (betas[0] ** t))
                v_b2 = v / (1 - betas[1] ** t)
                grad_update = lr * (m_b1 / (torch.sqrt(v_b2) + eps))
                weight_decay_update = lr * weight_decay * p.data

                p.data -= grad_update + weight_decay_update

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
    
    @classmethod
    def cosine_schedule(cls, it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
        if it < warmup_iters:
            lr = (it/warmup_iters) * max_learning_rate
        elif it >= warmup_iters and it <= cosine_cycle_iters:
            lr = min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters)/(cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
        else:
            lr = min_learning_rate
        return lr
    
    @classmethod
    def gradient_clipping(cls, parameters, max_l2_norm, eps=10e-6):
        for p in parameters:
            l2 = torch.norm(p.grad.data)
            if l2 > max_l2_norm:
                scale = max_l2_norm * (l2 + eps)
                p.grad.data = p.grad.data * scale
        return
