import torch
import math
from typing import Optional
from collections.abc import Callable, Iterable

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data = p.data - lr * weight_decay * p.data 
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                p.data = p.data - alpha_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
            
        return loss


def get_lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + 1.0 / 2 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(parameters, max_l2_norm):
    total_norm = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2) ** 2
        total_norm += param_norm.item()
    total_norm = total_norm ** 0.5

    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)