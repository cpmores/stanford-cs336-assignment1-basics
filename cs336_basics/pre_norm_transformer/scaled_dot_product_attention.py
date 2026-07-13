import torch
import einops
import math
from cs336_basics.transformer_lm.softmax import Softmax

class SDPA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask):
        d_k = Q.shape[-1]
        d_k_sqrt = math.sqrt(d_k)
        q_kt = einops.einsum(Q, K , "... query d_k, ... key d_k -> ... query key") 
        firstResult = q_kt / d_k_sqrt
        if mask is not None:
            firstResult = firstResult.masked_fill(~mask, float('-inf'))
        softmaxResult = self.softmax.forward(firstResult, dim=-1)
        lastResult = einops.einsum(softmaxResult, V, "... query key, ... key d_v -> ... query d_v")
        return lastResult
    