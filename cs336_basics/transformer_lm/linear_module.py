import torch
from torch import nn
from torch.nn.init import trunc_normal_
from jaxtyping import Bool, Float, Int
import math
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        # ── parameters and norms ──────────────────────────────────────────────────────────────────────
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self._trunk_weight(in_features, out_features)


        # ── basic info ──────────────────────────────────────────────────────────────────────
        self.device = device # device to store the parameters on
        self.dtype = dtype # Data type of the parameters
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # using row-major vector
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

    def add_weight(self, weight:Float[torch.Tensor, " d_out d_in"]):
        self.weight.data = weight
    
    def _trunk_weight(self, in_features, out_features):
        square = 2.0 / (in_features + out_features)
        std = math.sqrt(square)
        trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
