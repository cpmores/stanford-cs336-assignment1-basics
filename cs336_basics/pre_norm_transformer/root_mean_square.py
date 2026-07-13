from torch import nn
import torch
from einops import einsum

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()

        # ── layer info ──────────────────────────────────────────────────────────────────────
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model))

        # ── basic info ──────────────────────────────────────────────────────────────────────
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = self._rms(x)
        normal = x / rms
        result = einsum(normal, self.gain, "... d_model, d_model -> ... d_model")
        return result.to(in_dtype)

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
    
    def add_weight(self, weight: torch.Tensor):
        self.gain.data = weight