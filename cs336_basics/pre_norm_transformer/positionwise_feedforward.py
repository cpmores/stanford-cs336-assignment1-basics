from einops import einsum
import torch

class Swiglu(torch.nn.Module):
    def __init__(self, 
                d_model: int, 
                d_ff: int,
                device=None,
                dtype=None
                ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # ── basic info ──────────────────────────────────────────────────────────────────────
        self.device = device
        self.dtype = dtype

        # ── weight ──────────────────────────────────────────────────────────────────────
        self.w1_weight = torch.nn.Parameter(torch.randn(self.d_ff, self.d_model))
        self.w2_weight = torch.nn.Parameter(torch.randn(self.d_model, self.d_ff))
        self.w3_weight = torch.nn.Parameter(torch.randn(self.d_ff, self.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        input1 = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        value = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        gate = self._silu(input1)
        hidden = gate * value
        return einsum(hidden, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def add_weight(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor):
        self.w1_weight.data = w1
        self.w2_weight.data = w2
        self.w3_weight.data = w3