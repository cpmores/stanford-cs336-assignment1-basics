import torch
import math
from einops import rearrange 

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        cos_buf, sin_buf = self._precompute()
        self.register_buffer("cos_buf", cos_buf, persistent=False)
        self.register_buffer("sin_buf", sin_buf, persistent=False)

        # ── base info ──────────────────────────────────────────────────────────────────────
        self.device = device

    def forward(self, x: torch.Tensor, token_positions) -> torch.Tensor:
        x_pairs = rearrange(x, "... seq (pair two) -> ... seq pair two", two=2)
        cos = self.cos_buf[token_positions]
        sin = self.sin_buf[token_positions]

        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
        rotated = x_pairs.clone()
        rotated[..., 0] = x1 * cos - x2 * sin
        rotated[..., 1] = x1 * sin + x2 * cos
        return rearrange(rotated, "... seq pair two -> ... seq (pair two)", two=2)

    def _precompute(self) -> tuple[torch.Tensor, torch.Tensor]: 
        cos_buf = torch.zeros(self.max_seq_len, self.d_k // 2)
        sin_buf = torch.zeros(self.max_seq_len, self.d_k // 2)
        for i in range(0, self.max_seq_len):
            for k in range(0, self.d_k // 2):
                cos_buf[i, k] = math.cos(self._freq(i, k))
                sin_buf[i, k] = math.sin(self._freq(i, k))
        return cos_buf, sin_buf

    def _freq(self, position: int, pair: int):
        return position / (self.theta ** ((2 * pair) / self.d_k))