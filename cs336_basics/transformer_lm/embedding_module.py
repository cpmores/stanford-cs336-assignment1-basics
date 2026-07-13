import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__() 
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self._trunk_weight()

        # ── base info ──────────────────────────────────────────────────────────────────────
        self.device = device
        self.dtype = dtype
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

    def add_weight(self, weight):
        self.weight.data = weight
    
    def _trunk_weight(self):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)
