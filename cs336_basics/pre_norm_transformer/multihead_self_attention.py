import torch
import einops
from cs336_basics.pre_norm_transformer.scaled_dot_product_attention import SDPA
from cs336_basics.pre_norm_transformer.relative_positional_embeddings import RoPE

class MHA(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention = SDPA()

        self.weight_q = torch.nn.Parameter(torch.randn(d_model, d_model))
        self.weight_k = torch.nn.Parameter(torch.randn(d_model, d_model))
        self.weight_v = torch.nn.Parameter(torch.randn(d_model, d_model))
        self.weight_o = torch.nn.Parameter(torch.randn(d_model, d_model))

        # ── RoPE info ──────────────────────────────────────────────────────────────────────
        self.rope = None

    def forward(self, input: torch.Tensor):
        Q = einops.einsum(input, self.weight_q, "... seq d_in, d_out d_in -> ... seq d_out")
        K = einops.einsum(input, self.weight_k, "... seq d_in, d_out d_in -> ... seq d_out")
        V = einops.einsum(input, self.weight_v, "... seq d_in, d_out d_in -> ... seq d_out")

        Q = einops.rearrange(Q, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        K = einops.rearrange(K, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        V = einops.rearrange(V, "... seq (h d_v) -> ... h seq d_v", h=self.num_heads)

        seq_len = input.shape[-2]  
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        causal_mask = causal_mask.to(input.device)

        headResult =  self.attention.forward(Q, K, V, causal_mask)
        headResult = einops.rearrange(headResult, "... h seq d_k -> ... seq (h d_k)")
        return einops.einsum(headResult, self.weight_o, "... seq d_in, d_out d_in -> ... seq d_out")

    def forward_with_rope(self, input: torch.Tensor, token_positions):
        if isinstance(self.rope, RoPE):
            Q = einops.einsum(input, self.weight_q, "... seq d_in, d_out d_in -> ... seq d_out")
            K = einops.einsum(input, self.weight_k, "... seq d_in, d_out d_in -> ... seq d_out")
            V = einops.einsum(input, self.weight_v, "... seq d_in, d_out d_in -> ... seq d_out")

            Q = einops.rearrange(Q, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
            K = einops.rearrange(K, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
            V = einops.rearrange(V, "... seq (h d_v) -> ... h seq d_v", h=self.num_heads)

            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)

            seq_len = input.shape[-2]  
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            causal_mask = causal_mask.to(input.device)
            headResult =  self.attention.forward(Q, K, V, causal_mask)
            headResult = einops.rearrange(headResult, "... h seq d_k -> ... seq (h d_k)")
            return einops.einsum(headResult, self.weight_o, "... seq d_in, d_out d_in -> ... seq d_out")
        else:
            return input

    
    def add_rope_info(self, theta: float, max_seq_len: int):
        self.rope = RoPE(theta, self.d_model // self.num_heads, max_seq_len) 

    def add_weight(self, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor):
        self.weight_q.data = wq
        self.weight_k.data = wk
        self.weight_v.data = wv
        self.weight_o.data = wo



