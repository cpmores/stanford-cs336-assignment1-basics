import torch
from cs336_basics.pre_norm_transformer.multihead_self_attention import MHA
from cs336_basics.pre_norm_transformer.root_mean_square import RMSNorm
from cs336_basics.pre_norm_transformer.positionwise_feedforward import Swiglu

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.mha = MHA(d_model, num_heads)
        self.mha_RMSNorm = RMSNorm(d_model)
        self.ffn = Swiglu(d_model, d_ff)
        self.ffn_RMSNorm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, token_positions=None):
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        y = x + self.mha.forward_with_rope(self.mha_RMSNorm.forward(x), token_positions)
        return y + self.ffn.forward(self.ffn_RMSNorm.forward(y))

    def add_weight(self, max_seq_len: int, theta: float, weights: dict[str, torch.Tensor]):
        attn_q_proj_weight = weights["attn.q_proj.weight"]
        attn_k_proj_weight = weights["attn.k_proj.weight"]
        attn_v_proj_weight = weights["attn.v_proj.weight"]
        attn_output_proj_weight = weights["attn.output_proj.weight"]
        ln1_weight = weights["ln1.weight"]
        ln2_weight = weights["ln2.weight"]
        ffn_w1_weight = weights["ffn.w1.weight"]
        ffn_w2_weight = weights["ffn.w2.weight"]
        ffn_w3_weight = weights["ffn.w3.weight"]

        self.mha.add_rope_info(theta, max_seq_len)
        self.mha.add_weight(attn_q_proj_weight, attn_k_proj_weight, attn_v_proj_weight, attn_output_proj_weight)
        self.mha_RMSNorm.add_weight(ln1_weight)
        self.ffn.add_weight(ffn_w1_weight, ffn_w2_weight, ffn_w3_weight)
        self.ffn_RMSNorm.add_weight(ln2_weight)
