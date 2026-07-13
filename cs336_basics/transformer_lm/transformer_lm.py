import torch
from cs336_basics.pre_norm_transformer.transformer_block import TransformerBlock
from cs336_basics.transformer_lm.embedding_module import Embedding
from cs336_basics.transformer_lm.linear_module import Linear
from cs336_basics.transformer_lm.softmax import Softmax
from cs336_basics.pre_norm_transformer.root_mean_square import RMSNorm

class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.context_length = context_length

    def init_layers(self, d_model: int, num_heads: int, d_ff: int, rope_theta: float, weights: dict[str, torch.Tensor] | None = None):
        # ── weights ──────────────────────────────────────────────────────────────────────
        key_token_embeddings_weight = "token_embeddings.weight"
        key_ln_final_weight = "ln_final.weight"
        key_lm_head_weight = "lm_head.weight"
        
        # ── layer init ──────────────────────────────────────────────────────────────────────
        self.token_embedding = Embedding(self.vocab_size, d_model)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, self.vocab_size)
        if weights is not None:
            token_embeddings_weight = weights[key_token_embeddings_weight]
            ln_final_weight = weights[key_ln_final_weight]
            lm_head_weight = weights[key_lm_head_weight]
            self.token_embedding.add_weight(token_embeddings_weight)
            self.ln_final.add_weight(ln_final_weight)
            self.lm_head.add_weight(lm_head_weight)
            weights_for_blocks = self._transformer_blocks_weights(self. num_layers, weights)
        self.transformer_blocks = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(self.num_layers)])
        for block in self.transformer_blocks:
            block.mha.add_rope_info(rope_theta, self.context_length)
        if weights is not None:
            for index, block in enumerate(self.transformer_blocks):
                block.add_weight(self.context_length, rope_theta, weights_for_blocks[index])

    def forward(self, x: torch.Tensor):
        x =self.token_embedding.forward(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block.forward(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)
        return x

    def _transformer_blocks_weights(self, num_layers: int, weights: dict[str, torch.Tensor]):
        weights_for_blocks =[] 
        for index in range(num_layers):
            weights_for_block = dict()
            key_attn_q_proj_weight = f"layers.{index}.attn.q_proj.weight"
            key_attn_k_proj_weight = f"layers.{index}.attn.k_proj.weight"
            key_attn_v_proj_weight = f"layers.{index}.attn.v_proj.weight"
            key_attn_o_proj_weight = f"layers.{index}.attn.output_proj.weight"
            key_ln1_weight = f"layers.{index}.ln1.weight"
            key_ln2_weight = f"layers.{index}.ln2.weight"

            key_ffn_w1_weight = f"layers.{index}.ffn.w1.weight"
            key_ffn_w2_weight = f"layers.{index}.ffn.w2.weight"
            key_ffn_w3_weight = f"layers.{index}.ffn.w3.weight"
            weights_for_block["attn.q_proj.weight"] = weights[key_attn_q_proj_weight]
            weights_for_block["attn.k_proj.weight"] = weights[key_attn_k_proj_weight]
            weights_for_block["attn.v_proj.weight"] = weights[key_attn_v_proj_weight]
            weights_for_block["attn.output_proj.weight"] = weights[key_attn_o_proj_weight]
            weights_for_block["ln1.weight"] = weights[key_ln1_weight]
            weights_for_block["ln2.weight"] = weights[key_ln2_weight]
            weights_for_block["ffn.w1.weight"] = weights[key_ffn_w1_weight]
            weights_for_block["ffn.w2.weight"] = weights[key_ffn_w2_weight]
            weights_for_block["ffn.w3.weight"] = weights[key_ffn_w3_weight]
            weights_for_blocks.append(weights_for_block)
        return weights_for_blocks

