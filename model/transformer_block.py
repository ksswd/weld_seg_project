# weld_seg_project/model/transformer_block.py Transformer块
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, self_attn, ffn_dim):
        super().__init__()
        self.self_attn = self_attn
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, **kwargs)
        x = x + attn_output
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        
        return x