# weld_seg_project/model/model.py 主模型定义
import torch.nn as nn
from .attention.multi_head import MultiHeadGeometryAttention
from .attention.global_attention import GlobalAttention
from .attention.local_attention import LocalAttention
from .transformer_block import TransformerBlock

class GeometryAwareTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Input projection: map raw per-point features to model hidden dimension
        self.input_proj = nn.Linear(config.INPUT_DIM, config.D_MODEL) if config.INPUT_DIM != config.D_MODEL else nn.Identity()

        # Create attention layers — alternate between global and local attention
        # Assumption: we alternate layers: even-index layers use GlobalAttention,
        # odd-index layers use LocalAttention. This gives each layer different context.
        self.attention_layers = nn.ModuleList()
        for i in range(config.N_LAYERS):
            if i % 3 == 2:
                attn = GlobalAttention(d_model=config.D_MODEL, n_heads=config.N_HEADS, config=config)
            elif i % 3 == 1:
                attn = LocalAttention(d_model=config.D_MODEL, n_heads=config.N_HEADS, config=config)
            else:
                attn = MultiHeadGeometryAttention(d_model=config.D_MODEL, n_heads=config.N_HEADS, config=config)
            block = TransformerBlock(
                d_model=config.D_MODEL,
                self_attn=attn,
                ffn_dim=config.FFN_DIM
            )
            self.attention_layers.append(block)

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL // 2),
            nn.ReLU(),
            nn.Linear(config.D_MODEL // 2, 1),
            # NOTE: do NOT apply Sigmoid here. Use BCEWithLogitsLoss which expects logits.
        )

        # Reconstruction head: map model features back to input feature dimension
        # Used for self-supervised reconstruction tasks (predict masked features)
        self.recon_head = nn.Linear(config.D_MODEL, config.INPUT_DIM)

    def forward(self, features, principal_dir, curvature, density, normals, linearity, task='class'):
        # project input features to model dimension
        x = self.input_proj(features)
        for block in self.attention_layers:
            x = block(
                x,
                coordinate=features[..., :3],
                principal_dir=principal_dir,
                curvature=curvature,
                density=density,
                normals=normals,
                linearity=linearity
            )
        if task == 'class':
            weld_prob = self.classifier(x)
            return weld_prob
        elif task == 'recon':
            # reconstruct per-point features (same dim as input features)
            recon = self.recon_head(x)
            return recon
        else:
            raise ValueError(f"Unknown task={task}")