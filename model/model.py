# weld_seg_project/model/model.py 主模型定义
import torch
import torch.nn as nn
from .attention.multi_head import MultiHeadGeometryAttention
from .attention.global_attention import GlobalAttention
from .attention.local_attention import LocalAttention
from .transformer_block import TransformerBlock


class SimpleMLP(nn.Module):
    """简单的全连接网络版本，用于对比测试"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 输入特征维度：原始特征 + 几何特征
        # features(8) + coordinate(3) + principal_dir(3) + curvature(1) +
        # density(1) + normals(3) + linearity(1) = 20维
        input_dim = config.INPUT_DIM + 3 + 3 + 1 + 1 + 3 + 1  # 8 + 12 = 20

        # 简单的MLP网络
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.D_MODEL * 4),
            nn.ReLU(),
            nn.Linear(config.D_MODEL * 4, config.D_MODEL * 2),
            nn.ReLU(),
            nn.Linear(config.D_MODEL * 2, config.D_MODEL),
            nn.ReLU(),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL // 2),
            nn.ReLU(),
            nn.Linear(config.D_MODEL // 2, 1),
        )

        # 重建头
        self.recon_head = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL * 2),
            nn.ReLU(),
            nn.Linear(config.D_MODEL * 2, config.D_MODEL),
            nn.ReLU(),
            nn.Linear(config.D_MODEL, 1)  # curvature only
        )

    def forward(self, features, coordinate, principal_dir, curvature, density, normals, linearity, task='class'):
        # 拼接所有输入特征
        x = torch.cat([
            features,           # (B, N, 8)
            coordinate,         # (B, N, 3)
            principal_dir,      # (B, N, 3)
            curvature,          # (B, N, 1)
            density,            # (B, N, 1)
            normals,            # (B, N, 3)
            linearity           # (B, N, 1)
        ], dim=-1)  # (B, N, 20)

        # 通过MLP网络
        x = self.network(x)  # (B, N, D_MODEL)

        if task == 'class':
            weld_prob = self.classifier(x)
            return weld_prob
        elif task == 'recon':
            recon = self.recon_head(x)
            return recon
        else:
            raise ValueError(f"Unknown task={task}")

class GeometryAwareTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_type = getattr(config, 'MODEL_TYPE', 'transformer').lower()

        if model_type == 'mlp':
            # 使用简单的MLP模型
            self.model = SimpleMLP(config)
        elif model_type == 'transformer':
            # 使用复杂的Transformer模型
            self.model = TransformerModel(config)
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {model_type}. Use 'transformer' or 'mlp'")

    def forward(self, features, coordinate, principal_dir, curvature, density, normals, linearity, task='class'):
        return self.model(features, coordinate, principal_dir, curvature, density, normals, linearity, task)


class TransformerModel(nn.Module):
    """原始的Transformer模型实现"""
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
        # Output: curvature (1 channel) - for block-wise pretraining
        # Enhanced with MLP structure for better capacity (similar to classifier)
        self.recon_head = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL * 2),
            nn.ReLU(),
            nn.Linear(config.D_MODEL * 2, config.D_MODEL),
            nn.ReLU(),
            nn.Linear(config.D_MODEL, 1)  # curvature only
        )

    def forward(self, features, coordinate, principal_dir, curvature, density, normals, linearity, task='class'):
        # project input features to model dimension
        x = self.input_proj(features)
        for block in self.attention_layers:
            x = block(
                x,
                coordinate=coordinate,
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