# weld_seg_project/model/attention/side_gate_qwen.py 最简单的门控机制
import torch.nn as nn

class SideGating(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        gate_values = self.sigmoid(self.gate_proj(x))
        return gate_values