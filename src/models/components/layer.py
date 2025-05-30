# src/models/components/layer.py
import torch.nn as nn
from .attention import MultiHeadSelfAttention
from .feedforward import FeedForward

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attention(x_norm, mask)
        x = x + self.dropout(attn_output)
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        return x, attn_weights