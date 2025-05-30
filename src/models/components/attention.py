# src/models/components/attention.py
import torch
import torch.nn as nn
import math

class SelfAttentionHead(nn.Module):
    def __init__(self, d_model, head_size, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(head_size)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        Q = self.query(x)  # [B, T, head_size]
        K = self.key(x)    # [B, T, head_size]
        V = self.value(x)  # [B, T, head_size]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, T, T]
        if mask is not None:
            scores = scores + mask
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, V)  # [B, T, head_size]
        return out, attn_weights

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.head_size = d_model // nhead
        self.nhead = nhead
        self.heads = nn.ModuleList([
            SelfAttentionHead(d_model, self.head_size, dropout) for _ in range(nhead)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        head_outputs = []
        attn_weights = []
        for head in self.heads:
            out, weights = head(x, mask)
            head_outputs.append(out)
            attn_weights.append(weights)
        
        out = torch.cat(head_outputs, dim=-1)  # [B, T, d_model]
        out = self.dropout(self.proj(out))
        return out, torch.stack(attn_weights, dim=1)  # [B, nhead, T, T]