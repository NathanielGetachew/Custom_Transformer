# src/models/transformer.py
import torch
import torch.nn as nn
from .components.positional import PositionalEncoding
from .components.layer import TransformerLayer
from .utils import init_weights
import math

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3, dim_feedforward=1024, max_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.fc_out.weight = self.embedding.weight  # Weight tying
        self.dropout = nn.Dropout(dropout)

        self.tgt_mask = self.generate_causal_mask(max_len).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.apply(init_weights)

    def generate_causal_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, x):
        B, T = x.shape
        mask = self.tgt_mask[:T, :T]
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        attn_weights = []
        for layer in self.layers:
            x, weights = layer(x, mask)
            attn_weights.append(weights)
        
        x = self.norm(x)
        logits = self.fc_out(x)
        return logits, attn_weights