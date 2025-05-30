# tests/test_attention.py
import torch
from src.models.components.attention import SelfAttentionHead, MultiHeadSelfAttention

def test_attention_shapes():
    batch_size, seq_len, d_model, nhead = 2, 10, 64, 4
    head_size = d_model // nhead
    x = torch.rand(batch_size, seq_len, d_model)
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)

    head = SelfAttentionHead(d_model, head_size)
    out, weights = head(x, mask)
    assert out.shape == (batch_size, seq_len, head_size)
    assert weights.shape == (batch_size, seq_len, seq_len)

    mha = MultiHeadSelfAttention(d_model, nhead)
    out, weights = mha(x, mask)
    assert out.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, nhead, seq_len, seq_len)
    print("Attention shape tests passed!")

if __name__ == "__main__":
    test_attention_shapes()