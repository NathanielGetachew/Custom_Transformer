import torch
from src.models.components.feedforward import FeedForward

def test_feedforward_shapes():
    batch_size, seq_len, d_model, dim_feedforward = 2, 10, 64, 256
    x = torch.rand(batch_size, seq_len, d_model)

    ffn = FeedForward(d_model, dim_feedforward, dropout=0.1)
    out = ffn(x)
    assert out.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, but got {out.shape}"
    print("FeedForward shape test passed!")

def test_feedforward_dtype():
    batch_size, seq_len, d_model, dim_feedforward = 2, 10, 64, 256
    x = torch.rand(batch_size, seq_len, d_model, dtype=torch.float32)

    ffn = FeedForward(d_model, dim_feedforward, dropout=0.1)
    out = ffn(x)
    assert out.dtype == x.dtype, \
        f"Expected dtype {x.dtype}, but got {out.dtype}"
    print("FeedForward dtype test passed!")

if __name__ == "__main__":
    test_feedforward_shapes()
    test_feedforward_dtype()