import torch
import torch.nn as nn
from src.models.components.positional import PositionalEncoding

def test_positional_encoding():
    batch_size, seq_len, d_model, max_len = 2, 10, 64, 20
    x = torch.rand(batch_size, seq_len, d_model)

    pos_encoder = PositionalEncoding(d_model, max_len, dropout=0.1)
    out = pos_encoder(x)

    # Test output shape
    assert out.shape == x.shape, \
        f"Expected shape {x.shape}, but got {out.shape}"

    # Test positional encoding buffer shape
    pe_shape = pos_encoder.pe.shape
    assert pe_shape == (1, max_len, d_model), \
        f"Expected PE shape {(1, max_len, d_model)}, but got {pe_shape}"

    # Test value range of positional encodings
    pe = pos_encoder.pe.squeeze(0).numpy()
    assert all(-1.0 <= val <= 1.0 + 1e-7 for row in pe for val in row), \
        "Positional encoding values should be in [-1, 1]"

    # Test uniqueness of positional encodings
    assert not torch.allclose(pos_encoder.pe[:, 0], pos_encoder.pe[:, 1], atol=1e-6), \
        "First two positional encodings should be different"

    # Test dropout layer
    assert isinstance(pos_encoder.dropout, nn.Dropout), \
        "Dropout layer should be an instance of nn.Dropout"

    # Test output dtype
    assert out.dtype == torch.float32, \
        f"Expected dtype torch.float32, but got {out.dtype}"

    print("PositionalEncoding tests passed!")

if __name__ == "__main__":
    test_positional_encoding()