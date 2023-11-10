import torch

from models.v1 import PositionalMultiheadAttention


def test_forward_positional_multihead_attention():
    embeddings = torch.rand(128, 20, 64)

    positional_multihead_attention = PositionalMultiheadAttention(
        d_model=64,
        num_heads=8,
        dropout=0.1,
    )
    x, _ = positional_multihead_attention(embeddings)

    assert x.shape == (128, 20, 64)
