import flash_attn_minimal_slang as flash_attn
import torch.nn.functional as F
import torch
import numpy as np


def test_naive_attention():
    """test naive attention implementation"""
    batch_size = 2
    num_heads = 4
    num_query_tokens = 64
    num_key_tokens = 64
    query_dim = 32
    key_dim = 32
    value_dim = 32
    q = torch.randn(batch_size, num_heads, num_query_tokens, query_dim).to(
        torch.float32
    )
    k = torch.randn(batch_size, num_heads, num_key_tokens, key_dim).to(torch.float32)
    v = torch.randn(batch_size, num_heads, num_key_tokens, value_dim).to(torch.float32)

    mask = torch.rand(batch_size, 1, num_query_tokens, num_key_tokens) > 0.1

    out_expected = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=False
    ).numpy()

    out = flash_attn.attention.naive_attention(
        q.numpy().astype(np.float32),
        k.numpy().astype(np.float32),
        v.numpy().astype(np.float32),
        mask=mask.numpy().astype(np.bool_),
    )

    assert np.allclose(out_expected, out, atol=1e-5), (
        f"max abs err = {np.max(np.abs(out_expected - out)):.3e} in naive implementation"
    )
