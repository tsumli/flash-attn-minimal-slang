import flash_attn_minimal_slang as flash_attn
import torch.nn.functional as F
import torch
import numpy as np


def test_flash_attention():
    """test flash attention implementation"""
    batch_size = 32
    num_heads = 8
    num_query_tokens = 64
    num_key_tokens = 64
    query_dim = 64
    key_dim = 64
    value_dim = 64
    q = torch.randn(batch_size, num_heads, num_query_tokens, query_dim).to(
        torch.float32
    )
    k = torch.randn(batch_size, num_heads, num_key_tokens, key_dim).to(torch.float32)
    v = torch.randn(batch_size, num_heads, num_key_tokens, value_dim).to(torch.float32)
    # no mask
    mask = torch.ones(batch_size, 1, num_query_tokens, num_key_tokens).to(torch.bool)

    out_expected = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=False
    ).numpy()
    flash_attention = flash_attn.attention.FlashAttention(
        batch_size=batch_size,
        num_heads=num_heads,
        num_query_tokens=num_query_tokens,
        num_key_tokens=num_key_tokens,
        query_dim=query_dim,
        key_dim=key_dim,
        value_dim=value_dim,
    )

    out = flash_attention(
        q.numpy().astype(np.float32),
        k.numpy().astype(np.float32),
        v.numpy().astype(np.float32),
        mask=mask.numpy().astype(np.bool_),
    )

    assert np.allclose(out_expected, out, atol=1e-5), (
        f"max abs err = {np.max(np.abs(out_expected - out)):.3e} in flash implementation"
    )
