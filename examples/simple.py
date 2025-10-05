import numpy as np
import flash_attn_minimal_slang as flash_attn


def main():
    random_q = np.random.randn(2, 4, 64, 32).astype(np.float32)
    random_k = np.random.randn(2, 4, 64, 32).astype(np.float32)
    random_v = np.random.randn(2, 4, 64, 48).astype(np.float32)
    random_mask = (np.random.rand(2, 1, 64, 64) > 0.1).astype(np.bool_)

    flash_attention = flash_attn.attention.FlashAttention(
        batch_size=2,
        num_heads=4,
        num_query_tokens=64,
        num_key_tokens=64,
        query_dim=32,
        key_dim=32,
        value_dim=48,
    )

    out = flash_attention(random_q, random_k, random_v, random_mask)
    print(out.shape)
    print(out)


if __name__ == "__main__":
    main()
