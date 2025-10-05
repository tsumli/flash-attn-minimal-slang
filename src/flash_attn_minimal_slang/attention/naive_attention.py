import numpy as np
import numpy.typing as npt


def softmax(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def naive_attention(
    q: npt.NDArray[np.float32],
    k: npt.NDArray[np.float32],
    v: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.float32]:
    """
    Args:
        q: query tensor of shape [B, H, Nq, D]
        k: key tensor of shape [B, H, Nk, D]
        v: value tensor of shape [B, H, Nk, Dv]
        mask: mask tensor of shape [B, 1, Nq, Nk] or [B, H, Nq, Nk] (True=keep/1, False=mask/0)
    """
    B, H, Nq, D = q.shape
    _, _, Nk, _ = k.shape
    scale = 1.0 / (D**0.5)

    k_transposed = k.transpose(0, 1, 3, 2)
    scores = np.matmul(q, k_transposed)
    scores = scores * scale

    if mask is not None:
        scores = np.where(mask, scores, float("-inf"))

    prob = softmax(scores)
    out = np.matmul(prob, v)
    return out
