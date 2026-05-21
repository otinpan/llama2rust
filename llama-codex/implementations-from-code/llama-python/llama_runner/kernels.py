from __future__ import annotations

import numpy as np


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean_square = np.mean(x * x, dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(mean_square + eps))
    return weight * (x * scale)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted).astype(np.float32, copy=False)
    return exp / np.sum(exp, dtype=np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x).astype(np.float32, copy=False))


def matmul(weight: np.ndarray, x: np.ndarray) -> np.ndarray:
    return weight @ x
