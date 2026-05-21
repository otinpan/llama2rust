from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import Config


@dataclass(slots=True)
class RunState:
    x: np.ndarray
    xb: np.ndarray
    xb2: np.ndarray
    hb: np.ndarray
    hb2: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    att: np.ndarray
    logits: np.ndarray
    key_cache: np.ndarray
    value_cache: np.ndarray

    @classmethod
    def create(cls, config: Config) -> "RunState":
        kv_dim = (config.dim * config.n_kv_heads) // config.n_heads
        zeros = np.zeros
        return cls(
            x=zeros((config.dim,), dtype=np.float32),
            xb=zeros((config.dim,), dtype=np.float32),
            xb2=zeros((config.dim,), dtype=np.float32),
            hb=zeros((config.hidden_dim,), dtype=np.float32),
            hb2=zeros((config.hidden_dim,), dtype=np.float32),
            q=zeros((config.dim,), dtype=np.float32),
            k=zeros((kv_dim,), dtype=np.float32),
            v=zeros((kv_dim,), dtype=np.float32),
            att=zeros((config.n_heads, config.seq_len), dtype=np.float32),
            logits=zeros((config.vocab_size,), dtype=np.float32),
            key_cache=zeros((config.n_layers, config.seq_len, kv_dim), dtype=np.float32),
            value_cache=zeros((config.n_layers, config.seq_len, kv_dim), dtype=np.float32),
        )
